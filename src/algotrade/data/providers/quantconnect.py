"""QuantConnect data provider producing IBKR-compatible bar data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from ..contracts import ContractSpec
from ..schemas import IBKRBar, IBKRBarDataFrame
from .base import BaseDataProvider, HistoricalDataRequest

_QC_BASE_URL = "https://www.quantconnect.com/api/v2/data/read"
_SUPPORTED_BAR_SIZES = {"1 day", "1d", "1D"}
_DURATION_UNITS = {
    "D": 1,
    "W": 7,
    "M": 30,
    "Y": 365,
}


def _parse_duration(duration: str) -> timedelta:
    parts = duration.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unsupported duration format: {duration}")
    amount_str, unit = parts
    if not amount_str.isdigit():
        raise ValueError(f"Unsupported duration amount: {amount_str}")
    amount = int(amount_str)
    unit = unit.upper()
    if unit not in _DURATION_UNITS:
        raise ValueError(f"Unsupported duration unit: {unit}")
    return timedelta(days=amount * _DURATION_UNITS[unit])


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_timestamp(value: object, default_tz: timezone) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=default_tz)
        return value.astimezone(default_tz)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=default_tz)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=default_tz)
            return dt.astimezone(default_tz)
        except ValueError:
            pass
        # QuantConnect Lean data often formats as YYYYMMDD HH:MM
        try:
            dt = datetime.strptime(value, "%Y%m%d %H:%M")
            return dt.replace(tzinfo=default_tz)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unable to parse timestamp: {value}") from exc
    raise TypeError(f"Unsupported timestamp value: {value!r}")


def _to_quantconnect_symbol(contract: ContractSpec) -> str:
    return contract.symbol.upper()


class QuantConnectDailyEquityProvider(BaseDataProvider):
    """Fetch daily OHLCV bars from QuantConnect and normalize to IBKR schema."""

    def __init__(
        self,
        user_id: str,
        api_token: str,
        *,
        session: Optional[requests.Session] = None,
        base_url: str = _QC_BASE_URL,
    ) -> None:
        if not user_id:
            raise ValueError("QuantConnectDailyEquityProvider requires a user_id.")
        if not api_token:
            raise ValueError("QuantConnectDailyEquityProvider requires an api_token.")
        self.user_id = user_id
        self.api_token = api_token
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        if request.bar_size not in _SUPPORTED_BAR_SIZES:
            raise ValueError(
                f"QuantConnectDailyEquityProvider supports only daily bars: got {request.bar_size}"
            )

        end_dt = _ensure_timezone(request.end)
        start_dt = end_dt - _parse_duration(request.duration)
        params = {
            "userId": self.user_id,
            "apiToken": self.api_token,
            "ticker": _to_quantconnect_symbol(request.contract),
            "securityType": "Equity",
            "market": "usa",
            "resolution": "Daily",
            "start": start_dt.strftime("%Y%m%d"),
            "end": end_dt.strftime("%Y%m%d"),
            "dataFormat": "LeanCSV",
        }

        response = self.session.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict) and not payload.get("success", True):
            message = payload.get("errors") or payload.get("message") or "QuantConnect data request failed"
            raise RuntimeError(message)

        data = payload.get("data", []) if isinstance(payload, dict) else []
        bars: list[IBKRBar] = []
        for item in data:
            timestamp = _coerce_timestamp(item.get("time"), timezone.utc)
            bars.append(
                IBKRBar(
                    timestamp=timestamp,
                    open=float(item.get("open", 0.0)),
                    high=float(item.get("high", 0.0)),
                    low=float(item.get("low", 0.0)),
                    close=float(item.get("close", 0.0)),
                    volume=float(item.get("volume", 0.0)),
                    average=float(item.get("vwap")) if item.get("vwap") is not None else None,
                    bar_count=int(item.get("trades", 0)),
                )
            )

        return IBKRBarDataFrame.from_bars(bars)

    def close(self) -> None:
        self.session.close()
