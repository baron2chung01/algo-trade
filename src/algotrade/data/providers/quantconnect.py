"""QuantConnect data provider producing IBKR-compatible bar data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import time
from typing import Optional

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from ..contracts import ContractSpec
from ..schemas import IBKRBar, IBKRBarDataFrame
from .base import BaseDataProvider, HistoricalDataRequest

_QC_BASE_API = "https://www.quantconnect.com/api/v2"
_QC_DATA_ENDPOINT = "data/read"
_SUPPORTED_BAR_SIZES = {"1 day", "1d", "1D"}
_DURATION_UNITS = {
    "D": 1,
    "W": 7,
    "M": 30,
    "Y": 365,
}


class _QuantConnectAPI:
    """Lightweight authenticated client for QuantConnect API v2."""

    def __init__(
        self,
        user_id: str,
        api_token: str,
        *,
        session: Optional[requests.Session] = None,
        base_url: str = _QC_BASE_API,
    ) -> None:
        self.user_id = str(user_id)
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _auth(self) -> tuple[dict[str, str], HTTPBasicAuth]:
        timestamp = str(int(time.time()))
        hashed_token = hashlib.sha256(
            f"{self.api_token}:{timestamp}".encode("utf-8")).hexdigest()
        headers = {"Timestamp": timestamp}
        auth = HTTPBasicAuth(self.user_id, hashed_token)
        return headers, auth

    def post(self, endpoint: str, *, json: Optional[dict] = None, timeout: int = 30) -> dict:
        headers, auth = self._auth()
        response = self.session.post(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            json=json,
            headers=headers,
            auth=auth,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()


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
        base_url: str = _QC_BASE_API,
        organization_id: str | None = None,
    ) -> None:
        if not user_id:
            raise ValueError(
                "QuantConnectDailyEquityProvider requires a user_id.")
        if not api_token:
            raise ValueError(
                "QuantConnectDailyEquityProvider requires an api_token.")
        self.user_id = str(user_id)
        self.api_token = api_token
        self._organization_id = organization_id
        self._api = _QuantConnectAPI(
            self.user_id,
            self.api_token,
            session=session,
            base_url=base_url,
        )

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        if request.bar_size not in _SUPPORTED_BAR_SIZES:
            raise ValueError(
                f"QuantConnectDailyEquityProvider supports only daily bars: got {request.bar_size}"
            )

        end_dt = _ensure_timezone(request.end)
        start_dt = end_dt - _parse_duration(request.duration)
        payload = self._api.post(
            _QC_DATA_ENDPOINT,
            json={
                "ticker": _to_quantconnect_symbol(request.contract),
                "securityType": "Equity",
                "market": "usa",
                "resolution": "Daily",
                "start": start_dt.strftime("%Y%m%d"),
                "end": end_dt.strftime("%Y%m%d"),
                "dataFormat": "LeanCSV",
                "organizationId": self._resolve_organization_id(),
            },
        )

        if isinstance(payload, dict) and not payload.get("success", True):
            message = payload.get("errors") or payload.get(
                "message") or "QuantConnect data request failed"
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
                    average=float(item.get("vwap")) if item.get(
                        "vwap") is not None else None,
                    bar_count=int(item.get("trades", 0)),
                )
            )

        return IBKRBarDataFrame.from_bars(bars)

    def close(self) -> None:
        self._api.close()

    def _resolve_organization_id(self) -> str:
        if self._organization_id:
            return self._organization_id
        account = self._api.post("account/read")
        organization_id = account.get("organizationId")
        if not organization_id:
            raise RuntimeError(
                "Unable to resolve QuantConnect organization id; ensure your account has an active organization."
            )
        self._organization_id = organization_id
        return organization_id
