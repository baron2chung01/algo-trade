"""Polygon.io data provider producing IBKR-compatible bar data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from ..contracts import ContractSpec
from ..schemas import IBKRBar, IBKRBarDataFrame
from .base import BaseDataProvider, HistoricalDataRequest

# Polygon aggregates endpoint parameters
_POLYGON_BASE_URL = "https://api.polygon.io"
_SUPPORTED_BAR_SIZES = {"1 day", "1d", "1D"}
_UNITS_IN_DAYS = {
    "D": 1,
    "W": 7,
    "M": 30,
    "Y": 365,
}


def _parse_duration(duration: str) -> timedelta:
    """Approximate the number of days represented by an IB-style duration string."""

    parts = duration.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unsupported duration format: {duration}")
    amount_str, unit = parts
    if not amount_str.isdigit():
        raise ValueError(f"Unsupported duration amount: {amount_str}")
    amount = int(amount_str)
    unit = unit.upper()
    if unit not in _UNITS_IN_DAYS:
        raise ValueError(f"Unsupported duration unit: {unit}")
    return timedelta(days=amount * _UNITS_IN_DAYS[unit])


def _to_polygon_symbol(contract: ContractSpec) -> str:
    """Return the Polygon ticker for the provided contract."""

    # For US equities Polygon expects e.g. "AAPL" or if prefixed exchange like "IEX:AAPL".
    # We assume SMART routed stocks/ETFs listed in the US, so the raw symbol is acceptable.
    return contract.symbol.upper()


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class PolygonDailyBarsProvider(BaseDataProvider):
    """Fetch daily OHLCV bars from Polygon and normalize to IBKR schema."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, base_url: str = _POLYGON_BASE_URL):
        if not api_key:
            raise ValueError("PolygonDailyBarsProvider requires an API key.")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        if request.bar_size not in _SUPPORTED_BAR_SIZES:
            raise ValueError(f"PolygonDailyBarsProvider supports only daily bars: got {request.bar_size}")

        end_dt = _ensure_timezone(request.end)
        start_dt = end_dt - _parse_duration(request.duration)
        # Polygon requires YYYY-MM-DD date strings; include both endpoints.
        start_str = start_dt.date().isoformat()
        end_str = end_dt.date().isoformat()

        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": self.api_key,
        }
        ticker = _to_polygon_symbol(request.contract)
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", []) if isinstance(payload, dict) else []
        bars: list[IBKRBar] = []
        for item in results:
            timestamp = datetime.fromtimestamp(item["t"] / 1_000, tz=timezone.utc)
            bars.append(
                IBKRBar(
                    timestamp=timestamp,
                    open=float(item.get("o", 0.0)),
                    high=float(item.get("h", 0.0)),
                    low=float(item.get("l", 0.0)),
                    close=float(item.get("c", 0.0)),
                    volume=float(item.get("v", 0.0)),
                    average=float(item.get("vw")) if item.get("vw") is not None else None,
                    bar_count=int(item.get("n", 0)),
                )
            )

        df = IBKRBarDataFrame.from_bars(bars)
        if df.empty:
            return df
        return df

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()
