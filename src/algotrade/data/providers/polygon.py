"""Polygon data provider for free tier end-of-day equity bars."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import time
from typing import Optional

import pandas as pd
import requests

from ..contracts import ContractSpec
from ..schemas import IBKRBar, IBKRBarDataFrame
from .base import BaseDataProvider, HistoricalDataRequest

_SUPPORTED_BAR_SIZES = {"1 day", "1d", "1D"}


def _parse_duration(duration: str) -> timedelta:
    parts = duration.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unsupported duration format: {duration}")
    amount_str, unit = parts
    if not amount_str.isdigit():
        raise ValueError(f"Unsupported duration amount: {amount_str}")
    amount = int(amount_str)
    unit = unit.upper()
    if unit not in {"D", "W", "M", "Y"}:
        raise ValueError(f"Unsupported duration unit: {unit}")
    unit_to_days = {"D": 1, "W": 7, "M": 30, "Y": 365}
    return timedelta(days=amount * unit_to_days[unit])


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _PolygonAPI:
    """Thin wrapper around Polygon Aggregates API with rate limiting."""

    def __init__(
            self,
            api_key: str,
            *,
            session: Optional[requests.Session] = None,
            base_url: str = "https://api.polygon.io",
            rate_limit_per_minute: int = 5,
            timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout = timeout
        self._call_timestamps: deque[float] = deque()

    def close(self) -> None:
        self.session.close()

    def _throttle(self) -> None:
        if self.rate_limit_per_minute <= 0:
            return
        now = time.monotonic()
        window = 60.0
        while self._call_timestamps and now - self._call_timestamps[0] > window:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self.rate_limit_per_minute:
            sleep_time = window - (now - self._call_timestamps[0]) + 0.01
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._throttle()
        else:
            self._call_timestamps.append(now)

    def _request(self, url: str, params: Optional[dict] = None) -> dict:
        self._throttle()
        params = dict(params or {})
        params.setdefault("apiKey", self.api_key)
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ERROR":
            message = payload.get("error") or payload.get(
                "message") or "Polygon request failed"
            raise RuntimeError(message)
        return payload

    def aggregates(
            self,
            ticker: str,
            *,
            multiplier: int,
            timespan: str,
            start: datetime,
            end: datetime,
            adjusted: bool = True,
    ) -> list[dict]:
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
        }
        endpoint = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
            f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        )
        payload = self._request(endpoint, params=params)
        results = list(payload.get("results", []) or [])
        next_url = payload.get("next_url")
        while next_url:
            payload = self._request(next_url)
            results.extend(payload.get("results", []) or [])
            next_url = payload.get("next_url")
        return results


class PolygonDailyEquityProvider(BaseDataProvider):
    """Fetch daily OHLCV bars from Polygon.io and normalize to IBKR schema."""

    def __init__(
            self,
            api_key: str,
            *,
            session: Optional[requests.Session] = None,
            base_url: str = "https://api.polygon.io",
            rate_limit_per_minute: int = 5,
    ) -> None:
        if not api_key:
            raise ValueError("PolygonDailyEquityProvider requires an api_key.")
        self.api_key = api_key
        self._api = _PolygonAPI(
            api_key,
            session=session,
            base_url=base_url,
            rate_limit_per_minute=rate_limit_per_minute,
        )

    def close(self) -> None:
        self._api.close()

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        if request.bar_size not in _SUPPORTED_BAR_SIZES:
            raise ValueError(
                f"PolygonDailyEquityProvider supports only daily bars: got {request.bar_size}"
            )

        end_dt = _ensure_timezone(request.end)
        start_dt = end_dt - _parse_duration(request.duration)
        if start_dt > end_dt:
            raise ValueError(
                "Start date computed from duration exceeds end date.")

        results = self._api.aggregates(
            request.contract.symbol.upper(),
            multiplier=1,
            timespan="day",
            start=start_dt,
            end=end_dt,
        )

        if not results:
            return IBKRBarDataFrame.from_bars([])

        bars: list[IBKRBar] = []
        for item in results:
            timestamp = datetime.fromtimestamp(
                item["t"] / 1000, tz=timezone.utc)
            bars.append(
                IBKRBar(
                    timestamp=timestamp,
                    open=float(item.get("o", 0.0)),
                    high=float(item.get("h", 0.0)),
                    low=float(item.get("l", 0.0)),
                    close=float(item.get("c", 0.0)),
                    volume=float(item.get("v", 0.0)),
                    average=float(item.get("vw")) if item.get(
                        "vw") is not None else None,
                    bar_count=int(item.get("n", 0)),
                )
            )

        return IBKRBarDataFrame.from_bars(bars)


__all__ = ["PolygonDailyEquityProvider"]
