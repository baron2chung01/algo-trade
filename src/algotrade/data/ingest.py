"""Data ingestion helpers for external market data providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd
import requests
import time

from ..config import AppSettings
from .contracts import ContractSpec
from .providers.base import HistoricalDataRequest
from .providers.polygon import PolygonDailyEquityProvider
from .providers.quantconnect import QuantConnectDailyEquityProvider
from .schemas import IBKRBarDataFrame, IBKR_BAR_COLUMNS
from .stores.local import ParquetBarStore


def _duration_from_dates(start: date, end: date) -> str:
    if end < start:
        raise ValueError("End date must be on or after start date.")
    days = (end - start).days or 1
    return f"{days} D"


@dataclass(slots=True)
class IngestResult:
    """Result of an ingestion job."""

    paths: list[Path] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Path]:  # pragma: no cover - simple passthrough
        return iter(self.paths)

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return len(self.paths)


@dataclass(slots=True)
class IncrementalUpdateReport:
    """Summary of an incremental data refresh."""

    updated: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    written_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _RateLimiter:
    """Simple per-minute request rate limiter."""

    def __init__(
        self,
        max_requests_per_minute: int,
        *,
        window_seconds: float = 60.0,
        now: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._max_requests = max_requests_per_minute
        self._window_seconds = window_seconds
        self._now = now or time.monotonic
        self._sleep = sleeper or time.sleep
        self._window_start = self._now()
        self._count = 0

    def _reset_window(self, current_time: float) -> None:
        self._window_start = current_time
        self._count = 0

    def before_request(self) -> None:
        if self._max_requests <= 0:
            return

        current_time = self._now()
        elapsed = current_time - self._window_start
        if elapsed >= self._window_seconds:
            self._reset_window(current_time)
            return

        if self._count >= self._max_requests:
            wait_for = self._window_seconds - elapsed
            if wait_for > 0:
                self._sleep(wait_for)
            current_time = self._now()
            self._reset_window(current_time)

    def record_request(self) -> None:
        if self._max_requests <= 0:
            return
        self._count += 1


def ingest_quantconnect_daily(
    symbols: Iterable[str],
    start: date,
    end: date,
    settings: AppSettings | None = None,
    provider: QuantConnectDailyEquityProvider | None = None,
    store: ParquetBarStore | None = None,
    *,
    write: bool = True,
) -> IngestResult:
    """Download QuantConnect daily bars and store them as Parquet files.

    Parameters
    ----------
    symbols:
        Iterable of ticker symbols to download.
    start:
        Start date (inclusive).
    end:
        End date (inclusive).
    settings:
        Optional application settings. If omitted, defaults will be loaded from environment.
    provider:
        Optional provider instance. When omitted a new provider is created and closed after use.
    store:
        Optional bar store. When omitted data is written to the default raw data directory.
    write:
        When ``False``, skip writing frames to disk and only return them in memory.

    Returns
    -------
    IngestResult
        Summary of written paths and in-memory frames.
    """

    settings = settings or AppSettings()
    settings.data_paths.ensure()
    credentials = settings.require_quantconnect_credentials()

    created_provider = provider is None
    provider = provider or QuantConnectDailyEquityProvider(
        user_id=credentials.user_id,
        api_token=credentials.api_token,
        organization_id=credentials.organization_id,
    )
    store_root = settings.data_paths.raw / "quantconnect" / "daily"
    store = store if store is not None else (
        ParquetBarStore(store_root) if write else None)

    result = IngestResult()
    end_dt = datetime.combine(end, datetime.min.time())
    duration = _duration_from_dates(start, end)

    for symbol in symbols:
        normalized_symbol = symbol.upper()
        request = HistoricalDataRequest(
            contract=ContractSpec(symbol=normalized_symbol),
            end=end_dt,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )
        df = provider.fetch_historical_bars(request)
        if df.empty:
            continue
        result.frames[normalized_symbol] = df
        if store is not None:
            result.paths.append(store.save(normalized_symbol, "1d", df))

    if created_provider:
        provider.close()
    return result


def ingest_polygon_daily(
    symbols: Iterable[str],
    start: date,
    end: date,
    settings: AppSettings | None = None,
    provider: PolygonDailyEquityProvider | None = None,
    store: ParquetBarStore | None = None,
    *,
    write: bool = True,
    rate_limit_per_minute: int | None = 5,
    _rate_limiter: _RateLimiter | None = None,
) -> IngestResult:
    """Download Polygon daily bars and store them as Parquet files."""

    settings = settings or AppSettings()
    settings.data_paths.ensure()
    api_key = settings.require_polygon_api_key()

    created_provider = provider is None
    provider = provider or PolygonDailyEquityProvider(api_key=api_key)
    store_root = settings.data_paths.raw / "polygon" / "daily"
    store = store if store is not None else (
        ParquetBarStore(store_root) if write else None)

    result = IngestResult()
    end_dt = datetime.combine(end, datetime.min.time()
                              ).replace(tzinfo=timezone.utc)
    duration = _duration_from_dates(start, end)
    rate_limiter = _rate_limiter
    if rate_limiter is None and rate_limit_per_minute and rate_limit_per_minute > 0:
        rate_limiter = _RateLimiter(rate_limit_per_minute)

    for symbol in symbols:
        normalized_symbol = symbol.upper()
        request = HistoricalDataRequest(
            contract=ContractSpec(symbol=normalized_symbol),
            end=end_dt,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )
        if rate_limiter is not None:
            rate_limiter.before_request()
        try:
            df = provider.fetch_historical_bars(request)
        finally:
            if rate_limiter is not None:
                rate_limiter.record_request()
        if df.empty:
            continue
        result.frames[normalized_symbol] = df
        if store is not None:
            result.paths.append(store.save(normalized_symbol, "1d", df))

    if created_provider:
        provider.close()
    return result


def update_polygon_daily_incremental(
    symbols: Iterable[str],
    *,
    settings: AppSettings | None = None,
    provider: PolygonDailyEquityProvider | None = None,
    store: ParquetBarStore | None = None,
    lookback_years: int = 5,
    rate_limit_per_minute: int | None = 5,
    _rate_limiter: _RateLimiter | None = None,
) -> IncrementalUpdateReport:
    """Update Polygon daily bars by fetching only missing rows for each symbol.

    Parameters
    ----------
    symbols:
        Iterable of ticker symbols to refresh.
    settings:
        Optional :class:`AppSettings`. Uses defaults when omitted.
    lookback_years:
        Number of years of history to seed when a symbol has no cached data.

    Returns
    -------
    IncrementalUpdateReport
        Summary of symbols updated, skipped, and any warnings encountered.
    """

    settings = settings or AppSettings()
    settings.data_paths.ensure()
    api_key = settings.require_polygon_api_key()

    created_provider = provider is None
    provider = provider or PolygonDailyEquityProvider(api_key=api_key)

    store_root = settings.data_paths.raw / "polygon" / "daily"
    store = store if store is not None else ParquetBarStore(store_root)

    fallback_root = Path.home() / ".algo-trade" / "polygon" / "daily"
    fallback_store: ParquetBarStore | None = None

    report = IncrementalUpdateReport()
    today = date.today()
    end_dt = datetime.combine(today, datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    rate_limiter = _rate_limiter
    if rate_limiter is None and rate_limit_per_minute and rate_limit_per_minute > 0:
        rate_limiter = _RateLimiter(rate_limit_per_minute)

    for raw_symbol in symbols:
        symbol = raw_symbol.upper().strip()
        if not symbol:
            continue

        existing_count = 0
        try:
            existing = store.load(symbol, "1d")
            existing_count = len(existing)
        except FileNotFoundError:
            if fallback_store is None:
                fallback_store = ParquetBarStore(fallback_root)
            try:
                existing = fallback_store.load(symbol, "1d")
                existing_count = len(existing)
            except FileNotFoundError:
                existing = pd.DataFrame(columns=IBKR_BAR_COLUMNS)
                existing_count = 0

        if existing.empty:
            start_date = today - timedelta(days=lookback_years * 365)
        else:
            last_ts = pd.to_datetime(existing["timestamp"], utc=True).max()
            if pd.isna(last_ts):
                start_date = today - timedelta(days=lookback_years * 365)
            else:
                start_date = last_ts.date() + timedelta(days=1)

        if start_date > today:
            report.skipped.append(symbol)
            continue

        duration = _duration_from_dates(start_date, today)
        request = HistoricalDataRequest(
            contract=ContractSpec(symbol=symbol),
            end=end_dt,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )

        if rate_limiter is not None:
            rate_limiter.before_request()
        try:
            df_new = provider.fetch_historical_bars(request)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            reason = exc.response.reason if exc.response is not None else ""
            report.skipped.append(symbol)
            if status_code == 429:
                report.warnings.append(
                    "Polygon rate limit reached while downloading daily history. Wait about a minute and try again."
                )
                break

            sanitized_reason = reason or ""
            message = f"Polygon request failed for {symbol}"
            if status_code is not None:
                message += f" (HTTP {status_code})"
            if sanitized_reason:
                message += f": {sanitized_reason}"
            report.warnings.append(message)
            continue
        except requests.RequestException as exc:
            report.skipped.append(symbol)
            report.warnings.append(
                f"Network error while downloading {symbol} history from Polygon: {exc}"
            )
            continue
        except RuntimeError as exc:
            report.skipped.append(symbol)
            report.warnings.append(
                f"Polygon returned an error for {symbol}: {exc}"
            )
            continue
        finally:
            if rate_limiter is not None:
                rate_limiter.record_request()

        if df_new.empty:
            report.skipped.append(symbol)
            continue

        frames: list[pd.DataFrame] = []
        if not existing.empty:
            frames.append(existing)
        frames.append(IBKRBarDataFrame.ensure_schema(df_new))
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(
            subset=["timestamp"], keep="last", inplace=True)
        combined.sort_values("timestamp", inplace=True)
        combined = IBKRBarDataFrame.ensure_schema(combined)

        new_count = len(combined)
        if new_count <= existing_count:
            report.skipped.append(symbol)
            continue

        try:
            path = store.save(symbol, "1d", combined)
        except OSError as exc:
            if fallback_store is None:
                fallback_store = ParquetBarStore(fallback_root)
            try:
                path = fallback_store.save(symbol, "1d", combined)
                report.warnings.append(
                    f"Primary store unavailable for {symbol}; wrote to fallback at {path}: {exc}"
                )
            except OSError as fallback_exc:
                report.warnings.append(
                    f"Failed to persist {symbol} data to fallback store: {fallback_exc}"
                )
                continue

        new_rows = new_count - existing_count
        report.updated[symbol] = new_rows
        report.written_paths.append(path)

    if created_provider:
        provider.close()

    return report
