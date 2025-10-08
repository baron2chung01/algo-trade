"""Data ingestion helpers for external market data providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

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
        Iterable of ticker symbols (e.g., ["AAPL", "MSFT"]).
    start:
        Start date (inclusive).
    end:
        End date (inclusive).
    settings:
        Optional application settings. If omitted, defaults will be loaded from environment.

    Returns
    -------
    list[Path]
        Paths to the Parquet files written for each symbol.
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


def update_polygon_daily_incremental(
    symbols: Iterable[str],
    *,
    settings: AppSettings | None = None,
    provider: PolygonDailyEquityProvider | None = None,
    store: ParquetBarStore | None = None,
    lookback_years: int = 5,
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

        df_new = provider.fetch_historical_bars(request)
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
