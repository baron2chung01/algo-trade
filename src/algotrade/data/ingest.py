"""Data ingestion helpers for external market data providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from ..config import AppSettings
from .contracts import ContractSpec
from .providers.base import HistoricalDataRequest
from .providers.polygon import PolygonDailyEquityProvider
from .providers.quantconnect import QuantConnectDailyEquityProvider
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
