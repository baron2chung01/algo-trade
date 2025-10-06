#!/usr/bin/env python3
"""CLI utility to download QuantConnect daily bars into the local cache."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from algotrade.config import AppSettings
from algotrade.data.ingest import ingest_quantconnect_daily
from algotrade.data.universe import latest_symbols, load_universe


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:  # pragma: no cover - argparse error path
        raise argparse.ArgumentTypeError(f"Invalid date format: {value}") from exc


def _resolve_universe_path(settings: AppSettings, name: str | None, override: Path | None) -> Path:
    if override:
        return override
    if not name:
        raise ValueError("Universe name or explicit file path must be provided.")
    return settings.data_paths.raw / "universe" / name / "membership.parquet"


def _load_universe_symbols(path: Path, effective: date | None) -> list[str]:
    snapshots = load_universe(path)
    if not snapshots:
        raise ValueError(f"Universe file {path} contains no records.")
    if effective is not None:
        for snapshot in snapshots:
            if snapshot.effective_date == effective:
                return sorted(snapshot.symbols)
        raise ValueError(f"Effective date {effective} not found in universe {path}.")
    return sorted(latest_symbols(snapshots))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QuantConnect daily bars into Parquet cache.")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to download (e.g., AAPL MSFT)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). Defaults to --end minus --years.")
    parser.add_argument("--end", help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--years", type=int, default=3, help="Lookback window in years when --start omitted (default: 3).")
    parser.add_argument("--universe", help="Universe name under data/raw/universe to expand symbols (e.g., snp100).")
    parser.add_argument("--universe-file", type=Path, help="Explicit path to universe membership Parquet file.")
    parser.add_argument("--effective-date", help="Filter universe membership to a specific effective date (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data without writing Parquet files.")

    args = parser.parse_args()
    if not args.symbols and not args.universe and not args.universe_file:
        parser.error("Provide ticker symbols or --universe/--universe-file to determine the download set.")
    return args


def resolve_symbols(args: argparse.Namespace, settings: AppSettings) -> list[str]:
    effective: date | None = _parse_date(args.effective_date) if args.effective_date else None
    if args.universe or args.universe_file:
        universe_path = _resolve_universe_path(settings, args.universe, args.universe_file)
        return _load_universe_symbols(universe_path, effective)
    return [symbol.upper() for symbol in args.symbols]


def compute_date_range(args: argparse.Namespace) -> tuple[date, date]:
    end_date = _parse_date(args.end) if args.end else date.today()
    if args.start:
        start_date = _parse_date(args.start)
    else:
        start_date = end_date - timedelta(days=365 * args.years)
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    return start_date, end_date


def main() -> None:
    args = parse_args()
    settings = AppSettings()
    symbols = resolve_symbols(args, settings)
    start_date, end_date = compute_date_range(args)

    result = ingest_quantconnect_daily(
        symbols,
        start_date,
        end_date,
        settings=settings,
        write=not args.dry_run,
    )

    if args.dry_run:
        print("Dry run: fetched bars without writing to disk")
        for symbol, df in result.frames.items():
            print(f"  {symbol}: {len(df)} bars")
    else:
        print("Written Parquet files:")
        for path in result.paths:
            print(f"  {path}")

    skipped = set(symbols) - set(result.frames)
    if skipped:
        print("Symbols with no data returned:")
        for symbol in sorted(skipped):
            print(f"  {symbol}")

    print(f"Processed {len(result.frames)} symbols from {start_date} to {end_date}.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
