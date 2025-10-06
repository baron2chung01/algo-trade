"""CLI utility to ingest point-in-time universe membership snapshots."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from algotrade.config import AppSettings
from algotrade.data.universe import (
    DEFAULT_SNP100_SOURCE,
    fetch_snp100_members,
    ingest_universe_csv,
    ingest_universe_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a universe membership snapshot into the local cache.")
    parser.add_argument("csv", nargs="?", type=Path, help="Path to CSV with effective_date,symbol columns")
    parser.add_argument(
        "--name",
        default="snp100",
        help="Folder name under data/raw/universe where the parquet file will be written (default: snp100)",
    )
    parser.add_argument(
        "--fetch-snp100",
        action="store_true",
        help="Download the latest S&P 100 membership from Wikipedia instead of reading a CSV.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SNP100_SOURCE,
        help="Override the S&P 100 source URL (used with --fetch-snp100).",
    )
    parser.add_argument(
        "--effective-date",
        help="Effective date for fetched memberships (YYYY-MM-DD). Defaults to today's date.",
    )
    args = parser.parse_args()
    if not args.fetch_snp100 and args.csv is None:
        parser.error("CSV path is required unless --fetch-snp100 is specified.")
    return args


def main() -> None:
    args = parse_args()
    settings = AppSettings()
    if args.fetch_snp100:
        effective = date.fromisoformat(args.effective_date) if args.effective_date else date.today()
        df = fetch_snp100_members(effective_date=effective, source_url=args.source_url)
        target = ingest_universe_frame(df, settings=settings, universe_name=args.name)
    else:
        if args.csv is None:  # defensive, parse_args should enforce
            raise SystemExit("CSV path is required unless --fetch-snp100 is specified.")
        target = ingest_universe_csv(args.csv, settings=settings, universe_name=args.name)
    print(f"Universe ingested to {target}")


if __name__ == "__main__":
    main()
