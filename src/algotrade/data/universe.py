"""Universe membership utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from lxml import html as lxml_html

from ..config import AppSettings


@dataclass(slots=True)
class UniverseSnapshot:
    """Represents the active symbols for a specific effective date."""

    effective_date: date
    symbols: frozenset[str]


DEFAULT_SNP100_SOURCE = "https://en.wikipedia.org/wiki/S%26P_100"
DEFAULT_WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


def _validate_universe_frame(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = {"effective_date", "symbol"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
    df["symbol"] = df["symbol"].str.upper().str.strip()
    df = df[df["symbol"] != ""].dropna(subset=["symbol", "effective_date"])
    df.sort_values(["effective_date", "symbol"], inplace=True)
    return df


def _write_universe_frame(df: pd.DataFrame, settings: AppSettings, universe_name: str) -> Path:
    target_dir = settings.data_paths.raw / "universe" / universe_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "membership.parquet"
    df.to_parquet(target_path, index=False)
    return target_path


def ingest_universe_csv(
    source_csv: Path,
    settings: AppSettings | None = None,
    universe_name: str = "snp100",
) -> Path:
    """Ingest a point-in-time universe membership CSV into the cache.

    Parameters
    ----------
    source_csv:
        Path to a CSV with columns ``effective_date`` and ``symbol``.
    settings:
        Optional :class:`AppSettings`. If omitted, defaults are loaded from environment.
    universe_name:
        Folder name under ``data/universe``.

    Returns
    -------
    Path
        Location of the written Parquet file containing normalized membership data.
    """

    if not source_csv.exists():
        raise FileNotFoundError(source_csv)

    settings = settings or AppSettings()
    settings.data_paths.ensure()

    df = pd.read_csv(source_csv)
    return ingest_universe_frame(df, settings=settings, universe_name=universe_name)


def ingest_universe_frame(
    df: pd.DataFrame,
    settings: AppSettings | None = None,
    universe_name: str = "snp100",
) -> Path:
    """Persist a normalized DataFrame of universe memberships."""

    settings = settings or AppSettings()
    settings.data_paths.ensure()
    normalized = _validate_universe_frame(df)
    return _write_universe_frame(normalized, settings, universe_name)


def fetch_snp100_members(
    *,
    effective_date: date | None = None,
    source_url: str = DEFAULT_SNP100_SOURCE,
) -> pd.DataFrame:
    """Fetch the current S&P 100 membership from Wikipedia.

    Parses the *Components* section table so structural changes elsewhere on the page
    do not affect the result. Returns a DataFrame with columns ``effective_date`` and
    ``symbol``.
    """

    headers = {**DEFAULT_WIKIPEDIA_HEADERS, "Referer": source_url}
    response = requests.get(source_url, timeout=15, headers=headers)
    response.raise_for_status()

    document = lxml_html.fromstring(response.text)
    table_html = _extract_components_table_html(document)

    dataframes: list[pd.DataFrame] = []
    if table_html:
        dataframes = pd.read_html(StringIO(table_html))
    if not dataframes:
        # Fall back to scanning the entire document to maintain backwards compatibility.
        dataframes = pd.read_html(StringIO(response.text))

    table = _select_symbol_table(dataframes)
    if table is None:
        raise ValueError(
            "Unable to locate S&P 100 component table in source document")

    table = _normalize_component_table(table)
    eff_date = effective_date or date.today()
    table["effective_date"] = eff_date
    table = table[["effective_date", "symbol"]]
    return _validate_universe_frame(table)


def _extract_components_table_html(document: lxml_html.HtmlElement) -> str | None:
    components = document.xpath("//span[@id='Components']")
    if not components:
        return None

    heading = components[0].getparent()
    if heading is None:
        return None

    node = heading.getnext()
    while node is not None:
        tag = getattr(node, "tag", "")
        if not isinstance(tag, str):
            node = node.getnext()
            continue
        lowered = tag.lower()
        if lowered == "table":
            return lxml_html.tostring(node, encoding="unicode")
        if lowered in {"h2", "h3"}:
            break
        node = node.getnext()
    return None


def _select_symbol_table(tables: Iterable[pd.DataFrame]) -> pd.DataFrame | None:
    for table in tables:
        normalized_columns = [
            " ".join(col).strip().lower() if isinstance(
                col, tuple) else str(col).strip().lower()
            for col in table.columns
        ]
        if any("symbol" in column for column in normalized_columns):
            return table
    return None


def _normalize_component_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        " ".join(col).strip() if isinstance(col, tuple) else str(col).strip()
        for col in table.columns
    ]
    table = table.copy()
    table.columns = columns

    symbol_column = next(
        (column for column in table.columns if "symbol" in column.lower()),
        None,
    )
    if symbol_column is None:
        raise ValueError("Components table missing a symbol column")

    df = table[[symbol_column]].rename(columns={symbol_column: "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"])
    return df


def load_universe(target_path: Path) -> list[UniverseSnapshot]:
    """Load snapshots from the stored Parquet file."""

    df = pd.read_parquet(target_path)
    df = _validate_universe_frame(df)
    snapshots: list[UniverseSnapshot] = []
    for effective_date, group in df.groupby("effective_date"):
        snapshots.append(UniverseSnapshot(
            effective_date=effective_date, symbols=frozenset(group.symbol)))
    return snapshots


def latest_symbols(snapshots: Iterable[UniverseSnapshot]) -> frozenset[str]:
    """Return the most recent symbol set from snapshots."""

    snapshots = list(snapshots)
    if not snapshots:
        return frozenset()
    latest = max(snapshots, key=lambda snap: snap.effective_date)
    return latest.symbols
