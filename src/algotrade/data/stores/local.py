"""Local persistence helpers for historical bar data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schemas import IBKRBarDataFrame


class ParquetBarStore:
    """Read/write helper that persists IBKR-formatted bars as Parquet files."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, bar_size: str) -> Path:
        filename = f"{symbol}_{bar_size.replace(' ', '').lower()}.parquet"
        return self.root / filename

    def save(self, symbol: str, bar_size: str, df: pd.DataFrame) -> Path:
        validated = IBKRBarDataFrame.ensure_schema(df)
        path = self.path_for(symbol, bar_size)
        validated.to_parquet(path, engine="pyarrow")
        return path

    def load(self, symbol: str, bar_size: str) -> pd.DataFrame:
        path = self.path_for(symbol, bar_size)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, engine="pyarrow")
        return IBKRBarDataFrame.ensure_schema(df)

    def bulk_save(self, records: Iterable[tuple[str, str, pd.DataFrame]]) -> list[Path]:
        written: list[Path] = []
        for symbol, bar_size, df in records:
            written.append(self.save(symbol, bar_size, df))
        return written
