"""Data feed abstractions for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Iterator

import pandas as pd

from ..data.schemas import IBKRBar
from ..data.stores.local import ParquetBarStore


@dataclass(slots=True)
class BarSlice:
    """Collection of bars aligned on a single timestamp."""

    timestamp: datetime
    bars: Dict[str, IBKRBar]


class LocalDailyBarFeed:
    """Load daily bars from the local Parquet store and iterate chronologically."""

    def __init__(
        self,
        store: ParquetBarStore,
        symbols: Iterable[str],
        bar_size: str = "1d",
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        self.store = store
        self.symbols = [symbol.upper() for symbol in symbols]
        self.bar_size = bar_size
        self.start = start
        self.end = end
        self._frames: Dict[str, pd.DataFrame] = {}
        self._load_frames()

    def _load_frames(self) -> None:
        for symbol in self.symbols:
            try:
                df = self.store.load(symbol, self.bar_size)
            except FileNotFoundError:
                continue
            frame = df.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            if self.start is not None:
                frame = frame[frame["timestamp"].dt.date >= self.start]
            if self.end is not None:
                frame = frame[frame["timestamp"].dt.date <= self.end]
            if frame.empty:
                continue
            frame.set_index("timestamp", inplace=True)
            self._frames[symbol] = frame

    def __iter__(self) -> Iterator[BarSlice]:
        if not self._frames:
            return iter(())
        # Merge all timestamps across symbols
        combined_index = sorted({ts for frame in self._frames.values() for ts in frame.index})
        for ts in combined_index:
            bars: Dict[str, IBKRBar] = {}
            for symbol, frame in self._frames.items():
                if ts in frame.index:
                    row = frame.loc[ts]
                    if isinstance(row, pd.DataFrame):  # pragma: no cover - duplicate timestamps safeguard
                        row = row.iloc[0]
                    payload = row.to_dict()
                    payload["timestamp"] = ts.to_pydatetime()
                    bars[symbol] = IBKRBar(**payload)
            if bars:
                yield BarSlice(timestamp=ts, bars=bars)

    def __len__(self) -> int:  # pragma: no cover - not essential but handy for tests
        return sum(len(frame) for frame in self._frames.values())
