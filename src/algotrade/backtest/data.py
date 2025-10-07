"""Data feed abstractions for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Iterator

import pandas as pd
from exchange_calendars import ExchangeCalendar, get_calendar

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
        *,
        calendar: ExchangeCalendar | None = None,
        calendar_name: str = "XNYS",
    ) -> None:
        self.store = store
        self.symbols = [symbol.upper() for symbol in symbols]
        self.bar_size = bar_size
        self.start = start
        self.end = end
        self.calendar = calendar or get_calendar(calendar_name)
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
            frame.index = frame.index.tz_convert("UTC")
            session_mask = self._session_filter(frame.index)
            frame = frame[session_mask]
            self._frames[symbol] = frame

    def _session_filter(self, index: pd.DatetimeIndex) -> pd.Series:
        if index.empty:
            return pd.Series(dtype=bool, index=index)

        normalized = index.tz_convert("UTC").normalize()
        start_label = normalized.min()
        end_label = normalized.max()

        if self.start is not None:
            start_label = max(start_label, pd.Timestamp(self.start, tz="UTC"))
        if self.end is not None:
            end_label = min(end_label, pd.Timestamp(self.end, tz="UTC"))

        if start_label > end_label:
            return pd.Series(False, index=index)

        start_naive = start_label.tz_localize(
            None) if start_label.tzinfo is not None else start_label
        end_naive = end_label.tz_localize(
            None) if end_label.tzinfo is not None else end_label
        normalized_naive = normalized.tz_localize(
            None) if normalized.tz is not None else normalized
        sessions = self.calendar.sessions_in_range(start_naive, end_naive)
        session_set = set(sessions)
        mask = normalized_naive.isin(session_set)
        return pd.Series(mask, index=index)

    def __iter__(self) -> Iterator[BarSlice]:
        if not self._frames:
            return iter(())
        # Merge all timestamps across symbols
        combined_index = sorted(
            {ts for frame in self._frames.values() for ts in frame.index})
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
