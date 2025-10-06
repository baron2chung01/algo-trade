"""Data schemas aligned with IBKR TWS historical bar payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Iterable

import pandas as pd
from pydantic import BaseModel, Field

IBKR_BAR_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "average",
    "bar_count",
)


class IBKRBar(BaseModel):
    """Canonical representation of an IBKR historical bar."""

    timestamp: datetime = Field(...,
                                description="Bar end timestamp in exchange timezone.")
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)
    average: float | None = Field(
        None, ge=0, description="IBKR average price for the bar.")
    bar_count: int = Field(..., ge=0,
                           description="Number of trades composing the bar.")

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    @classmethod
    def from_ibkr_bar(cls, bar: object) -> "IBKRBar":
        """Create an :class:`IBKRBar` from an ``ib_insync.objects.BarData`` instance."""

        return cls(
            timestamp=getattr(bar, "date"),
            open=float(getattr(bar, "open")),
            high=float(getattr(bar, "high")),
            low=float(getattr(bar, "low")),
            close=float(getattr(bar, "close")),
            volume=float(getattr(bar, "volume", 0.0)),
            average=float(getattr(bar, "average", 0.0)) if getattr(
                bar, "average", None) is not None else None,
            bar_count=int(getattr(bar, "barCount", 0)),
        )

    def to_row(self) -> dict[str, float | int | datetime | None]:
        """Return the bar as a dictionary matching :data:`IBKR_BAR_COLUMNS`."""

        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "average": self.average,
            "bar_count": self.bar_count,
        }


@dataclass(slots=True)
class IBKRBarDataFrame:
    """Helper to construct validated :class:`pandas.DataFrame` objects for bar data."""

    columns: ClassVar[tuple[str, ...]] = IBKR_BAR_COLUMNS

    @classmethod
    def from_bars(cls, bars: Iterable[IBKRBar]) -> pd.DataFrame:
        """Convert an iterable of bars to a validated DataFrame."""

        df = pd.DataFrame([bar.to_row() for bar in bars], columns=cls.columns)
        if df.empty:
            return df
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    def ensure_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame contains the expected columns in correct order."""

        missing = set(cls.columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {sorted(missing)}")
        return df.loc[:, cls.columns].copy()
