"""Abstract base classes for market data providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from ..contracts import ContractSpec


@dataclass(slots=True)
class HistoricalDataRequest:
    """Parameters for requesting historical bar data."""

    contract: ContractSpec
    end: datetime
    duration: str
    bar_size: str
    what_to_show: str = "TRADES"
    use_rth: bool = True


class BaseDataProvider(Protocol):
    """Interface for historical data providers."""

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        """Return bars matching :class:`HistoricalDataRequest` in IBKR schema order."""

        raise NotImplementedError
