"""Strategy interface for the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from .data import BarSlice

if True:  # pragma: no cover - typing convenience
    from .portfolio import PortfolioState


@dataclass(slots=True)
class Order:
    """Simple market order representation."""

    symbol: str
    quantity: int  # positive = buy, negative = sell


@dataclass(slots=True)
class StrategyContext:
    """View of the portfolio exposed to strategies."""

    portfolio: "PortfolioState"


class Strategy(Protocol):
    """User-defined strategy must implement the bar callback."""

    def on_bar(self, context: StrategyContext, data: BarSlice) -> List[Order]:
        """Return a list of orders to execute for the current bar slice."""