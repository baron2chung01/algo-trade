"""Portfolio accounting primitives for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Mapping

from ..data.schemas import IBKRBar


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0


@dataclass(slots=True)
class Trade:
    timestamp: datetime
    symbol: str
    quantity: int
    price: float
    cash_after: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass(slots=True)
class PortfolioState:
    cash: float
    positions: Dict[str, Position]

    def position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def equity(self, prices: Mapping[str, IBKRBar]) -> float:
        equity = self.cash
        for pos in self.positions.values():
            bar = prices.get(pos.symbol)
            reference_price = bar.close if bar else pos.avg_price
            equity += pos.quantity * reference_price
        return equity


class Portfolio:
    """Track positions and cash with instant fills at provided prices."""

    def __init__(self, initial_cash: float) -> None:
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        self.trades: list[Trade] = []

    def snapshot(self) -> PortfolioState:
        positions = {
            symbol: Position(symbol=pos.symbol,
                             quantity=pos.quantity, avg_price=pos.avg_price)
            for symbol, pos in self.positions.items()
        }
        return PortfolioState(cash=self.cash, positions=positions)

    def _get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def execute(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        *,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> Trade:
        if quantity == 0:
            raise ValueError("Order quantity must be non-zero")

        position = self._get_position(symbol)
        cost = price * quantity
        self.cash -= cost
        self.cash -= commission

        prev_qty = position.quantity
        new_qty = prev_qty + quantity

        if prev_qty == 0 or (prev_qty > 0 and new_qty > 0 and quantity > 0) or (prev_qty < 0 and new_qty < 0 and quantity < 0):
            # Same direction accumulation
            position.avg_price = (
                (position.avg_price * prev_qty) + cost
            ) / new_qty
        elif prev_qty > 0 and new_qty >= 0 and quantity < 0:
            # Selling long position
            if new_qty == 0:
                position.avg_price = 0.0
        elif prev_qty < 0 and new_qty <= 0 and quantity > 0:
            # Covering short position
            if new_qty == 0:
                position.avg_price = 0.0
        else:
            # Crossing through zero into new direction
            position.avg_price = price

        position.quantity = new_qty
        if position.quantity == 0:
            position.avg_price = 0.0

        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            cash_after=self.cash,
        )
        self.trades.append(trade)
        return trade

    def mark_to_market(self, prices: Mapping[str, IBKRBar]) -> float:
        return self.snapshot().equity(prices)
