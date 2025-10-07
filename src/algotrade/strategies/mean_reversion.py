"""Mean reversion strategy implementation using RSI(2) signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence

from ..backtest.data import BarSlice
from ..backtest.portfolio import Position
from ..backtest.strategy import Order, StrategyContext


@dataclass(slots=True)
class MeanReversionConfig:
    """Configuration parameters for :class:`MeanReversionStrategy`."""

    symbols: Sequence[str]
    initial_cash: float
    rsi_period: int = 2
    entry_threshold: float = 5.0
    exit_threshold: float = 70.0
    target_position_pct: float = 0.2
    max_positions: int = 5
    max_hold_days: int = 5
    stop_loss_pct: float | None = None
    lot_size: int = 1
    cash_reserve_pct: float = 0.05


def _calculate_rsi(closes: Sequence[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("RSI period must be positive")
    if len(closes) <= period:
        return None

    gains = 0.0
    losses = 0.0
    start_idx = len(closes) - period
    for idx in range(start_idx, len(closes)):
        change = closes[idx] - closes[idx - 1]
        if change > 0:
            gains += change
        else:
            losses -= change

    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 0.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionStrategy:
    """RSI-based mean reversion strategy following the project blueprint."""

    def __init__(self, config: MeanReversionConfig) -> None:
        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not config.symbols:
            raise ValueError("At least one symbol is required")
        if config.target_position_pct <= 0:
            raise ValueError("target_position_pct must be positive")
        if config.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if config.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if config.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if config.entry_threshold < 0 or config.exit_threshold < 0:
            raise ValueError("RSI thresholds must be non-negative")
        if config.cash_reserve_pct < 0:
            raise ValueError("cash_reserve_pct cannot be negative")

        self.config = config
        self.symbols = list(dict.fromkeys(symbol.upper()
                            for symbol in config.symbols))
        self.history_window = max(
            config.rsi_period + 1, config.max_hold_days + 1, 5)
        self.price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.entry_prices: Dict[str, float] = {}
        self.hold_counters: Dict[str, int] = {}

    def on_bar(self, context: StrategyContext, data: BarSlice) -> List[Order]:
        orders: List[Order] = []
        self._update_price_history(data)
        active_positions = self._sync_hold_counters(context)

        exit_orders, freed_cash = self._build_exit_orders(
            active_positions, data)
        orders.extend(exit_orders)

        projected_open = max(self._count_active_positions(
            active_positions.values()) - len(exit_orders), 0)
        available_cash = context.portfolio.cash + freed_cash
        reserve = self.config.initial_cash * self.config.cash_reserve_pct
        available_cash = max(available_cash - reserve, 0.0)

        entry_orders = self._build_entry_orders(
            data,
            active_positions,
            projected_open,
            available_cash,
            exit_orders,
        )
        orders.extend(entry_orders)
        return orders

    def _update_price_history(self, data: BarSlice) -> None:
        for symbol, bar in data.bars.items():
            key = symbol.upper()
            history = self.price_history.setdefault(
                key,
                deque(maxlen=self.history_window),
            )
            history.append(float(bar.close))

    def _sync_hold_counters(self, context: StrategyContext) -> Dict[str, Position]:
        positions = {
            symbol.upper(): pos
            for symbol, pos in context.portfolio.positions.items()
            if pos.quantity != 0
        }
        # Remove counters for closed symbols
        for symbol in list(self.hold_counters.keys()):
            if symbol not in positions:
                self.hold_counters.pop(symbol, None)
                self.entry_prices.pop(symbol, None)
        # Increment counters for active holdings
        for symbol in positions:
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1
        return positions

    def _build_exit_orders(
        self, positions: Dict[str, Position], data: BarSlice
    ) -> tuple[List[Order], float]:
        orders: List[Order] = []
        freed_cash = 0.0
        for symbol, position in positions.items():
            bar = data.bars.get(symbol)
            if bar is None:
                continue
            history = self.price_history.get(symbol, deque())
            rsi = _calculate_rsi(
                list(history), self.config.rsi_period) if history else None

            should_exit = False
            if rsi is not None and rsi >= self.config.exit_threshold:
                should_exit = True
            if (
                self.config.max_hold_days > 0
                and self.hold_counters.get(symbol, 0) >= self.config.max_hold_days
            ):
                should_exit = True
            entry_price = self.entry_prices.get(symbol, position.avg_price)
            if (
                self.config.stop_loss_pct is not None
                and entry_price > 0
                and bar.close <= entry_price * (1 - self.config.stop_loss_pct)
            ):
                should_exit = True

            if should_exit and position.quantity != 0:
                quantity = -position.quantity
                orders.append(Order(symbol=symbol, quantity=quantity))
                freed_cash += -quantity * float(bar.close)
                self.entry_prices.pop(symbol, None)
                self.hold_counters.pop(symbol, None)
        return orders, freed_cash

    def _build_entry_orders(
        self,
        data: BarSlice,
        positions: Dict[str, Position],
        open_positions: int,
        available_cash: float,
        exit_orders: List[Order],
    ) -> List[Order]:
        orders: List[Order] = []
        pending_cash = 0.0
        pending_entries = 0
        exit_symbols = {order.symbol.upper() for order in exit_orders}

        for symbol in self.symbols:
            bar = data.bars.get(symbol)
            if bar is None:
                continue
            if symbol in exit_symbols:
                continue
            position = positions.get(symbol)
            if position and position.quantity != 0:
                continue
            history = self.price_history.get(symbol, deque())
            rsi = _calculate_rsi(
                list(history), self.config.rsi_period) if history else None
            if rsi is None or rsi > self.config.entry_threshold:
                continue
            if open_positions + pending_entries >= self.config.max_positions:
                break

            target_value = self.config.initial_cash * self.config.target_position_pct
            budget = available_cash - pending_cash
            if budget <= 0:
                break
            allocation = min(target_value, budget)
            lot = max(self.config.lot_size, 1)
            raw_quantity = allocation / float(bar.close)
            lots = int(raw_quantity // lot)
            if lots == 0:
                continue
            quantity = lots * lot
            cost = quantity * float(bar.close)
            if cost <= 0 or cost + pending_cash > available_cash:
                continue

            orders.append(Order(symbol=symbol, quantity=quantity))
            self.entry_prices[symbol] = float(bar.close)
            self.hold_counters[symbol] = 0
            pending_cash += cost
            pending_entries += 1

        return orders

    @staticmethod
    def _count_active_positions(positions: Iterable[Position]) -> int:
        return sum(1 for pos in positions if pos.quantity != 0)
