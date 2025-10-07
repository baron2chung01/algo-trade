"""Momentum breakout strategy implementations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Iterable, List, Sequence

from ..backtest.data import BarSlice
from ..backtest.portfolio import Position
from ..backtest.strategy import Order, StrategyContext


class BreakoutPattern(str, Enum):
    """Supported breakout pattern templates."""

    TWENTY_DAY_HIGH = "twenty_day_high"
    DONCHIAN_CHANNEL = "donchian_channel"
    FIFTY_TWO_WEEK_HIGH = "fifty_two_week_high"
    VOLUME_SPIKE_HIGH = "volume_spike_high"
    VOLATILITY_CONTRACTION = "volatility_contraction"


VOLATILITY_CONTRACTION_RATIO = 0.65
VOLATILITY_CONTRACTION_POSITION_RATIO = 0.75


@dataclass(slots=True)
class BreakoutConfig:
    """Configuration values for the breakout strategy."""

    symbols: Sequence[str]
    initial_cash: float
    pattern: BreakoutPattern = BreakoutPattern.TWENTY_DAY_HIGH
    lookback_days: int = 20
    breakout_buffer_pct: float = 0.002
    volume_ratio_threshold: float = 1.0
    volume_lookback_days: int = 20
    max_positions: int = 5
    max_hold_days: int = 20
    stop_loss_pct: float | None = 0.05
    trailing_stop_pct: float | None = 0.05
    profit_target_pct: float | None = None
    target_position_pct: float = 0.2
    lot_size: int = 1
    cash_reserve_pct: float = 0.05


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class BreakoutStrategy:
    """Pattern-based breakout strategy with risk controls."""

    def __init__(self, config: BreakoutConfig) -> None:
        if not config.symbols:
            raise ValueError("At least one symbol is required")
        _validate_positive("initial_cash", config.initial_cash)
        _validate_positive("lookback_days", config.lookback_days)
        _validate_positive("volume_lookback_days", config.volume_lookback_days)
        _validate_positive("target_position_pct", config.target_position_pct)
        _validate_positive("max_positions", config.max_positions)
        _validate_positive("lot_size", config.lot_size)
        if config.breakout_buffer_pct < 0:
            raise ValueError("breakout_buffer_pct cannot be negative")
        if config.volume_ratio_threshold < 0:
            raise ValueError("volume_ratio_threshold cannot be negative")
        if config.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if config.cash_reserve_pct < 0:
            raise ValueError("cash_reserve_pct cannot be negative")
        if config.stop_loss_pct is not None and config.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive when provided")
        if config.trailing_stop_pct is not None and config.trailing_stop_pct <= 0:
            raise ValueError(
                "trailing_stop_pct must be positive when provided")
        if config.profit_target_pct is not None and config.profit_target_pct <= 0:
            raise ValueError(
                "profit_target_pct must be positive when provided")

        self.config = config
        self.symbols = list(dict.fromkeys(symbol.upper()
                            for symbol in config.symbols))
        effective_lookback = self._effective_lookback()
        self.history_window = max(
            (effective_lookback * 2) + 2,
            config.volume_lookback_days + 2,
            10,
        )
        self.high_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.close_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.volume_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.entry_prices: Dict[str, float] = {}
        self.hold_counters: Dict[str, int] = {}
        self.peak_prices: Dict[str, float] = {}

    def on_bar(self, context: StrategyContext, data: BarSlice) -> List[Order]:
        self._update_history(data)
        positions = self._sync_positions(context, data)

        orders: List[Order] = []
        exit_orders, freed_cash = self._build_exit_orders(positions, data)
        orders.extend(exit_orders)

        active_after_exits = max(
            self._count_active_positions(
                positions.values()) - len(exit_orders),
            0,
        )
        available_cash = context.portfolio.cash + freed_cash
        reserve = self.config.initial_cash * self.config.cash_reserve_pct
        available_cash = max(available_cash - reserve, 0.0)

        entry_orders = self._build_entry_orders(
            data,
            positions,
            active_after_exits,
            available_cash,
            exit_orders,
        )
        orders.extend(entry_orders)
        return orders

    def _update_history(self, data: BarSlice) -> None:
        for symbol, bar in data.bars.items():
            key = symbol.upper()
            self.high_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.high))
            self.close_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.close))
            self.volume_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.volume))

    def _sync_positions(self, context: StrategyContext, data: BarSlice) -> Dict[str, Position]:
        positions = {
            symbol.upper(): pos
            for symbol, pos in context.portfolio.positions.items()
            if pos.quantity != 0
        }
        for symbol in list(self.hold_counters.keys()):
            if symbol not in positions:
                self.hold_counters.pop(symbol, None)
                self.entry_prices.pop(symbol, None)
                self.peak_prices.pop(symbol, None)
        for symbol, position in positions.items():
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1
            bar = data.bars.get(symbol)
            if bar is not None:
                current_price = float(bar.close)
                peak = self.peak_prices.get(symbol, current_price)
                self.peak_prices[symbol] = max(peak, current_price)
        return positions

    def _build_exit_orders(
        self,
        positions: Dict[str, Position],
        data: BarSlice,
    ) -> tuple[List[Order], float]:
        orders: List[Order] = []
        freed_cash = 0.0
        for symbol, position in positions.items():
            bar = data.bars.get(symbol)
            if bar is None:
                continue
            price = float(bar.close)
            should_exit = False

            entry_price = self.entry_prices.get(symbol, position.avg_price)
            if (
                self.config.stop_loss_pct is not None
                and entry_price > 0
                and price <= entry_price * (1 - self.config.stop_loss_pct)
            ):
                should_exit = True

            peak_price = self.peak_prices.get(symbol, price)
            if (
                not should_exit
                and self.config.trailing_stop_pct is not None
                and peak_price > 0
                and price <= peak_price * (1 - self.config.trailing_stop_pct)
            ):
                should_exit = True

            if (
                not should_exit
                and self.config.profit_target_pct is not None
                and entry_price > 0
                and price >= entry_price * (1 + self.config.profit_target_pct)
            ):
                should_exit = True

            if (
                not should_exit
                and self.config.max_hold_days > 0
                and self.hold_counters.get(symbol, 0) >= self.config.max_hold_days
            ):
                should_exit = True

            if should_exit:
                quantity = -position.quantity
                orders.append(Order(symbol=symbol, quantity=quantity))
                freed_cash += -quantity * price
                self.entry_prices.pop(symbol, None)
                self.hold_counters.pop(symbol, None)
                self.peak_prices.pop(symbol, None)
        return orders, freed_cash

    def _build_entry_orders(
        self,
        data: BarSlice,
        positions: Dict[str, Position],
        active_positions: int,
        available_cash: float,
        exit_orders: List[Order],
    ) -> List[Order]:
        orders: List[Order] = []
        pending_cash = 0.0
        pending_entries = 0
        exit_symbols = {order.symbol.upper() for order in exit_orders}

        for symbol in self.symbols:
            if active_positions + pending_entries >= self.config.max_positions:
                break
            if symbol in exit_symbols:
                continue
            position = positions.get(symbol)
            if position and position.quantity != 0:
                continue

            bar = data.bars.get(symbol)
            if bar is None:
                continue

            if not self._is_breakout(symbol, bar):
                continue

            target_value = self.config.initial_cash * self.config.target_position_pct
            budget = available_cash - pending_cash
            if budget <= 0:
                break
            allocation = min(target_value, budget)
            price = float(bar.close)
            lot = max(self.config.lot_size, 1)
            raw_quantity = allocation / price
            lots = int(raw_quantity // lot)
            if lots <= 0:
                continue
            quantity = lots * lot
            cost = quantity * price
            if cost <= 0 or cost + pending_cash > available_cash:
                continue

            orders.append(Order(symbol=symbol, quantity=quantity))
            self.entry_prices[symbol] = price
            self.hold_counters[symbol] = 0
            self.peak_prices[symbol] = price
            pending_cash += cost
            pending_entries += 1

        return orders

    def _is_breakout(self, symbol: str, bar) -> bool:
        highs = self.high_history.get(symbol, deque())
        closes = self.close_history.get(symbol, deque())
        volumes = self.volume_history.get(symbol, deque())
        if len(highs) <= 1 or len(closes) <= 1:
            return False

        lookback = self._effective_lookback()
        recent_highs = list(highs)[:-1][-lookback:]
        recent_closes = list(closes)[:-1][-lookback:]
        if len(recent_highs) < lookback:
            return False

        breakout_level = max(recent_highs)
        price = float(bar.close)
        if breakout_level <= 0:
            return False

        buffer_multiplier = 1 + self.config.breakout_buffer_pct
        if price < breakout_level * buffer_multiplier:
            return False

        if self.config.pattern == BreakoutPattern.VOLATILITY_CONTRACTION:
            if not self._passes_volatility_contraction(closes, price, lookback):
                return False

        if self.config.pattern == BreakoutPattern.DONCHIAN_CHANNEL:
            lower_band = min(recent_closes)
            if lower_band <= 0:
                return False
            if price / lower_band < 1 + (self.config.breakout_buffer_pct / 2):
                return False

        if self.config.pattern == BreakoutPattern.VOLUME_SPIKE_HIGH or self.config.volume_ratio_threshold > 1.0:
            recent_volumes = list(
                volumes)[:-1][-self.config.volume_lookback_days:]
            if len(recent_volumes) < min(self.config.volume_lookback_days, len(volumes) - 1):
                return False
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            if avg_volume <= 0:
                return False
            if float(bar.volume) / avg_volume < self.config.volume_ratio_threshold:
                return False

        return True

    def _effective_lookback(self) -> int:
        if self.config.pattern == BreakoutPattern.FIFTY_TWO_WEEK_HIGH:
            return max(self.config.lookback_days, 252)
        if self.config.pattern == BreakoutPattern.DONCHIAN_CHANNEL:
            return max(self.config.lookback_days, 55)
        if self.config.pattern == BreakoutPattern.VOLATILITY_CONTRACTION:
            return max(self.config.lookback_days, 40)
        return max(self.config.lookback_days, 2)

    def _passes_volatility_contraction(
        self,
        closes: Deque[float],
        price: float,
        lookback: int,
    ) -> bool:
        if lookback <= 1:
            return False
        history = list(closes)
        if len(history) < (lookback * 2) + 1:
            return False

        past_history = history[:-1]
        if len(past_history) < lookback * 2:
            return False

        recent_window = past_history[-lookback:]
        prior_window = past_history[-lookback * 2:-lookback]
        if len(recent_window) < lookback or len(prior_window) < lookback:
            return False

        recent_high = max(recent_window)
        recent_low = min(recent_window)
        prior_high = max(prior_window)
        prior_low = min(prior_window)
        recent_range = recent_high - recent_low
        prior_range = prior_high - prior_low
        if prior_range <= 0 or recent_range <= 0:
            return False

        if recent_range / prior_range > VOLATILITY_CONTRACTION_RATIO:
            return False

        threshold_price = recent_low + recent_range * \
            VOLATILITY_CONTRACTION_POSITION_RATIO
        if price < threshold_price:
            return False

        return True

    @staticmethod
    def _count_active_positions(positions: Iterable[Position]) -> int:
        return sum(1 for pos in positions if pos.quantity != 0)


__all__ = ["BreakoutPattern", "BreakoutConfig", "BreakoutStrategy"]
