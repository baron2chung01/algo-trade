"""Cross-sectional momentum rotation strategy implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence

import statistics

from ..backtest.data import BarSlice
from ..backtest.portfolio import Position
from ..backtest.strategy import Order, StrategyContext


@dataclass(slots=True)
class MomentumConfig:
    """Configuration for :class:`MomentumStrategy`."""

    symbols: Sequence[str]
    initial_cash: float
    lookback_days: int = 126
    skip_days: int = 21
    rebalance_days: int = 21
    max_positions: int = 5
    lot_size: int = 1
    cash_reserve_pct: float = 0.05
    min_momentum: float = 0.0
    volatility_window: int = 20
    volatility_exponent: float = 1.0


class MomentumStrategy:
    """Cross-sectional momentum rotation with optional volatility scaling."""

    def __init__(self, config: MomentumConfig) -> None:
        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not config.symbols:
            raise ValueError("symbols must contain at least one ticker")
        if config.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if config.skip_days < 0:
            raise ValueError("skip_days cannot be negative")
        if config.rebalance_days <= 0:
            raise ValueError("rebalance_days must be positive")
        if config.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if config.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if config.cash_reserve_pct < 0 or config.cash_reserve_pct >= 1:
            raise ValueError("cash_reserve_pct must be within [0, 1)")
        if config.volatility_window < 0:
            raise ValueError("volatility_window cannot be negative")
        if config.volatility_exponent < 0:
            raise ValueError("volatility_exponent cannot be negative")

        self.config = config
        self.symbols = [symbol.upper() for symbol in config.symbols]
        min_history = config.lookback_days + config.skip_days + 1
        if config.volatility_window:
            min_history = max(min_history, config.volatility_window + 1)
        self.min_history = min_history
        self.history_window = max(min_history, config.volatility_window + 1)
        self.price_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.days_since_rebalance = max(config.rebalance_days - 1, 0)

    def on_bar(self, context: StrategyContext, data: BarSlice) -> List[Order]:
        self._update_price_history(data)
        if not self._has_sufficient_history():
            return []

        self.days_since_rebalance += 1
        if self.days_since_rebalance < self.config.rebalance_days:
            return []

        orders = self._build_rebalance_orders(context, data)
        self.days_since_rebalance = 0
        return orders

    def _update_price_history(self, data: BarSlice) -> None:
        for symbol, bar in data.bars.items():
            history = self.price_history.setdefault(
                symbol.upper(), deque(maxlen=self.history_window)
            )
            history.append(float(bar.close))

    def _has_sufficient_history(self) -> bool:
        for history in self.price_history.values():
            if len(history) >= self.min_history:
                return True
        return False

    def _build_rebalance_orders(
        self, context: StrategyContext, data: BarSlice
    ) -> List[Order]:
        scores = self._rank_momentum()
        selected = [symbol for symbol,
                    _ in scores[: self.config.max_positions]]
        selected_set = set(selected)

        orders: List[Order] = []
        positions = context.portfolio.positions

        # Close positions no longer in the selection set
        for symbol, position in positions.items():
            if position.quantity == 0:
                continue
            if symbol not in selected_set:
                orders.append(
                    Order(symbol=symbol, quantity=-position.quantity))

        if not selected:
            return orders

        equity = context.portfolio.equity(data.bars)
        investable_equity = equity * (1.0 - self.config.cash_reserve_pct)
        if investable_equity <= 0:
            return orders

        per_position_value = investable_equity / len(selected)
        lot_size = max(self.config.lot_size, 1)

        for symbol in selected:
            bar = data.bars.get(symbol)
            if bar is None or bar.close <= 0:
                continue
            price = float(bar.close)
            target_quantity = int(per_position_value / price)
            if target_quantity <= 0:
                continue
            target_quantity = (target_quantity // lot_size) * lot_size
            if target_quantity <= 0:
                continue
            current_position = positions.get(symbol, Position(symbol=symbol))
            delta = target_quantity - current_position.quantity
            if delta != 0:
                orders.append(Order(symbol=symbol, quantity=delta))

        return orders

    def _rank_momentum(self) -> List[tuple[str, float]]:
        scored: List[tuple[str, float]] = []
        for symbol, history in self.price_history.items():
            if len(history) < self.min_history:
                continue
            closes = list(history)
            skip_offset = self.config.skip_days + 1
            lookback_offset = self.config.lookback_days + self.config.skip_days + 1
            recent_idx = -skip_offset
            base_idx = -lookback_offset
            if abs(base_idx) > len(closes):
                continue
            recent_price = closes[recent_idx]
            base_price = closes[base_idx]
            if base_price <= 0:
                continue
            momentum = (recent_price / base_price) - 1.0
            if momentum < self.config.min_momentum:
                continue
            score = momentum
            if self.config.volatility_window > 1:
                window = closes[-(self.config.volatility_window + 1):]
                returns = [
                    (window[idx] / window[idx - 1]) - 1.0
                    for idx in range(1, len(window))
                    if window[idx - 1] > 0
                ]
                if returns:
                    vol = statistics.pstdev(returns)
                    if vol > 0 and self.config.volatility_exponent > 0:
                        score = momentum / \
                            (vol ** self.config.volatility_exponent)
            scored.append((symbol, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def reset(self) -> None:
        """Reset the strategy state (handy for repeated backtests)."""

        for history in self.price_history.values():
            history.clear()
        self.days_since_rebalance = max(self.config.rebalance_days - 1, 0)

    @staticmethod
    def _count_active_positions(positions: Iterable[Position]) -> int:
        return sum(1 for pos in positions if pos.quantity != 0)
