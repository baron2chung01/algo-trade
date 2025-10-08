"""Volatility Contraction Pattern (VCP) strategy implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, MutableMapping, Sequence

from ..backtest.data import BarSlice
from ..backtest.portfolio import Position
from ..backtest.strategy import Order, StrategyContext


@dataclass(slots=True)
class VCPConfig:
    """Configuration for the VCP strategy."""

    symbols: Sequence[str]
    initial_cash: float
    base_lookback_days: int = 60
    pivot_lookback_days: int = 5
    min_contractions: int = 3
    max_contraction_pct: float = 0.18
    contraction_decay: float = 0.75
    breakout_buffer_pct: float = 0.002
    volume_squeeze_ratio: float = 0.8
    breakout_volume_ratio: float = 1.8
    volume_lookback_days: int = 20
    trend_ma_period: int = 50
    stop_loss_r_multiple: float = 1.0
    profit_target_r_multiple: float = 2.5
    trailing_stop_r_multiple: float | None = 1.5
    max_positions: int = 5
    max_hold_days: int = 45
    target_position_pct: float = 0.15
    lot_size: int = 1
    cash_reserve_pct: float = 0.1


@dataclass(slots=True)
class VCPEntryState:
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_share: float
    breakout_timestamp: datetime
    resistance: float
    base_low: float


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class VCPStrategy:
    """Daily VCP breakout strategy with risk-based exits."""

    def __init__(self, config: VCPConfig) -> None:
        if not config.symbols:
            raise ValueError("At least one symbol is required")
        _validate_positive("initial_cash", config.initial_cash)
        _validate_positive("base_lookback_days", config.base_lookback_days)
        _validate_positive("pivot_lookback_days", config.pivot_lookback_days)
        _validate_positive("min_contractions", config.min_contractions)
        _validate_positive("max_contraction_pct", config.max_contraction_pct)
        _validate_positive("volume_lookback_days", config.volume_lookback_days)
        _validate_positive("trend_ma_period", config.trend_ma_period)
        _validate_positive("max_positions", config.max_positions)
        _validate_positive("target_position_pct", config.target_position_pct)
        _validate_positive("lot_size", config.lot_size)
        if config.contraction_decay <= 0 or config.contraction_decay > 1:
            raise ValueError("contraction_decay must be within (0, 1]")
        if config.stop_loss_r_multiple <= 0:
            raise ValueError("stop_loss_r_multiple must be positive")
        if config.profit_target_r_multiple <= 0:
            raise ValueError("profit_target_r_multiple must be positive")
        if config.trailing_stop_r_multiple is not None and config.trailing_stop_r_multiple <= 0:
            raise ValueError(
                "trailing_stop_r_multiple must be positive when provided")
        if config.cash_reserve_pct < 0 or config.cash_reserve_pct >= 1:
            raise ValueError("cash_reserve_pct must be in [0, 1)")
        if config.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if config.volume_squeeze_ratio <= 0 or config.volume_squeeze_ratio > 1:
            raise ValueError("volume_squeeze_ratio must be within (0, 1]")
        if config.breakout_volume_ratio <= 0:
            raise ValueError("breakout_volume_ratio must be positive")

        self.config = config
        self.symbols = list(dict.fromkeys(symbol.upper()
                            for symbol in config.symbols))
        self.history_window = max(
            config.base_lookback_days * 2,
            config.trend_ma_period + 2,
            config.volume_lookback_days * 2,
        )
        self.close_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.high_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.low_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.volume_history: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.history_window) for symbol in self.symbols
        }
        self.entry_states: Dict[str, VCPEntryState] = {}
        self.hold_counters: Dict[str, int] = {}
        self.annotations: Dict[str,
                               List[MutableMapping[str, float | str]]] = {}

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

    # ------------------------------------------------------------------
    # History & bookkeeping helpers
    # ------------------------------------------------------------------

    def _update_history(self, data: BarSlice) -> None:
        for symbol, bar in data.bars.items():
            key = symbol.upper()
            self.close_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.close))
            self.high_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.high))
            self.low_history.setdefault(key, deque(
                maxlen=self.history_window)).append(float(bar.low))
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
                self.entry_states.pop(symbol, None)
        for symbol, position in positions.items():
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1
        return positions

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

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
            state = self.entry_states.get(symbol)
            price = float(bar.close)
            should_exit = False

            if state:
                if price <= state.stop_price:
                    should_exit = True
                elif price >= state.target_price:
                    should_exit = True
                elif (
                    self.config.max_hold_days > 0
                    and self.hold_counters.get(symbol, 0) >= self.config.max_hold_days
                ):
                    should_exit = True
                elif self.config.trailing_stop_r_multiple is not None:
                    trail_stop = price - \
                        (state.risk_per_share * self.config.trailing_stop_r_multiple)
                    state.stop_price = max(state.stop_price, trail_stop)
            else:
                if self.config.max_hold_days > 0 and self.hold_counters.get(symbol, 0) >= self.config.max_hold_days:
                    should_exit = True

            if should_exit:
                quantity = -position.quantity
                orders.append(Order(symbol=symbol, quantity=quantity))
                freed_cash += -quantity * price
                self.entry_states.pop(symbol, None)
                self.hold_counters.pop(symbol, None)
        return orders, freed_cash

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

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

            detection = self._detect_breakout(
                symbol, float(bar.close), float(bar.volume))
            if detection is None:
                continue

            entry_price, stop_price, target_price, resistance, base_low, risk_per_share = detection
            target_value = self.config.initial_cash * self.config.target_position_pct
            budget = available_cash - pending_cash
            if budget <= 0:
                break
            allocation = min(target_value, budget)
            lot = max(self.config.lot_size, 1)
            quantity = int((allocation / entry_price) // lot) * lot
            if quantity <= 0:
                continue
            cost = quantity * entry_price
            if cost <= 0 or cost + pending_cash > available_cash:
                continue

            orders.append(Order(symbol=symbol, quantity=quantity))
            pending_cash += cost
            pending_entries += 1

            self.entry_states[symbol] = VCPEntryState(
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                risk_per_share=risk_per_share,
                breakout_timestamp=data.timestamp,
                resistance=resistance,
                base_low=base_low,
            )
            self.hold_counters[symbol] = 0
            self._record_annotation(
                symbol,
                data.timestamp,
                entry_price,
                stop_price,
                target_price,
                resistance,
                base_low,
                risk_per_share,
            )

        return orders

    # ------------------------------------------------------------------
    # Pattern detection helpers
    # ------------------------------------------------------------------

    def _detect_breakout(
        self,
        symbol: str,
        price: float,
        volume: float,
    ) -> tuple[float, float, float, float, float, float] | None:
        closes = list(self.close_history.get(symbol, []))
        highs = list(self.high_history.get(symbol, []))
        lows = list(self.low_history.get(symbol, []))
        volumes = list(self.volume_history.get(symbol, []))
        base_window = self.config.base_lookback_days
        if len(closes) < max(base_window, self.config.trend_ma_period) + 5:
            return None
        base_closes = closes[-base_window:]
        base_highs = highs[-base_window:]
        base_lows = lows[-base_window:]
        if len(base_closes) < base_window:
            return None

        if not self._trend_ok(closes, price):
            return None

        segments = self._split_segments(
            base_closes, self.config.min_contractions)
        if segments is None:
            return None
        contraction_sizes: List[float] = []
        segment_highs: List[float] = []
        segment_lows: List[float] = []
        for segment in segments:
            high = max(segment)
            low = min(segment)
            if high <= 0:
                return None
            drop = (high - low) / high
            contraction_sizes.append(drop)
            segment_highs.append(high)
            segment_lows.append(low)
            if drop > self.config.max_contraction_pct:
                return None

        for idx in range(1, len(contraction_sizes)):
            prev = contraction_sizes[idx - 1]
            current = contraction_sizes[idx]
            if prev <= 0:
                return None
            if current > prev * self.config.contraction_decay:
                return None

        resistance = segment_highs[-1]
        base_low = segment_lows[-1]
        overall_high = max(base_highs)
        if resistance < overall_high * 0.97:
            return None
        buffer_multiplier = 1 + self.config.breakout_buffer_pct
        if price < resistance * buffer_multiplier:
            return None

        if base_low <= 0 or price <= base_low:
            return None
        risk_per_share = price - base_low
        if risk_per_share <= 0:
            return None

        if not self._volume_ok(volumes, volume):
            return None

        target_price = price + \
            (risk_per_share * self.config.profit_target_r_multiple)
        stop_price = price - \
            (risk_per_share * self.config.stop_loss_r_multiple)
        stop_price = max(stop_price, base_low)
        return price, stop_price, target_price, resistance, base_low, risk_per_share

    def _trend_ok(self, closes: Sequence[float], price: float) -> bool:
        period = self.config.trend_ma_period
        if len(closes) < period:
            return False
        window = closes[-period:]
        ma = sum(window) / period
        return price >= ma

    def _volume_ok(self, volumes: Sequence[float], breakout_volume: float) -> bool:
        lookback = self.config.volume_lookback_days
        if len(volumes) < lookback * 2:
            return False
        recent = volumes[-lookback:]
        prior = volumes[-lookback * 2:-lookback]
        if not recent or not prior:
            return False
        recent_avg = sum(recent) / len(recent)
        prior_avg = sum(prior) / len(prior)
        if prior_avg <= 0 or recent_avg <= 0:
            return False
        if recent_avg > prior_avg * self.config.volume_squeeze_ratio:
            return False
        if breakout_volume < recent_avg * self.config.breakout_volume_ratio:
            return False
        return True

    def _split_segments(self, series: Sequence[float], parts: int) -> List[List[float]] | None:
        if parts <= 0 or len(series) < parts * 3:
            return None
        length = len(series)
        base_size = length // parts
        remainder = length % parts
        segments: List[List[float]] = []
        start = 0
        for idx in range(parts):
            end = start + base_size + (1 if idx < remainder else 0)
            segment = list(series[start:end])
            if not segment:
                return None
            segments.append(segment)
            start = end
        return segments

    def _record_annotation(
        self,
        symbol: str,
        timestamp: datetime,
        entry: float,
        stop: float,
        target: float,
        resistance: float,
        base_low: float,
        risk: float,
    ) -> None:
        payload: MutableMapping[str, float | str] = {
            "timestamp": timestamp.isoformat(),
            "entry": float(entry),
            "stop": float(stop),
            "target": float(target),
            "resistance": float(resistance),
            "base_low": float(base_low),
            "risk_per_share": float(risk),
        }
        self.annotations.setdefault(symbol, []).append(payload)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_active_positions(positions: Iterable[Position]) -> int:
        return sum(1 for pos in positions if pos.quantity != 0)

    def export_annotations(self) -> Dict[str, List[MutableMapping[str, float | str]]]:
        return {symbol: list(entries) for symbol, entries in self.annotations.items()}


__all__ = ["VCPConfig", "VCPStrategy"]
