from collections import deque
from types import SimpleNamespace

import pytest

from algotrade.strategies.breakout import (
    BreakoutConfig,
    BreakoutPattern,
    BreakoutStrategy,
)
from algotrade.experiments.breakout import default_breakout_spec


@pytest.fixture
def volatility_strategy() -> BreakoutStrategy:
    config = BreakoutConfig(
        symbols=["AAPL"],
        initial_cash=100_000.0,
        pattern=BreakoutPattern.VOLATILITY_CONTRACTION,
        lookback_days=40,
        breakout_buffer_pct=0.0,
        volume_ratio_threshold=1.0,
        volume_lookback_days=40,
        max_positions=3,
        target_position_pct=0.2,
        lot_size=10,
    )
    return BreakoutStrategy(config)


def _build_history(values, history_window):
    return deque(values, maxlen=history_window)


def test_volatility_contraction_passes_when_range_tightens(volatility_strategy):
    lookback = volatility_strategy._effective_lookback()
    prior_window = [100 + 0.8 * idx for idx in range(lookback)]
    recent_window = [120 + 0.05 * idx for idx in range(lookback)]
    latest_price = recent_window[-1] + 0.5
    history = prior_window + recent_window + [latest_price]
    closes = _build_history(history, volatility_strategy.history_window)

    assert volatility_strategy._passes_volatility_contraction(
        closes,
        latest_price,
        lookback,
    )


def test_volatility_contraction_rejects_without_contraction(volatility_strategy):
    lookback = volatility_strategy._effective_lookback()
    prior_window = [90 + 0.5 * idx for idx in range(lookback)]
    recent_window = [110 + 1.2 * idx for idx in range(lookback)]
    latest_price = recent_window[-1] + 0.5
    history = prior_window + recent_window + [latest_price]
    closes = _build_history(history, volatility_strategy.history_window)

    assert not volatility_strategy._passes_volatility_contraction(
        closes,
        latest_price,
        lookback,
    )


def test_volatility_contraction_triggers_breakout_signal(volatility_strategy):
    symbol = "AAPL"
    lookback = volatility_strategy._effective_lookback()

    prior_highs = [100 + 0.8 * idx for idx in range(lookback)]
    recent_highs = [120 + 0.05 * idx for idx in range(lookback)]
    latest_high = recent_highs[-1] + 0.6
    high_history = _build_history(
        prior_highs + recent_highs + [latest_high], volatility_strategy.history_window)

    prior_closes = [value - 0.2 for value in prior_highs]
    recent_closes = [value - 0.05 for value in recent_highs]
    latest_close = recent_closes[-1] + 0.55
    close_history = _build_history(
        prior_closes + recent_closes + [latest_close], volatility_strategy.history_window)

    volume_history = _build_history(
        [900.0] * (lookback * 2) + [1_500.0], volatility_strategy.history_window)

    volatility_strategy.high_history[symbol] = high_history
    volatility_strategy.close_history[symbol] = close_history
    volatility_strategy.volume_history[symbol] = volume_history

    bar = SimpleNamespace(close=latest_close, high=latest_high,
                          low=latest_close * 0.98, volume=1_500.0)

    assert volatility_strategy._is_breakout(symbol, bar)


def test_default_spec_includes_volatility_contraction():
    spec = default_breakout_spec()
    patterns = {BreakoutPattern(pattern) for pattern in spec.patterns}
    assert BreakoutPattern.VOLATILITY_CONTRACTION in patterns
