"""Reusable helpers for computing backtest performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import sqrt
from typing import Iterable, Sequence

import statistics

from ..backtest.engine import BacktestResult


@dataclass(slots=True)
class BacktestMetrics:
    """Container for standard performance metrics."""

    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    trade_count: float
    final_cash: float
    sharpe_ratio: float
    net_profit: float


def compute_backtest_metrics(
    result: BacktestResult,
    *,
    initial_cash: float,
    start: date,
    end: date,
) -> BacktestMetrics:
    """Compute the canonical set of metrics for a backtest run."""

    curve = result.equity_curve
    final_equity = curve[-1][1] if curve else initial_cash
    total_return = (final_equity / initial_cash) - 1.0 if initial_cash else 0.0
    duration_years = _years_between(start, end)
    if duration_years > 0 and initial_cash > 0 and final_equity > 0:
        cagr = (final_equity / initial_cash) ** (1 / duration_years) - 1
    else:
        cagr = 0.0
    max_drawdown = _max_drawdown(curve) if curve else 0.0
    trade_count = float(len(result.trades))
    cash = result.final_state.cash
    sharpe = _sharpe_ratio(curve)
    net_profit = final_equity - initial_cash
    return BacktestMetrics(
        final_equity=float(final_equity),
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=float(max_drawdown),
        trade_count=trade_count,
        final_cash=float(cash),
        sharpe_ratio=float(sharpe),
        net_profit=float(net_profit),
    )


def _years_between(start: date, end: date) -> float:
    delta = (datetime.combine(end, time.min) -
             datetime.combine(start, time.min)).days
    return max(delta / 365.25, 1e-6)


def _max_drawdown(curve: Sequence[tuple[datetime, float]]) -> float:
    drawdown = 0.0
    peak = float("-inf")
    for _, value in curve:
        v = float(value)
        peak = v if peak == float("-inf") else max(peak, v)
        if peak <= 0:
            continue
        drawdown = min(drawdown, (v / peak) - 1.0)
    return abs(drawdown)


def _sharpe_ratio(curve: Sequence[tuple[datetime, float]] | None) -> float:
    if not curve or len(curve) < 2:
        return 0.0
    returns: list[float] = []
    for idx in range(1, len(curve)):
        previous = float(curve[idx - 1][1])
        current = float(curve[idx][1])
        if previous <= 0:
            continue
        daily_return = (current / previous) - 1.0
        returns.append(daily_return)
    if not returns:
        return 0.0
    mean_return = statistics.fmean(returns)
    if len(returns) == 1:
        return 0.0
    std_dev = statistics.stdev(returns)
    if std_dev == 0:
        return 0.0
    return (mean_return / std_dev) * sqrt(252)


__all__ = [
    "BacktestMetrics",
    "compute_backtest_metrics",
]
