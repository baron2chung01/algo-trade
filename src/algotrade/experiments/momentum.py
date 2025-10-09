"""Experiment runner for cross-sectional momentum strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

from ..backtest import BacktestConfig, BacktestEngine
from ..backtest.engine import BacktestResult
from ..backtest.fees import CommissionModel, SlippageModel
from ..data.stores.local import ParquetBarStore
from ..strategies import MomentumConfig, MomentumStrategy
from .metrics import BacktestMetrics, compute_backtest_metrics


@dataclass(slots=True)
class MomentumParameters:
    """Parameter set for :class:`MomentumStrategy`."""

    lookback_days: int = 126
    skip_days: int = 21
    rebalance_days: int = 21
    max_positions: int = 5
    lot_size: int = 1
    cash_reserve_pct: float = 0.05
    min_momentum: float = 0.0
    volatility_window: int = 20
    volatility_exponent: float = 1.0

    def label(self) -> str:
        return (
            "lookback={look}|skip={skip}|rebalance={reb}|maxpos={maxp}|"
            "reserve={reserve:.3f}|minmom={minmom:.3f}|volwin={volwin}|volexp={volexp:.2f}"
        ).format(
            look=self.lookback_days,
            skip=self.skip_days,
            reb=self.rebalance_days,
            maxp=self.max_positions,
            reserve=self.cash_reserve_pct,
            minmom=self.min_momentum,
            volwin=self.volatility_window,
            volexp=self.volatility_exponent,
        )

    def to_strategy_config(self, symbols: Sequence[str], initial_cash: float) -> MomentumConfig:
        return MomentumConfig(
            symbols=symbols,
            initial_cash=initial_cash,
            lookback_days=self.lookback_days,
            skip_days=self.skip_days,
            rebalance_days=self.rebalance_days,
            max_positions=self.max_positions,
            lot_size=self.lot_size,
            cash_reserve_pct=self.cash_reserve_pct,
            min_momentum=self.min_momentum,
            volatility_window=self.volatility_window,
            volatility_exponent=self.volatility_exponent,
        )


@dataclass(slots=True)
class MomentumExperimentConfig:
    store_path: Path
    universe: Sequence[str]
    initial_cash: float
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    bar_size: str = "1d"
    calendar_name: str = "XNYS"
    commission_model: CommissionModel | None = None
    slippage_model: SlippageModel | None = None

    def validate(self) -> None:
        if not self.universe:
            raise ValueError("universe must contain at least one symbol")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.training_window[0] >= self.training_window[1]:
            raise ValueError("training_window start must be before end")
        if self.paper_window[0] >= self.paper_window[1]:
            raise ValueError("paper_window start must be before end")

    def base_backtest_config(self, start: date, end: date) -> BacktestConfig:
        return BacktestConfig(
            symbols=self.universe,
            start=start,
            end=end,
            bar_size=self.bar_size,
            initial_cash=self.initial_cash,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            calendar_name=self.calendar_name,
        )


@dataclass(slots=True)
class MomentumExperimentOutcome:
    parameters: MomentumParameters
    training_metrics: BacktestMetrics
    paper_metrics: BacktestMetrics
    training_result: BacktestResult
    paper_result: BacktestResult

    def as_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        payload.update({f"train_{key}": value for key,
                       value in asdict(self.training_metrics).items()})
        payload.update({f"paper_{key}": value for key,
                       value in asdict(self.paper_metrics).items()})
        return payload


def run_momentum_experiment(
    config: MomentumExperimentConfig,
    parameters: MomentumParameters,
) -> MomentumExperimentOutcome:
    """Run momentum strategy over training and paper windows."""

    config.validate()
    store = ParquetBarStore(config.store_path)
    strategy_config = parameters.to_strategy_config(
        config.universe, config.initial_cash)

    # Training backtest
    training_bt = BacktestEngine(
        config=config.base_backtest_config(*config.training_window),
        store=store,
    )
    training_strategy = MomentumStrategy(strategy_config)
    training_result = training_bt.run(training_strategy)
    training_metrics = compute_backtest_metrics(
        training_result,
        initial_cash=config.initial_cash,
        start=config.training_window[0],
        end=config.training_window[1],
    )

    # Paper trading window (fresh strategy instance to avoid lookahead)
    paper_bt = BacktestEngine(
        config=config.base_backtest_config(*config.paper_window),
        store=store,
    )
    paper_strategy = MomentumStrategy(strategy_config)
    paper_result = paper_bt.run(paper_strategy)
    paper_metrics = compute_backtest_metrics(
        paper_result,
        initial_cash=config.initial_cash,
        start=config.paper_window[0],
        end=config.paper_window[1],
    )

    return MomentumExperimentOutcome(
        parameters=parameters,
        training_metrics=training_metrics,
        paper_metrics=paper_metrics,
        training_result=training_result,
        paper_result=paper_result,
    )


DEFAULT_MAX_OPTIMIZATION_EVALUATIONS = 80


@dataclass(slots=True)
class MomentumOptimizationSpec:
    """Collection of candidate values for momentum optimization."""

    lookback_days: Sequence[int]
    skip_days: Sequence[int]
    rebalance_days: Sequence[int]
    max_positions: Sequence[int]
    lot_size: Sequence[int]
    cash_reserve_pct: Sequence[float]
    min_momentum: Sequence[float]
    volatility_window: Sequence[int]
    volatility_exponent: Sequence[float]


@dataclass(slots=True)
class MomentumOptimizationEvaluation:
    parameters: MomentumParameters
    outcome: MomentumExperimentOutcome

    def score(self) -> Tuple[float, float, float]:
        metrics = self.outcome.paper_metrics
        return (
            float(metrics.sharpe_ratio),
            float(metrics.total_return),
            -float(metrics.max_drawdown),
        )


@dataclass(slots=True)
class MomentumOptimizationSummary:
    evaluations: List[MomentumOptimizationEvaluation]

    def best(self) -> MomentumOptimizationEvaluation:
        if not self.evaluations:
            raise ValueError("No momentum evaluations were produced.")
        return self.evaluations[0]


def _unique_int_values(values: Iterable[int]) -> List[int]:
    normalized: List[int] = []
    seen: set[int] = set()
    for value in values:
        candidate = int(value)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if not normalized:
        raise ValueError(
            "Momentum optimization integer range produced no values.")
    normalized.sort()
    return normalized


def _unique_float_values(values: Iterable[float]) -> List[float]:
    normalized: List[float] = []
    seen: set[float] = set()
    for raw in values:
        candidate = round(float(raw), 6)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if not normalized:
        raise ValueError(
            "Momentum optimization float range produced no values.")
    normalized.sort()
    return normalized


def generate_momentum_parameter_grid(
    spec: MomentumOptimizationSpec,
    *,
    max_evaluations: int = DEFAULT_MAX_OPTIMIZATION_EVALUATIONS,
) -> List[MomentumParameters]:
    """Generate a capped list of momentum parameter combinations."""

    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")

    lookback_options = _unique_int_values(spec.lookback_days)
    skip_options = _unique_int_values(spec.skip_days)
    rebalance_options = _unique_int_values(spec.rebalance_days)
    max_position_options = _unique_int_values(spec.max_positions)
    lot_size_options = _unique_int_values(spec.lot_size)
    cash_reserve_options = _unique_float_values(spec.cash_reserve_pct)
    min_momentum_options = _unique_float_values(spec.min_momentum)
    volatility_window_options = _unique_int_values(spec.volatility_window)
    volatility_exponent_options = _unique_float_values(
        spec.volatility_exponent)

    axis_lengths = (
        len(lookback_options)
        * len(skip_options)
        * len(rebalance_options)
        * len(max_position_options)
        * len(lot_size_options)
        * len(cash_reserve_options)
        * len(min_momentum_options)
        * len(volatility_window_options)
        * len(volatility_exponent_options)
    )
    if axis_lengths == 0:
        raise ValueError("Momentum optimization parameter grid is empty.")

    stride = max(1, axis_lengths // max_evaluations)
    parameters: List[MomentumParameters] = []

    for index, values in enumerate(
        product(
            lookback_options,
            skip_options,
            rebalance_options,
            max_position_options,
            lot_size_options,
            cash_reserve_options,
            min_momentum_options,
            volatility_window_options,
            volatility_exponent_options,
        )
    ):
        if index % stride != 0 and len(parameters) < max_evaluations:
            continue

        (
            lookback,
            skip,
            rebalance,
            max_positions,
            lot_size,
            cash_reserve,
            min_momentum,
            vol_window,
            vol_exponent,
        ) = values

        if skip >= lookback:
            continue

        params = MomentumParameters(
            lookback_days=lookback,
            skip_days=skip,
            rebalance_days=rebalance,
            max_positions=max_positions,
            lot_size=lot_size,
            cash_reserve_pct=max(0.0, min(float(cash_reserve), 0.95)),
            min_momentum=float(min_momentum),
            volatility_window=vol_window,
            volatility_exponent=float(vol_exponent),
        )
        parameters.append(params)

        if len(parameters) >= max_evaluations:
            break

    if not parameters:
        raise ValueError("No valid momentum parameter combinations produced.")

    return parameters


def optimize_momentum_parameters(
    store_path: Path,
    universe: Sequence[str],
    initial_cash: float,
    training_window: Tuple[date, date],
    paper_window: Tuple[date, date],
    parameters: Sequence[MomentumParameters],
    *,
    bar_size: str = "1d",
    calendar_name: str = "XNYS",
) -> MomentumOptimizationSummary:
    if not parameters:
        raise ValueError(
            "Momentum optimization requires at least one parameter set.")

    experiment_config = MomentumExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        training_window=training_window,
        paper_window=paper_window,
        bar_size=bar_size,
        calendar_name=calendar_name,
    )

    evaluations: List[MomentumOptimizationEvaluation] = []
    for params in parameters:
        outcome = run_momentum_experiment(experiment_config, params)
        evaluations.append(MomentumOptimizationEvaluation(
            parameters=params, outcome=outcome))

    evaluations.sort(key=lambda evaluation: evaluation.score(), reverse=True)
    return MomentumOptimizationSummary(evaluations=evaluations)


__all__ = [
    "MomentumParameters",
    "MomentumExperimentConfig",
    "MomentumExperimentOutcome",
    "MomentumOptimizationSpec",
    "MomentumOptimizationEvaluation",
    "MomentumOptimizationSummary",
    "generate_momentum_parameter_grid",
    "optimize_momentum_parameters",
    "run_momentum_experiment",
]
