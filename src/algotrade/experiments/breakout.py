"""Experiment runner and optimizer for breakout strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from ..backtest import BacktestConfig, BacktestEngine
from ..backtest.engine import BacktestResult
from ..backtest.fees import CommissionModel, SlippageModel
from ..data.stores.local import ParquetBarStore
from ..strategies import BreakoutConfig, BreakoutPattern, BreakoutStrategy
from .mean_reversion import ExperimentSplit, HoldRange, ParameterRange
from .metrics import compute_backtest_metrics

MAX_PARAMETER_COMBINATIONS = 10_000


@dataclass(slots=True)
class BreakoutParameters:
    """Parameter overrides for the breakout strategy."""

    pattern: BreakoutPattern
    lookback_days: int
    breakout_buffer_pct: float
    volume_ratio_threshold: float
    volume_lookback_days: int
    max_hold_days: int
    target_position_pct: float
    stop_loss_pct: float | None
    trailing_stop_pct: float | None
    profit_target_pct: float | None
    lot_size: int = 1

    def label(self) -> str:
        stop = "none" if self.stop_loss_pct is None else f"{self.stop_loss_pct:.3f}"
        trail = "none" if self.trailing_stop_pct is None else f"{self.trailing_stop_pct:.3f}"
        profit = "none" if self.profit_target_pct is None else f"{self.profit_target_pct:.3f}"
        hold = "inf" if self.max_hold_days == 0 else str(self.max_hold_days)
        return (
            f"pattern={self.pattern.value}|lookback={self.lookback_days}|"
            f"buffer={self.breakout_buffer_pct:.3f}|vol={self.volume_ratio_threshold:.2f}|"
            f"hold={hold}|target={self.target_position_pct:.3f}|stop={stop}|"
            f"trail={trail}|profit={profit}|lot={self.lot_size}"
        )

    def validate(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.breakout_buffer_pct < 0:
            raise ValueError("breakout_buffer_pct cannot be negative")
        if self.volume_ratio_threshold < 0:
            raise ValueError("volume_ratio_threshold cannot be negative")
        if self.volume_lookback_days <= 0:
            raise ValueError("volume_lookback_days must be positive")
        if self.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if self.target_position_pct <= 0 or self.target_position_pct > 1:
            raise ValueError("target_position_pct must be within (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive when provided")
        if self.trailing_stop_pct is not None and self.trailing_stop_pct <= 0:
            raise ValueError(
                "trailing_stop_pct must be positive when provided")
        if self.profit_target_pct is not None and self.profit_target_pct <= 0:
            raise ValueError(
                "profit_target_pct must be positive when provided")


@dataclass(slots=True)
class FloatRange:
    """Inclusive float range definition with step."""

    minimum: float
    maximum: float
    step: float

    def values(self) -> List[float]:
        if self.step <= 0:
            raise ValueError("Range step must be positive")
        if self.maximum < self.minimum:
            raise ValueError("Range maximum must be >= minimum")
        values: List[float] = []
        current = self.minimum
        epsilon = self.step / 10
        while current <= self.maximum + epsilon:
            values.append(round(current, 6))
            current += self.step
        return values


@dataclass(slots=True)
class BreakoutParameterSpec:
    patterns: Sequence[BreakoutPattern | str] = field(
        default_factory=lambda: [BreakoutPattern.TWENTY_DAY_HIGH]
    )
    lookback_days: ParameterRange = field(
        default_factory=lambda: ParameterRange(20, 20, 1))
    breakout_buffer_pct: FloatRange = field(
        default_factory=lambda: FloatRange(0.0, 0.0, 1.0))
    volume_ratio_threshold: FloatRange = field(
        default_factory=lambda: FloatRange(1.0, 1.0, 1.0))
    volume_lookback_days: ParameterRange = field(
        default_factory=lambda: ParameterRange(20, 20, 1))
    max_hold_days: HoldRange = field(
        default_factory=lambda: HoldRange(5, 20, 5))
    target_position_pct: ParameterRange = field(
        default_factory=lambda: ParameterRange(10, 10, 1))
    stop_loss_pct: FloatRange | None = None
    trailing_stop_pct: FloatRange | None = None
    profit_target_pct: FloatRange | None = None
    include_no_stop_loss: bool = True
    include_no_trailing_stop: bool = True
    include_no_profit_target: bool = True
    lot_size: int = 1


@dataclass(slots=True)
class BreakoutExperimentConfig:
    store_path: Path
    universe: Sequence[str]
    initial_cash: float
    splits: Sequence[ExperimentSplit]
    parameters: Sequence[BreakoutParameters]
    bar_size: str = "1d"
    calendar_name: str = "XNYS"
    commission_model: CommissionModel | None = None
    slippage_model: SlippageModel | None = None

    def base_backtest_config(self) -> BacktestConfig:
        return BacktestConfig(
            symbols=self.universe,
            start=None,
            end=None,
            bar_size=self.bar_size,
            initial_cash=self.initial_cash,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            calendar_name=self.calendar_name,
        )

    def strategy_config(self, parameters: BreakoutParameters) -> BreakoutConfig:
        return BreakoutConfig(
            symbols=self.universe,
            initial_cash=self.initial_cash,
            pattern=parameters.pattern,
            lookback_days=parameters.lookback_days,
            breakout_buffer_pct=parameters.breakout_buffer_pct,
            volume_ratio_threshold=parameters.volume_ratio_threshold,
            volume_lookback_days=parameters.volume_lookback_days,
            max_positions=max(len(self.universe), 5),
            max_hold_days=parameters.max_hold_days,
            stop_loss_pct=parameters.stop_loss_pct,
            trailing_stop_pct=parameters.trailing_stop_pct,
            profit_target_pct=parameters.profit_target_pct,
            target_position_pct=parameters.target_position_pct,
            lot_size=parameters.lot_size,
            cash_reserve_pct=0.0,
        )

    def validate(self) -> None:
        if not self.universe:
            raise ValueError("universe must contain at least one symbol")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not self.splits:
            raise ValueError("At least one experiment split is required")
        if not self.parameters:
            raise ValueError("At least one parameter set is required")
        seen_names: Set[str] = set()
        for split in self.splits:
            split.validate()
            key = split.name.lower()
            if key in seen_names:
                raise ValueError(
                    f"Duplicate split name detected: '{split.name}'")
            seen_names.add(key)
        for params in self.parameters:
            params.validate()


@dataclass(slots=True)
class BreakoutExperimentRow:
    split: str
    parameter_label: str
    metrics: Dict[str, float]


@dataclass(slots=True)
class BreakoutExperimentResult:
    rows: List[BreakoutExperimentRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        records = []
        for row in self.rows:
            payload = {"split": row.split, "params": row.parameter_label}
            payload.update(row.metrics)
            records.append(payload)
        if not records:
            return pd.DataFrame(columns=["split", "params"])
        return pd.DataFrame.from_records(records)


def run_breakout_experiment(config: BreakoutExperimentConfig) -> BreakoutExperimentResult:
    config.validate()
    store = ParquetBarStore(config.store_path)
    base_backtest = config.base_backtest_config()
    result = BreakoutExperimentResult()

    for parameters in config.parameters:
        label = parameters.label()
        for split in config.splits:
            split_config = split.as_backtest_config(base_backtest)
            strategy_cfg = config.strategy_config(parameters)
            strategy = BreakoutStrategy(strategy_cfg)
            engine = BacktestEngine(config=split_config, store=store)
            backtest_result = engine.run(strategy)
            metrics = asdict(
                compute_backtest_metrics(
                    backtest_result,
                    initial_cash=split_config.initial_cash,
                    start=split.start,
                    end=split.end,
                )
            )
            result.rows.append(
                BreakoutExperimentRow(
                    split=split.name,
                    parameter_label=label,
                    metrics=metrics,
                )
            )
    return result


@dataclass(slots=True)
class BreakoutOptimizationOutcome:
    best_parameters: BreakoutParameters
    training_metrics: Dict[str, float]
    paper_metrics: Dict[str, float]
    parameter_frame: pd.DataFrame
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    paper_result: BacktestResult


def default_breakout_spec() -> BreakoutParameterSpec:
    return BreakoutParameterSpec(
        patterns=[
            BreakoutPattern.TWENTY_DAY_HIGH,
            BreakoutPattern.DONCHIAN_CHANNEL,
            BreakoutPattern.FIFTY_TWO_WEEK_HIGH,
            BreakoutPattern.VOLATILITY_CONTRACTION,
        ],
        lookback_days=ParameterRange(20, 60, 20),
        breakout_buffer_pct=FloatRange(0.0, 0.010, 0.005),
        volume_ratio_threshold=FloatRange(1.0, 1.5, 0.5),
        volume_lookback_days=ParameterRange(20, 20, 1),
        max_hold_days=HoldRange(10, 20, 10, include_infinite=True),
        target_position_pct=ParameterRange(10, 20, 10),
        stop_loss_pct=FloatRange(0.05, 0.05, 0.01),
        trailing_stop_pct=FloatRange(0.08, 0.08, 0.01),
        profit_target_pct=FloatRange(0.15, 0.15, 0.01),
        include_no_stop_loss=True,
        include_no_trailing_stop=True,
        include_no_profit_target=True,
        lot_size=10,
    )


def generate_breakout_parameter_grid(spec: BreakoutParameterSpec) -> List[BreakoutParameters]:
    patterns = [BreakoutPattern(pattern) for pattern in spec.patterns]
    lookbacks = spec.lookback_days.values()
    buffers = spec.breakout_buffer_pct.values()
    volume_ratios = spec.volume_ratio_threshold.values()
    volume_lookbacks = spec.volume_lookback_days.values()
    hold_values = spec.max_hold_days.values()
    target_values = [
        max(value, 1) / 100.0 for value in spec.target_position_pct.values()]
    stop_values = _resolve_optional_float_range(
        spec.stop_loss_pct, spec.include_no_stop_loss)
    trailing_values = _resolve_optional_float_range(
        spec.trailing_stop_pct, spec.include_no_trailing_stop)
    profit_values = _resolve_optional_float_range(
        spec.profit_target_pct, spec.include_no_profit_target)

    combinations = (
        len(patterns)
        * len(lookbacks)
        * len(buffers)
        * len(volume_ratios)
        * len(volume_lookbacks)
        * len(hold_values)
        * len(target_values)
        * len(stop_values)
        * len(trailing_values)
        * len(profit_values)
    )
    if combinations > MAX_PARAMETER_COMBINATIONS:
        raise ValueError(
            f"Parameter grid too large ({combinations} combinations). Please narrow your ranges."
        )

    parameters: List[BreakoutParameters] = []
    for pattern in patterns:
        for lookback in lookbacks:
            for buffer in buffers:
                for volume_ratio in volume_ratios:
                    for volume_lookback in volume_lookbacks:
                        for hold in hold_values:
                            hold_value = max(hold, 0)
                            for target in target_values:
                                for stop in stop_values:
                                    for trailing in trailing_values:
                                        for profit in profit_values:
                                            parameters.append(
                                                BreakoutParameters(
                                                    pattern=pattern,
                                                    lookback_days=int(
                                                        lookback),
                                                    breakout_buffer_pct=float(
                                                        buffer),
                                                    volume_ratio_threshold=float(
                                                        volume_ratio),
                                                    volume_lookback_days=int(
                                                        volume_lookback),
                                                    max_hold_days=hold_value,
                                                    target_position_pct=float(
                                                        target),
                                                    stop_loss_pct=stop,
                                                    trailing_stop_pct=trailing,
                                                    profit_target_pct=profit,
                                                    lot_size=spec.lot_size,
                                                )
                                            )
    if not parameters:
        raise ValueError(
            "No parameter combinations generated; check specification ranges.")
    return parameters


def _resolve_optional_float_range(
    range_spec: FloatRange | None,
    include_none: bool,
) -> List[float | None]:
    if range_spec is None:
        return [None]
    base_values = range_spec.values()
    values: List[float | None] = [float(value) for value in base_values]
    if include_none:
        values.insert(0, None)
    return values


def optimize_breakout_parameters(
    store_path: Path,
    universe: Sequence[str],
    initial_cash: float,
    training_window: Tuple[date, date],
    paper_window: Tuple[date, date],
    *,
    bar_size: str = "1d",
    calendar_name: str = "XNYS",
    parameter_grid: Sequence[BreakoutParameters] | None = None,
    parameter_spec: BreakoutParameterSpec | None = None,
) -> BreakoutOptimizationOutcome:
    if parameter_grid is not None and parameter_spec is not None:
        raise ValueError(
            "Specify either parameter_grid or parameter_spec, not both.")

    if parameter_grid is not None:
        parameters = list(parameter_grid)
    elif parameter_spec is not None:
        parameters = generate_breakout_parameter_grid(parameter_spec)
    else:
        parameters = generate_breakout_parameter_grid(default_breakout_spec())

    if not parameters:
        raise ValueError(
            "Parameter grid must contain at least one configuration.")

    train_split = ExperimentSplit(
        name="train", start=training_window[0], end=training_window[1])
    paper_split = ExperimentSplit(
        name="paper", start=paper_window[0], end=paper_window[1])

    experiment_config = BreakoutExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        splits=[train_split],
        parameters=parameters,
        bar_size=bar_size,
        calendar_name=calendar_name,
    )
    experiment_result = run_breakout_experiment(experiment_config)
    frame = experiment_result.to_frame()
    if frame.empty:
        raise ValueError("Experiment frame is empty; no parameters evaluated.")
    best_row = frame.sort_values(
        ["cagr", "final_equity"], ascending=[False, False]).iloc[0]

    lookup = {params.label(): params for params in parameters}
    best_label = best_row["params"]
    if best_label not in lookup:
        raise ValueError(
            f"Unable to locate parameters for label '{best_label}'.")
    best_params = lookup[best_label]

    store = ParquetBarStore(store_path)
    strategy_config = experiment_config.strategy_config(best_params)
    strategy = BreakoutStrategy(strategy_config)

    paper_backtest_config = BacktestConfig(
        symbols=universe,
        start=paper_split.start,
        end=paper_split.end,
        bar_size=bar_size,
        initial_cash=initial_cash,
        calendar_name=calendar_name,
    )
    engine = BacktestEngine(config=paper_backtest_config, store=store)
    paper_result = engine.run(strategy)

    training_metrics = {
        key: float(best_row[key])
        for key in best_row.index
        if key not in {"split", "params"}
    }

    paper_metrics = asdict(
        compute_backtest_metrics(
            paper_result,
            initial_cash=initial_cash,
            start=paper_split.start,
            end=paper_split.end,
        )
    )

    return BreakoutOptimizationOutcome(
        best_parameters=best_params,
        training_metrics=training_metrics,
        paper_metrics=paper_metrics,
        parameter_frame=frame,
        training_window=(train_split.start, train_split.end),
        paper_window=(paper_split.start, paper_split.end),
        paper_result=paper_result,
    )


__all__ = [
    "BreakoutParameters",
    "FloatRange",
    "BreakoutParameterSpec",
    "BreakoutExperimentConfig",
    "BreakoutExperimentResult",
    "BreakoutOptimizationOutcome",
    "default_breakout_spec",
    "generate_breakout_parameter_grid",
    "optimize_breakout_parameters",
    "run_breakout_experiment",
]
