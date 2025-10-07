"""Baseline experiment runner for the RSI(2) mean reversion strategy."""

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
from ..strategies import MeanReversionConfig, MeanReversionStrategy
from .metrics import compute_backtest_metrics

MAX_PARAMETER_COMBINATIONS = 10_000


@dataclass(slots=True)
class ExperimentSplit:
    """Date range for a single experiment segment."""

    name: str
    start: date
    end: date

    def as_backtest_config(self, base: BacktestConfig) -> BacktestConfig:
        return BacktestConfig(
            symbols=base.symbols,
            start=self.start,
            end=self.end,
            bar_size=base.bar_size,
            initial_cash=base.initial_cash,
            commission_model=base.commission_model,
            slippage_model=base.slippage_model,
            calendar_name=base.calendar_name,
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Split name must be provided")
        if self.start > self.end:
            raise ValueError(
                f"Split '{self.name}' start date must be on or before end date")


@dataclass(slots=True)
class MeanReversionParameters:
    """Parameter overrides for the mean reversion strategy."""

    entry_threshold: float
    exit_threshold: float
    max_hold_days: int
    target_position_pct: float
    stop_loss_pct: float | None = None
    lot_size: int = 1

    def as_kwargs(self) -> Dict[str, float | int | None]:
        return {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "max_hold_days": self.max_hold_days,
            "target_position_pct": self.target_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "lot_size": self.lot_size,
        }

    def label(self) -> str:
        stop = "none" if self.stop_loss_pct is None else f"{self.stop_loss_pct:.2f}"
        hold = "inf" if self.max_hold_days == 0 else str(self.max_hold_days)
        return (
            f"entry={self.entry_threshold:.0f}|exit={self.exit_threshold:.0f}|"
            f"hold={hold}|alloc={self.target_position_pct:.2f}|"
            f"stop={stop}|lot={self.lot_size}"
        )

    def validate(self) -> None:
        if self.entry_threshold < 0 or self.entry_threshold > 100:
            raise ValueError("entry_threshold must be within [0, 100]")
        if self.exit_threshold < 0 or self.exit_threshold > 100:
            raise ValueError("exit_threshold must be within [0, 100]")
        if self.entry_threshold >= self.exit_threshold:
            raise ValueError(
                "entry_threshold must be less than exit_threshold")
        if self.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if self.target_position_pct <= 0 or self.target_position_pct > 1:
            raise ValueError(
                "target_position_pct must be in the interval (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.stop_loss_pct is not None and (self.stop_loss_pct <= 0 or self.stop_loss_pct >= 1):
            raise ValueError("stop_loss_pct must be between 0 and 1")


@dataclass(slots=True)
class MeanReversionExperimentConfig:
    """Configuration for running a grid of mean reversion experiments."""

    store_path: Path
    universe: Sequence[str]
    initial_cash: float
    splits: Sequence[ExperimentSplit]
    parameters: Sequence[MeanReversionParameters]
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

    def strategy_config(self, parameters: MeanReversionParameters) -> MeanReversionConfig:
        return MeanReversionConfig(
            symbols=self.universe,
            initial_cash=self.initial_cash,
            entry_threshold=parameters.entry_threshold,
            exit_threshold=parameters.exit_threshold,
            max_hold_days=parameters.max_hold_days,
            target_position_pct=parameters.target_position_pct,
            stop_loss_pct=parameters.stop_loss_pct,
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
class MeanReversionExperimentRow:
    split: str
    parameter_label: str
    metrics: Dict[str, float]


@dataclass(slots=True)
class MeanReversionExperimentResult:
    rows: List[MeanReversionExperimentRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        records = []
        for row in self.rows:
            entry = {"split": row.split, "params": row.parameter_label}
            entry.update(row.metrics)
            records.append(entry)
        if not records:
            return pd.DataFrame(columns=["split", "params"])
        return pd.DataFrame.from_records(records)


def run_mean_reversion_experiment(config: MeanReversionExperimentConfig) -> MeanReversionExperimentResult:
    config.validate()
    store = ParquetBarStore(config.store_path)
    base_backtest = config.base_backtest_config()
    result = MeanReversionExperimentResult()

    for parameters in config.parameters:
        parameter_label = parameters.label()

        for split in config.splits:
            split_config = split.as_backtest_config(base_backtest)
            strategy_cfg = config.strategy_config(parameters)
            strategy = MeanReversionStrategy(strategy_cfg)
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
                MeanReversionExperimentRow(
                    split=split.name,
                    parameter_label=parameter_label,
                    metrics=metrics,
                )
            )

    return result


@dataclass(slots=True)
class OptimizationOutcome:
    best_parameters: MeanReversionParameters
    training_metrics: Dict[str, float]
    paper_metrics: Dict[str, float]
    parameter_frame: pd.DataFrame
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    paper_result: BacktestResult


@dataclass(slots=True)
class ParameterRange:
    """Inclusive integer range definition with step."""

    minimum: int
    maximum: int
    step: int = 1

    def values(self) -> List[int]:
        if self.step <= 0:
            raise ValueError("Range step must be positive")
        if self.maximum < self.minimum:
            raise ValueError("Range maximum must be >= minimum")
        count = ((self.maximum - self.minimum) // self.step) + 1
        return [self.minimum + idx * self.step for idx in range(count)]


@dataclass(slots=True)
class HoldRange(ParameterRange):
    include_infinite: bool = False

    def values(self) -> List[int]:  # type: ignore[override]
        base = ParameterRange.values(self)
        if self.include_infinite and 0 not in base:
            base.append(0)
        return base


@dataclass(slots=True)
class OptimizationParameterSpec:
    entry_threshold: ParameterRange
    exit_threshold: ParameterRange
    max_hold_days: HoldRange
    target_position_pct: ParameterRange
    stop_loss_pct: ParameterRange | None = None
    include_no_stop_loss: bool = True
    lot_size: int = 1


def default_parameter_grid() -> List[MeanReversionParameters]:
    spec = OptimizationParameterSpec(
        entry_threshold=ParameterRange(5, 15, 5),
        exit_threshold=ParameterRange(60, 80, 10),
        max_hold_days=HoldRange(3, 10, 2),
        target_position_pct=ParameterRange(10, 20, 10),
        stop_loss_pct=ParameterRange(5, 5, 1),
    )
    return generate_parameter_grid(spec)


def generate_parameter_grid(spec: OptimizationParameterSpec) -> List[MeanReversionParameters]:
    entry_values = spec.entry_threshold.values()
    exit_values = spec.exit_threshold.values()
    hold_values = spec.max_hold_days.values()
    target_values = spec.target_position_pct.values()
    stop_values: List[float | None]
    if spec.stop_loss_pct is None:
        stop_values = [None]
    else:
        percent_values = spec.stop_loss_pct.values()
        stop_values = [value / 100.0 for value in percent_values]
        if spec.include_no_stop_loss:
            stop_values.insert(0, None)

    target_fractions = [max(value, 1) / 100.0 for value in target_values]

    combinations = len(entry_values) * len(exit_values) * \
        len(hold_values) * len(target_fractions) * len(stop_values)
    if combinations > MAX_PARAMETER_COMBINATIONS:
        raise ValueError(
            f"Parameter grid too large ({combinations} combinations). Please narrow your ranges."
        )

    parameters: List[MeanReversionParameters] = []
    for entry in entry_values:
        for exit in exit_values:
            if entry >= exit:
                continue
            for hold in hold_values:
                hold_value = max(hold, 0)
                for target_pct in target_fractions:
                    for stop in stop_values:
                        parameters.append(
                            MeanReversionParameters(
                                entry_threshold=float(entry),
                                exit_threshold=float(exit),
                                max_hold_days=hold_value,
                                target_position_pct=float(target_pct),
                                stop_loss_pct=stop,
                                lot_size=spec.lot_size,
                            )
                        )
    if not parameters:
        raise ValueError(
            "No valid parameter combinations produced. Check your ranges.")
    return parameters


def _select_best_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Experiment frame is empty; no parameters evaluated.")
    return frame.sort_values(["cagr", "final_equity"], ascending=[False, False]).iloc[0]


def optimize_mean_reversion_parameters(
    store_path: Path,
    universe: Sequence[str],
    initial_cash: float,
    training_window: Tuple[date, date],
    paper_window: Tuple[date, date],
    *,
    bar_size: str = "1d",
    calendar_name: str = "XNYS",
    parameter_grid: Sequence[MeanReversionParameters] | None = None,
    parameter_spec: OptimizationParameterSpec | None = None,
) -> OptimizationOutcome:
    if parameter_grid is not None and parameter_spec is not None:
        raise ValueError(
            "Specify either parameter_grid or parameter_spec, not both.")

    if parameter_grid is not None:
        parameters = list(parameter_grid)
    elif parameter_spec is not None:
        parameters = generate_parameter_grid(parameter_spec)
    else:
        parameters = default_parameter_grid()

    if not parameters:
        raise ValueError(
            "Parameter grid must contain at least one configuration.")

    train_split = ExperimentSplit(
        name="train", start=training_window[0], end=training_window[1])
    paper_split = ExperimentSplit(
        name="paper", start=paper_window[0], end=paper_window[1])

    experiment_config = MeanReversionExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        splits=[train_split],
        parameters=parameters,
        bar_size=bar_size,
        calendar_name=calendar_name,
    )
    experiment_result = run_mean_reversion_experiment(experiment_config)
    frame = experiment_result.to_frame()
    best_row = _select_best_row(frame)

    param_lookup = {params.label(): params for params in parameters}
    best_label = best_row["params"]
    if best_label not in param_lookup:
        raise ValueError(
            f"Unable to locate parameters for label '{best_label}'.")
    best_params = param_lookup[best_label]

    store = ParquetBarStore(store_path)
    strategy_config = experiment_config.strategy_config(best_params)
    strategy = MeanReversionStrategy(strategy_config)

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
        k: float(best_row[k]) for k in best_row.index if k not in {"split", "params"}
    }
    training_metrics["net_profit"] = training_metrics.get(
        "final_equity", 0.0) - initial_cash

    paper_metrics = asdict(
        compute_backtest_metrics(
            paper_result,
            initial_cash=initial_cash,
            start=paper_split.start,
            end=paper_split.end,
        )
    )

    return OptimizationOutcome(
        best_parameters=best_params,
        training_metrics=training_metrics,
        paper_metrics=paper_metrics,
        parameter_frame=frame,
        training_window=(train_split.start, train_split.end),
        paper_window=(paper_split.start, paper_split.end),
        paper_result=paper_result,
    )
