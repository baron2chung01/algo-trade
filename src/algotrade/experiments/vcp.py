"""Experiment runner and optimizer for the VCP strategy."""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests

from ..backtest import BacktestConfig, BacktestEngine
from ..backtest.engine import BacktestResult
from ..backtest.fees import CommissionModel, SlippageModel
from ..data.stores.local import ParquetBarStore
from ..data.universe import latest_symbols, load_universe
from ..config import AppSettings
from ..strategies import VCPConfig, VCPStrategy
from .breakout import FloatRange
from .mean_reversion import ExperimentSplit, HoldRange, ParameterRange
from .metrics import compute_backtest_metrics

MAX_PARAMETER_COMBINATIONS = 10_000
_TECH_UNIVERSE_CACHE = "nasdaq_technology_universe.json"

_SCAN_TIMEFRAME_PRESETS: Dict[str, Dict[str, float | int | None]] = {
    "short": {
        "base_lookback_days": 45,
        "pivot_lookback_days": 4,
        "min_contractions": 3,
        "max_contraction_pct": 0.14,
        "contraction_decay": 0.7,
        "breakout_buffer_pct": 0.002,
        "volume_squeeze_ratio": 0.75,
        "breakout_volume_ratio": 2.0,
        "volume_lookback_days": 18,
        "trend_ma_period": 40,
        "stop_loss_r_multiple": 1.0,
        "profit_target_r_multiple": 2.5,
        "trailing_stop_r_multiple": 1.5,
        "max_hold_days": 0,
        "target_position_pct": 0.15,
        "lot_size": 1,
        "cash_reserve_pct": 0.1,
    },
    "medium": {
        "base_lookback_days": 60,
        "pivot_lookback_days": 5,
        "min_contractions": 3,
        "max_contraction_pct": 0.16,
        "contraction_decay": 0.72,
        "breakout_buffer_pct": 0.0025,
        "volume_squeeze_ratio": 0.78,
        "breakout_volume_ratio": 2.1,
        "volume_lookback_days": 24,
        "trend_ma_period": 50,
        "stop_loss_r_multiple": 1.0,
        "profit_target_r_multiple": 2.5,
        "trailing_stop_r_multiple": 1.5,
        "max_hold_days": 0,
        "target_position_pct": 0.15,
        "lot_size": 1,
        "cash_reserve_pct": 0.1,
    },
    "long": {
        "base_lookback_days": 75,
        "pivot_lookback_days": 7,
        "min_contractions": 4,
        "max_contraction_pct": 0.18,
        "contraction_decay": 0.75,
        "breakout_buffer_pct": 0.003,
        "volume_squeeze_ratio": 0.8,
        "breakout_volume_ratio": 2.2,
        "volume_lookback_days": 30,
        "trend_ma_period": 60,
        "stop_loss_r_multiple": 1.0,
        "profit_target_r_multiple": 2.8,
        "trailing_stop_r_multiple": 1.6,
        "max_hold_days": 0,
        "target_position_pct": 0.15,
        "lot_size": 1,
        "cash_reserve_pct": 0.1,
    },
}


@dataclass(slots=True)
class VCPParameters:
    base_lookback_days: int
    pivot_lookback_days: int
    min_contractions: int
    max_contraction_pct: float
    contraction_decay: float
    breakout_buffer_pct: float
    volume_squeeze_ratio: float
    breakout_volume_ratio: float
    volume_lookback_days: int
    trend_ma_period: int
    stop_loss_r_multiple: float
    profit_target_r_multiple: float
    trailing_stop_r_multiple: float | None
    max_hold_days: int
    target_position_pct: float
    lot_size: int
    cash_reserve_pct: float

    def label(self) -> str:
        trailing = "none" if self.trailing_stop_r_multiple is None else f"{self.trailing_stop_r_multiple:.2f}"
        stop = f"{self.stop_loss_r_multiple:.2f}"
        target = f"{self.profit_target_r_multiple:.2f}"
        return (
            f"base={self.base_lookback_days}|pivot={self.pivot_lookback_days}|"
            f"contractions={self.min_contractions}|max_drop={self.max_contraction_pct:.3f}|"
            f"decay={self.contraction_decay:.2f}|buffer={self.breakout_buffer_pct:.3f}|"
            f"vol_squeeze={self.volume_squeeze_ratio:.2f}|vol_breakout={self.breakout_volume_ratio:.2f}|"
            f"trend_ma={self.trend_ma_period}|stopR={stop}|targetR={target}|trailR={trailing}|"
            f"hold={self.max_hold_days}|target_pct={self.target_position_pct:.3f}|lot={self.lot_size}"
        )

    def validate(self) -> None:
        if self.base_lookback_days <= 0:
            raise ValueError("base_lookback_days must be positive")
        if self.pivot_lookback_days <= 0:
            raise ValueError("pivot_lookback_days must be positive")
        if self.min_contractions <= 0:
            raise ValueError("min_contractions must be positive")
        if not (0 < self.max_contraction_pct <= 1):
            raise ValueError("max_contraction_pct must be within (0, 1]")
        if not (0 < self.contraction_decay <= 1):
            raise ValueError("contraction_decay must be within (0, 1]")
        if self.breakout_buffer_pct < 0:
            raise ValueError("breakout_buffer_pct cannot be negative")
        if not (0 < self.volume_squeeze_ratio <= 1):
            raise ValueError("volume_squeeze_ratio must be within (0, 1]")
        if self.breakout_volume_ratio <= 0:
            raise ValueError("breakout_volume_ratio must be positive")
        if self.volume_lookback_days <= 0:
            raise ValueError("volume_lookback_days must be positive")
        if self.trend_ma_period <= 0:
            raise ValueError("trend_ma_period must be positive")
        if self.stop_loss_r_multiple <= 0:
            raise ValueError("stop_loss_r_multiple must be positive")
        if self.profit_target_r_multiple <= 0:
            raise ValueError("profit_target_r_multiple must be positive")
        if self.trailing_stop_r_multiple is not None and self.trailing_stop_r_multiple <= 0:
            raise ValueError(
                "trailing_stop_r_multiple must be positive when provided")
        if self.max_hold_days < 0:
            raise ValueError("max_hold_days cannot be negative")
        if not (0 < self.target_position_pct <= 1):
            raise ValueError("target_position_pct must be within (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.cash_reserve_pct < 0 or self.cash_reserve_pct >= 1:
            raise ValueError("cash_reserve_pct must be in [0, 1)")


@dataclass(slots=True)
class VCPParameterSpec:
    base_lookback_days: ParameterRange = field(
        default_factory=lambda: ParameterRange(45, 60, 15)
    )
    pivot_lookback_days: ParameterRange = field(
        default_factory=lambda: ParameterRange(4, 6, 2)
    )
    min_contractions: ParameterRange = field(
        default_factory=lambda: ParameterRange(3, 3, 1)
    )
    max_contraction_pct: FloatRange = field(
        default_factory=lambda: FloatRange(0.12, 0.16, 0.04)
    )
    contraction_decay: FloatRange = field(
        default_factory=lambda: FloatRange(0.6, 0.8, 0.2)
    )
    breakout_buffer_pct: FloatRange = field(
        default_factory=lambda: FloatRange(0.001, 0.003, 0.002)
    )
    volume_squeeze_ratio: FloatRange = field(
        default_factory=lambda: FloatRange(0.65, 0.85, 0.2)
    )
    breakout_volume_ratio: FloatRange = field(
        default_factory=lambda: FloatRange(1.8, 2.1, 0.3)
    )
    volume_lookback_days: ParameterRange = field(
        default_factory=lambda: ParameterRange(18, 24, 6)
    )
    trend_ma_period: ParameterRange = field(
        default_factory=lambda: ParameterRange(45, 60, 15)
    )
    stop_loss_r_multiple: FloatRange = field(
        default_factory=lambda: FloatRange(0.9, 1.1, 0.2)
    )
    profit_target_r_multiple: FloatRange = field(
        default_factory=lambda: FloatRange(2.0, 2.5, 0.5)
    )
    trailing_stop_r_multiple: FloatRange | None = field(
        default_factory=lambda: FloatRange(1.5, 1.5, 0.1)
    )
    include_no_trailing_stop: bool = True
    max_hold_days: HoldRange = field(
        default_factory=lambda: HoldRange(0, 0, 1, include_infinite=True)
    )
    target_position_pct: ParameterRange = field(
        default_factory=lambda: ParameterRange(15, 15, 1)
    )
    lot_size: int = 1
    cash_reserve_pct: float = 0.1


@dataclass(slots=True)
class VCPExperimentConfig:
    store_path: Path
    universe: Sequence[str]
    initial_cash: float
    splits: Sequence[ExperimentSplit]
    parameters: Sequence[VCPParameters]
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

    def strategy_config(self, params: VCPParameters) -> VCPConfig:
        return VCPConfig(
            symbols=self.universe,
            initial_cash=self.initial_cash,
            base_lookback_days=params.base_lookback_days,
            pivot_lookback_days=params.pivot_lookback_days,
            min_contractions=params.min_contractions,
            max_contraction_pct=params.max_contraction_pct,
            contraction_decay=params.contraction_decay,
            breakout_buffer_pct=params.breakout_buffer_pct,
            volume_squeeze_ratio=params.volume_squeeze_ratio,
            breakout_volume_ratio=params.breakout_volume_ratio,
            volume_lookback_days=params.volume_lookback_days,
            trend_ma_period=params.trend_ma_period,
            stop_loss_r_multiple=params.stop_loss_r_multiple,
            profit_target_r_multiple=params.profit_target_r_multiple,
            trailing_stop_r_multiple=params.trailing_stop_r_multiple,
            max_hold_days=params.max_hold_days,
            target_position_pct=params.target_position_pct,
            lot_size=params.lot_size,
            cash_reserve_pct=params.cash_reserve_pct,
            max_positions=max(len(self.universe), 5),
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
        seen: set[str] = set()
        for split in self.splits:
            split.validate()
            key = split.name.lower()
            if key in seen:
                raise ValueError(
                    f"Duplicate split name detected: '{split.name}'")
            seen.add(key)
        for params in self.parameters:
            params.validate()


@dataclass(slots=True)
class VCPExperimentRow:
    split: str
    parameter_label: str
    metrics: Dict[str, float]


@dataclass(slots=True)
class VCPExperimentResult:
    rows: List[VCPExperimentRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        records = []
        for row in self.rows:
            payload = {"split": row.split, "params": row.parameter_label}
            payload.update(row.metrics)
            records.append(payload)
        if not records:
            return pd.DataFrame(columns=["split", "params"])
        return pd.DataFrame.from_records(records)


@dataclass(slots=True)
class VCPOptimizationOutcome:
    best_parameters: VCPParameters
    training_metrics: Dict[str, float]
    paper_metrics: Dict[str, float]
    parameter_frame: pd.DataFrame
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    paper_result: BacktestResult
    paper_annotations: Dict[str, List[Dict[str, float | str]]]


def default_vcp_spec() -> VCPParameterSpec:
    return VCPParameterSpec()


@dataclass(slots=True)
class VCPScanCandidate:
    symbol: str
    close_price: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_share: float
    resistance: float
    base_low: float
    breakout_timestamp: datetime
    reward_to_risk: float
    volume: float


@dataclass(slots=True)
class VCPScanSummary:
    timeframe: str
    parameters: VCPParameters
    symbols_scanned: int
    candidates: List[VCPScanCandidate]
    warnings: List[str]
    analysis_timestamp: datetime | None


@dataclass(slots=True)
class VCPPatternDetection:
    breakout_timestamp: datetime
    base_start: datetime
    base_end: datetime
    entry_price: float
    stop_price: float
    target_price: float
    resistance: float
    base_low: float
    breakout_price: float
    breakout_volume: float
    risk_per_share: float
    reward_to_risk: float


@dataclass(slots=True)
class VCPPatternSeries:
    symbol: str
    parameters: VCPParameters
    frame: pd.DataFrame
    detections: List[VCPPatternDetection]
    warnings: List[str] = field(default_factory=list)
    analysis_start: datetime | None = None
    analysis_end: datetime | None = None


@dataclass(slots=True)
class VCPPatternHistorySummary:
    results: Dict[str, VCPPatternSeries] = field(default_factory=dict)
    missing: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def generate_vcp_parameter_grid(spec: VCPParameterSpec) -> List[VCPParameters]:
    base_values = spec.base_lookback_days.values()
    pivot_values = spec.pivot_lookback_days.values()
    contraction_counts = spec.min_contractions.values()
    max_drops = spec.max_contraction_pct.values()
    decays = spec.contraction_decay.values()
    buffers = spec.breakout_buffer_pct.values()
    squeeze = spec.volume_squeeze_ratio.values()
    breakout_volumes = spec.breakout_volume_ratio.values()
    volume_lookbacks = spec.volume_lookback_days.values()
    trend_periods = spec.trend_ma_period.values()
    stop_r_values = spec.stop_loss_r_multiple.values()
    target_r_values = spec.profit_target_r_multiple.values()
    trailing_values = (
        [None] + spec.trailing_stop_r_multiple.values()
        if spec.trailing_stop_r_multiple and spec.include_no_trailing_stop
        else (spec.trailing_stop_r_multiple.values() if spec.trailing_stop_r_multiple else [None])
    )
    hold_values = spec.max_hold_days.values()
    target_pct_values = [
        max(value, 1) / 100.0 for value in spec.target_position_pct.values()]

    combinations = (
        len(base_values)
        * len(pivot_values)
        * len(contraction_counts)
        * len(max_drops)
        * len(decays)
        * len(buffers)
        * len(squeeze)
        * len(breakout_volumes)
        * len(volume_lookbacks)
        * len(trend_periods)
        * len(stop_r_values)
        * len(target_r_values)
        * len(trailing_values)
        * len(hold_values)
        * len(target_pct_values)
    )
    if combinations > MAX_PARAMETER_COMBINATIONS:
        raise ValueError(
            f"Parameter grid too large ({combinations} combinations). Please narrow your ranges."
        )

    parameters: List[VCPParameters] = []
    for base in base_values:
        for pivot in pivot_values:
            for contraction_count in contraction_counts:
                for max_drop in max_drops:
                    for decay in decays:
                        for buffer in buffers:
                            for squeeze_ratio in squeeze:
                                for breakout_ratio in breakout_volumes:
                                    for vol_lookback in volume_lookbacks:
                                        for trend in trend_periods:
                                            for stop_r in stop_r_values:
                                                for target_r in target_r_values:
                                                    for trailing in trailing_values:
                                                        for hold in hold_values:
                                                            hold_value = max(
                                                                hold, 0)
                                                            for target_pct in target_pct_values:
                                                                params = VCPParameters(
                                                                    base_lookback_days=int(
                                                                        base),
                                                                    pivot_lookback_days=int(
                                                                        pivot),
                                                                    min_contractions=int(
                                                                        contraction_count),
                                                                    max_contraction_pct=float(
                                                                        max_drop),
                                                                    contraction_decay=float(
                                                                        decay),
                                                                    breakout_buffer_pct=float(
                                                                        buffer),
                                                                    volume_squeeze_ratio=float(
                                                                        squeeze_ratio),
                                                                    breakout_volume_ratio=float(
                                                                        breakout_ratio),
                                                                    volume_lookback_days=int(
                                                                        vol_lookback),
                                                                    trend_ma_period=int(
                                                                        trend),
                                                                    stop_loss_r_multiple=float(
                                                                        stop_r),
                                                                    profit_target_r_multiple=float(
                                                                        target_r),
                                                                    trailing_stop_r_multiple=float(
                                                                        trailing) if trailing is not None else None,
                                                                    max_hold_days=hold_value,
                                                                    target_position_pct=float(
                                                                        target_pct),
                                                                    lot_size=spec.lot_size,
                                                                    cash_reserve_pct=spec.cash_reserve_pct,
                                                                )
                                                                parameters.append(
                                                                    params)
    if not parameters:
        raise ValueError(
            "No parameter combinations generated; check specification ranges.")
    return parameters


def _parameter_options_from_spec(spec: VCPParameterSpec) -> Dict[str, List[float | int | None]]:
    base_values = [int(value) for value in spec.base_lookback_days.values()]
    pivot_values = [int(value) for value in spec.pivot_lookback_days.values()]
    contraction_counts = [int(value)
                          for value in spec.min_contractions.values()]
    max_drops = [float(value) for value in spec.max_contraction_pct.values()]
    decays = [float(value) for value in spec.contraction_decay.values()]
    buffers = [float(value) for value in spec.breakout_buffer_pct.values()]
    squeeze = [float(value) for value in spec.volume_squeeze_ratio.values()]
    breakout_volumes = [float(value)
                        for value in spec.breakout_volume_ratio.values()]
    volume_lookbacks = [int(value)
                        for value in spec.volume_lookback_days.values()]
    trend_periods = [int(value) for value in spec.trend_ma_period.values()]
    stop_r_values = [float(value)
                     for value in spec.stop_loss_r_multiple.values()]
    target_r_values = [float(value)
                       for value in spec.profit_target_r_multiple.values()]
    trailing_values: List[float | None]
    if spec.trailing_stop_r_multiple is None:
        trailing_values = [None]
    else:
        trailing_values = [float(value)
                           for value in spec.trailing_stop_r_multiple.values()]
        if spec.include_no_trailing_stop and None not in trailing_values:
            trailing_values.insert(0, None)
    hold_values = [max(int(value), 0)
                   for value in spec.max_hold_days.values()]
    target_pct_values = [max(int(value), 1) / 100.0
                         for value in spec.target_position_pct.values()]

    return {
        "base_lookback_days": base_values,
        "pivot_lookback_days": pivot_values,
        "min_contractions": contraction_counts,
        "max_contraction_pct": max_drops,
        "contraction_decay": decays,
        "breakout_buffer_pct": buffers,
        "volume_squeeze_ratio": squeeze,
        "breakout_volume_ratio": breakout_volumes,
        "volume_lookback_days": volume_lookbacks,
        "trend_ma_period": trend_periods,
        "stop_loss_r_multiple": stop_r_values,
        "profit_target_r_multiple": target_r_values,
        "trailing_stop_r_multiple": trailing_values,
        "max_hold_days": hold_values,
        "target_position_pct": target_pct_values,
        "lot_size": [spec.lot_size],
        "cash_reserve_pct": [spec.cash_reserve_pct],
    }


def _values_to_vcp_parameters(values: Dict[str, float | int | None]) -> VCPParameters:
    return VCPParameters(
        base_lookback_days=int(values["base_lookback_days"]),
        pivot_lookback_days=int(values["pivot_lookback_days"]),
        min_contractions=int(values["min_contractions"]),
        max_contraction_pct=float(values["max_contraction_pct"]),
        contraction_decay=float(values["contraction_decay"]),
        breakout_buffer_pct=float(values["breakout_buffer_pct"]),
        volume_squeeze_ratio=float(values["volume_squeeze_ratio"]),
        breakout_volume_ratio=float(values["breakout_volume_ratio"]),
        volume_lookback_days=int(values["volume_lookback_days"]),
        trend_ma_period=int(values["trend_ma_period"]),
        stop_loss_r_multiple=float(values["stop_loss_r_multiple"]),
        profit_target_r_multiple=float(values["profit_target_r_multiple"]),
        trailing_stop_r_multiple=None
        if values["trailing_stop_r_multiple"] is None
        else float(values["trailing_stop_r_multiple"]),
        max_hold_days=int(values["max_hold_days"]),
        target_position_pct=float(values["target_position_pct"]),
        lot_size=int(values["lot_size"]),
        cash_reserve_pct=float(values["cash_reserve_pct"]),
    )


def _random_parameter_from_spec(
    spec: VCPParameterSpec, rng: random.Random
) -> VCPParameters:
    options = _parameter_options_from_spec(spec)
    return _random_parameter_from_options(options, rng)


def _random_parameter_from_options(
    options: Dict[str, List[float | int | None]], rng: random.Random
) -> VCPParameters:
    selection: Dict[str, float | int | None] = {
        key: rng.choice(values) for key, values in options.items()
    }
    params = _values_to_vcp_parameters(selection)
    params.validate()
    return params


def _closest_option_index(values: Sequence[float | int | None], current: float | int | None) -> int:
    if current is None:
        for idx, value in enumerate(values):
            if value is None:
                return idx
        return 0
    best_idx = 0
    best_diff = float("inf")
    current_float = float(current)
    for idx, option in enumerate(values):
        if option is None:
            continue
        diff = abs(float(option) - current_float)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def _mutate_parameter(
    current: VCPParameters,
    options: Dict[str, List[float | int | None]],
    rng: random.Random,
    *,
    max_changes: int = 3,
) -> VCPParameters:
    mutable_keys = [key for key, values in options.items() if len(values) > 1]
    if not mutable_keys:
        return current

    attempts = 0
    while attempts < 8:
        attempts += 1
        candidate = replace(current)
        num_changes = rng.randint(1, min(max_changes, len(mutable_keys)))
        selected_keys = rng.sample(mutable_keys, num_changes)
        changed = False
        for key in selected_keys:
            values = options[key]
            current_value = getattr(candidate, key)
            if isinstance(current_value, float) and key in {
                "base_lookback_days",
                "pivot_lookback_days",
                "min_contractions",
                "volume_lookback_days",
                "trend_ma_period",
                "max_hold_days",
                "lot_size",
            }:
                current_value = int(round(current_value))
            if key == "target_position_pct":
                current_value = float(current_value)

            index = _closest_option_index(values, current_value)
            neighbor_indices = [idx for idx in (index - 1, index + 1)
                                if 0 <= idx < len(values)]
            if neighbor_indices:
                new_index = rng.choice(neighbor_indices)
            else:
                available = [idx for idx in range(len(values)) if idx != index]
                if not available:
                    continue
                new_index = rng.choice(available)

            new_value = values[new_index]
            if new_value == current_value:
                continue
            setattr(candidate, key, new_value)
            changed = True

        if changed:
            candidate.validate()
            return candidate

    # Fallback to a fresh random sample if we could not mutate successfully
    return _random_parameter_from_options(options, rng)


def _score_metrics(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("cagr", 0.0)),
        float(metrics.get("sharpe", 0.0)),
        float(metrics.get("final_equity", 0.0)),
    )


def _evaluate_vcp_training_metrics(
    store: ParquetBarStore,
    train_config: BacktestConfig,
    params: VCPParameters,
    universe: Sequence[str],
    initial_cash: float,
    *,
    commission_model: CommissionModel | None,
    slippage_model: SlippageModel | None,
) -> Dict[str, float]:
    strategy_config = VCPConfig(
        symbols=universe,
        initial_cash=initial_cash,
        base_lookback_days=params.base_lookback_days,
        pivot_lookback_days=params.pivot_lookback_days,
        min_contractions=params.min_contractions,
        max_contraction_pct=params.max_contraction_pct,
        contraction_decay=params.contraction_decay,
        breakout_buffer_pct=params.breakout_buffer_pct,
        volume_squeeze_ratio=params.volume_squeeze_ratio,
        breakout_volume_ratio=params.breakout_volume_ratio,
        volume_lookback_days=params.volume_lookback_days,
        trend_ma_period=params.trend_ma_period,
        stop_loss_r_multiple=params.stop_loss_r_multiple,
        profit_target_r_multiple=params.profit_target_r_multiple,
        trailing_stop_r_multiple=params.trailing_stop_r_multiple,
        max_hold_days=params.max_hold_days,
        target_position_pct=params.target_position_pct,
        lot_size=params.lot_size,
        cash_reserve_pct=params.cash_reserve_pct,
        max_positions=max(len(universe), 5),
    )
    strategy = VCPStrategy(strategy_config)
    engine = BacktestEngine(config=train_config, store=store)
    result = engine.run(strategy)
    metrics = asdict(
        compute_backtest_metrics(
            result,
            initial_cash=train_config.initial_cash,
            start=train_config.start,
            end=train_config.end,
        )
    )
    return {key: float(value) for key, value in metrics.items()}


def _resolve_scan_parameters(
    timeframe: str,
    overrides: Dict[str, float | int | None] | None = None,
) -> VCPParameters:
    key = timeframe.lower()
    if key not in _SCAN_TIMEFRAME_PRESETS:
        raise ValueError(
            f"Unsupported VCP scan timeframe '{timeframe}'. "
            f"Valid options: {sorted(_SCAN_TIMEFRAME_PRESETS.keys())}."
        )
    preset = dict(_SCAN_TIMEFRAME_PRESETS[key])
    if overrides:
        for name, value in overrides.items():
            if value is None:
                continue
            if name not in preset:
                raise ValueError(f"Unsupported VCP scan override '{name}'.")
            preset[name] = value

    params = VCPParameters(
        base_lookback_days=int(preset["base_lookback_days"]),
        pivot_lookback_days=int(preset["pivot_lookback_days"]),
        min_contractions=int(preset["min_contractions"]),
        max_contraction_pct=float(preset["max_contraction_pct"]),
        contraction_decay=float(preset["contraction_decay"]),
        breakout_buffer_pct=float(preset["breakout_buffer_pct"]),
        volume_squeeze_ratio=float(preset["volume_squeeze_ratio"]),
        breakout_volume_ratio=float(preset["breakout_volume_ratio"]),
        volume_lookback_days=int(preset["volume_lookback_days"]),
        trend_ma_period=int(preset["trend_ma_period"]),
        stop_loss_r_multiple=float(preset["stop_loss_r_multiple"]),
        profit_target_r_multiple=float(preset["profit_target_r_multiple"]),
        trailing_stop_r_multiple=float(preset["trailing_stop_r_multiple"])
        if preset.get("trailing_stop_r_multiple") is not None
        else None,
        max_hold_days=int(preset.get("max_hold_days", 0)),
        target_position_pct=float(preset.get("target_position_pct", 0.15)),
        lot_size=int(preset.get("lot_size", 1)),
        cash_reserve_pct=float(preset.get("cash_reserve_pct", 0.1)),
    )
    params.validate()
    return params


def _resolve_scan_universe(
    store: ParquetBarStore,
    bar_size: str,
    symbols: Sequence[str] | None,
) -> tuple[List[str], List[str], List[str]]:
    available = {symbol.upper() for symbol in store.list_symbols(bar_size)}
    warnings: List[str] = []

    if not available:
        raise ValueError(
            "No cached historical bars available for VCP scanning.")

    if symbols:
        requested = [symbol.upper() for symbol in symbols]
        selected = [symbol for symbol in requested if symbol in available]
        missing = [symbol for symbol in requested if symbol not in available]
        return selected, missing, warnings

    settings = AppSettings()
    universe_path = settings.data_paths.raw / \
        "universe" / "nasdaq" / "membership.parquet"
    if universe_path.exists():
        snapshots = load_universe(universe_path)
        if snapshots:
            nasdaq_symbols = sorted(latest_symbols(snapshots))
            selected = [
                symbol for symbol in nasdaq_symbols if symbol in available]
            missing = [
                symbol for symbol in nasdaq_symbols if symbol not in available]
            if selected:
                return selected, missing, warnings
            warnings.append(
                "NASDAQ universe present but no symbols overlap cached OHLC data; falling back to cached list.")
        else:
            warnings.append(
                "NASDAQ universe file is empty; falling back to cached symbols.")
    else:
        warnings.append(
            "NASDAQ universe membership not found; scanning all cached symbols.")

    return sorted(available), [], warnings


def _load_cached_technology_symbols(cache_path: Path) -> tuple[list[str], list[str]]:
    if not cache_path.exists():
        return [], []
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        symbols = payload.get("symbols", [])
        if not isinstance(symbols, list):
            raise ValueError("cached universe symbols must be a list")
        normalized = [str(symbol).strip().upper()
                      for symbol in symbols if str(symbol).strip()]
        return sorted(set(normalized)), []
    except Exception as exc:  # pragma: no cover - cache parse errors
        return [], [f"Failed to read cached technology universe: {exc}"]


def _fetch_polygon_technology_symbols(settings: AppSettings) -> tuple[list[str], list[str]]:
    try:
        api_key = settings.require_polygon_api_key()
    except RuntimeError as exc:
        return [], [str(exc)]

    base_url = os.getenv("POLYGON_BASE_URL",
                         "https://api.polygon.io").rstrip("/")
    params = {
        "market": "stocks",
        "primary_exchange": "XNAS",
        "active": "true",
        "sector": "Technology",
        "limit": 1000,
        "order": "asc",
        "sort": "ticker",
    }
    url = f"{base_url}/v3/reference/tickers"
    collected: set[str] = set()
    warnings: list[str] = []

    while url:
        try:
            response = requests.get(
                url,
                params=params if "?" not in url else None,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            warnings.append(
                f"Failed to fetch technology universe from Polygon: {exc}")
            return sorted(collected), warnings
        except ValueError as exc:  # JSON decode error
            warnings.append(
                f"Invalid response decoding technology universe: {exc}")
            return sorted(collected), warnings

        results = payload.get("results") or []
        for item in results:
            ticker = item.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                collected.add(ticker.strip().upper())

        next_url = payload.get("next_url")
        if next_url:
            url = next_url
            params = None
        else:
            url = None

    return sorted(collected), warnings


def _scrape_nasdaq_technology_symbols() -> tuple[list[str], list[str]]:
    """Scrape public sources for NASDAQ technology tickers as a fallback."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def _normalize_columns(table: pd.DataFrame) -> list[str]:
        normalized: list[str] = []
        for column in table.columns:
            if isinstance(column, tuple):
                parts = [str(part).strip()
                         for part in column if str(part).strip()]
                normalized.append(" ".join(parts))
            else:
                normalized.append(str(column).strip())
        return normalized

    def _scrape_wikipedia() -> tuple[list[str], list[str]]:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            return [], [f"Failed to download NASDAQ-100 listing from Wikipedia: {exc}"]

        try:
            tables = pd.read_html(StringIO(response.text))
        except ValueError as exc:
            return [], [f"Unable to parse NASDAQ-100 listing from Wikipedia: {exc}"]

        extracted: set[str] = set()
        for table in tables:
            if table.empty:
                continue
            table.columns = _normalize_columns(table)
            if "Ticker" not in table.columns:
                continue

            if "GICS Sector" in table.columns:
                sectors = table["GICS Sector"].astype(str).str.strip()
                mask = sectors.str.contains(
                    "Information Technology", case=False, na=False)
            elif "Sector" in table.columns:
                sectors = table["Sector"].astype(str).str.strip()
                mask = sectors.str.contains("Technology", case=False, na=False)
            else:
                continue

            if not mask.any():
                continue

            tickers = table.loc[mask, "Ticker"].dropna().astype(str)
            for value in tickers:
                token = value.strip().upper()
                cleaned = token.replace(".", "").replace("-", "")
                if token and cleaned and cleaned.isalnum():
                    extracted.add(token)

        if not extracted:
            return [], ["Wikipedia NASDAQ-100 listing did not expose technology tickers."]

        return sorted(extracted), []

    def _scrape_stockmonitor() -> tuple[list[str], list[str]]:
        url = "https://www.stockmonitor.com/sector/technology/"
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            return [], [f"Failed to download technology listing from StockMonitor: {exc}"]

        try:
            tables = pd.read_html(StringIO(response.text))
        except ValueError as exc:
            return [], [f"Unable to parse technology listing from StockMonitor: {exc}"]

        extracted: set[str] = set()
        for table in tables:
            if table.empty:
                continue
            table.columns = _normalize_columns(table)
            for column in ("Symbol", "Ticker", "Ticker Symbol", "Code"):
                if column in table.columns:
                    symbols = table[column].dropna().astype(str)
                    for value in symbols:
                        token = value.strip().upper()
                        cleaned = token.replace(".", "").replace("-", "")
                        if token and cleaned and cleaned.isalnum():
                            extracted.add(token)
                    break

        if not extracted:
            return [], ["StockMonitor technology listing did not expose recognizable symbol columns."]

        return sorted(extracted), []

    warnings: list[str] = []
    sources: list[tuple[str, Callable[[], tuple[list[str], list[str]]]]] = [
        ("Wikipedia NASDAQ-100", _scrape_wikipedia),
        ("StockMonitor Technology", _scrape_stockmonitor),
    ]

    for source_name, scraper in sources:
        symbols, source_warnings = scraper()
        warnings.extend(source_warnings)
        if symbols:
            warnings.append(f"Scraped technology symbols from {source_name}.")
            return symbols, warnings

    return [], warnings


def _load_technology_universe(
    settings: AppSettings,
    *,
    force_refresh: bool = False,
) -> tuple[list[str], list[str]]:
    cache_path = settings.data_paths.cache / _TECH_UNIVERSE_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh:
        symbols, warnings = _load_cached_technology_symbols(cache_path)
        if symbols:
            return symbols, warnings
    else:
        symbols, warnings = [], []

    fetched, fetch_warnings = _fetch_polygon_technology_symbols(settings)
    warnings.extend(fetch_warnings)
    if fetched:
        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "symbols": fetched,
                        "source": "polygon",
                        "cached_at": datetime.now(timezone.utc).isoformat(),
                    },
                    handle,
                )
        except Exception as exc:  # pragma: no cover - cache write errors
            warnings.append(f"Failed to cache technology universe: {exc}")
        return fetched, warnings

    scraped, scrape_warnings = _scrape_nasdaq_technology_symbols()
    warnings.extend(scrape_warnings)
    if scraped:
        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "symbols": scraped,
                        "source": "web",
                        "cached_at": datetime.now(timezone.utc).isoformat(),
                    },
                    handle,
                )
        except Exception as exc:  # pragma: no cover - cache write errors
            warnings.append(
                f"Failed to cache scraped technology universe: {exc}")
        return scraped, warnings

    if not force_refresh:
        return [], warnings

    cached_symbols, cache_warnings = _load_cached_technology_symbols(
        cache_path)
    warnings.extend(cache_warnings)
    return cached_symbols, warnings


def default_vcp_scan_universe(
    store_path: Path,
    *,
    bar_size: str = "1d",
) -> tuple[List[str], List[str], List[str]]:
    """Resolve the default NASDAQ technology-sector universe for VCP scanning."""

    settings = AppSettings()
    tech_symbols, tech_warnings = _load_technology_universe(settings)
    store = ParquetBarStore(store_path)

    universe_symbols: Sequence[str] | None = tech_symbols if tech_symbols else None
    selected, missing, warnings = _resolve_scan_universe(
        store, bar_size, universe_symbols)

    all_warnings = list(tech_warnings)
    all_warnings.extend(warnings)

    if universe_symbols and missing:
        all_warnings.append(
            f"{len(missing)} technology-sector symbols missing cached bars; consider fetching latest data."
        )

    if not selected and universe_symbols:
        fallback_selected, _fallback_missing, fallback_warnings = _resolve_scan_universe(
            store, bar_size, symbols=None
        )
        all_warnings.append(
            "Technology-sector universe unavailable in cache; falling back to all cached symbols."
        )
        all_warnings.extend(fallback_warnings)
        selected = fallback_selected
        missing = []

    return selected, missing, all_warnings


def technology_universe_symbols(
    *,
    force_refresh: bool = False,
) -> tuple[List[str], List[str]]:
    """Return the NASDAQ technology universe symbols with optional refresh."""

    settings = AppSettings()
    symbols, warnings = _load_technology_universe(
        settings, force_refresh=force_refresh)
    return symbols, warnings


def refresh_technology_universe() -> tuple[List[str], List[str]]:
    """Force refresh the NASDAQ technology universe from Polygon."""

    return technology_universe_symbols(force_refresh=True)


def _build_vcp_scan_strategy(symbol: str, parameters: VCPParameters) -> VCPStrategy:
    config = VCPConfig(
        symbols=[symbol],
        initial_cash=100_000.0,
        base_lookback_days=parameters.base_lookback_days,
        pivot_lookback_days=parameters.pivot_lookback_days,
        min_contractions=parameters.min_contractions,
        max_contraction_pct=parameters.max_contraction_pct,
        contraction_decay=parameters.contraction_decay,
        breakout_buffer_pct=parameters.breakout_buffer_pct,
        volume_squeeze_ratio=parameters.volume_squeeze_ratio,
        breakout_volume_ratio=parameters.breakout_volume_ratio,
        volume_lookback_days=parameters.volume_lookback_days,
        trend_ma_period=parameters.trend_ma_period,
        stop_loss_r_multiple=parameters.stop_loss_r_multiple,
        profit_target_r_multiple=parameters.profit_target_r_multiple,
        trailing_stop_r_multiple=parameters.trailing_stop_r_multiple,
        max_hold_days=parameters.max_hold_days,
        target_position_pct=parameters.target_position_pct,
        lot_size=parameters.lot_size,
        cash_reserve_pct=parameters.cash_reserve_pct,
        max_positions=1,
    )
    return VCPStrategy(config)


def scan_vcp_candidates(
    store_path: Path,
    *,
    timeframe: str,
    overrides: Dict[str, float | int | None] | None = None,
    bar_size: str = "1d",
    symbols: Sequence[str] | None = None,
    max_candidates: int = 50,
) -> VCPScanSummary:
    parameters = _resolve_scan_parameters(timeframe, overrides)
    store = ParquetBarStore(store_path)
    if symbols:
        selected_symbols, missing_symbols, universe_warnings = _resolve_scan_universe(
            store, bar_size, symbols
        )
    else:
        selected_symbols, missing_symbols, universe_warnings = default_vcp_scan_universe(
            store_path, bar_size=bar_size
        )

    if not selected_symbols:
        raise ValueError(
            "None of the requested or default universe symbols are available in the local store."
        )

    candidates: List[VCPScanCandidate] = []
    warnings: List[str] = list(universe_warnings)
    analysis_timestamp: datetime | None = None

    for symbol in selected_symbols:
        try:
            frame = store.load(symbol, bar_size)
        except FileNotFoundError:
            warnings.append(f"Missing historical data for {symbol}.")
            continue
        if frame.empty:
            continue

        frame = frame.sort_values("timestamp")
        closes = [float(value) for value in frame["close"].tolist()]
        highs = [float(value) for value in frame["high"].tolist()]
        lows = [float(value) for value in frame["low"].tolist()]
        volumes = [float(value) for value in frame["volume"].tolist()]

        if len(closes) < max(parameters.base_lookback_days,
                             parameters.trend_ma_period) + 5:
            continue

        strategy = _build_vcp_scan_strategy(symbol, parameters)
        strategy.close_history[symbol] = deque(
            closes, maxlen=strategy.history_window)
        strategy.high_history[symbol] = deque(
            highs, maxlen=strategy.history_window)
        strategy.low_history[symbol] = deque(
            lows, maxlen=strategy.history_window)
        strategy.volume_history[symbol] = deque(
            volumes, maxlen=strategy.history_window)

        price = closes[-1]
        volume = volumes[-1]
        detection = strategy._detect_breakout(symbol, price, volume)
        if detection is None:
            continue

        entry_price, stop_price, target_price, resistance, base_low, risk = detection
        if risk <= 0:
            continue

        ts_value = pd.to_datetime(frame["timestamp"].iloc[-1], utc=True)
        breakout_ts = ts_value.to_pydatetime()
        analysis_timestamp = (
            breakout_ts
            if analysis_timestamp is None or breakout_ts > analysis_timestamp
            else analysis_timestamp
        )

        reward_to_risk = (target_price - entry_price) / risk
        candidates.append(
            VCPScanCandidate(
                symbol=symbol,
                close_price=price,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                risk_per_share=risk,
                resistance=resistance,
                base_low=base_low,
                breakout_timestamp=breakout_ts,
                reward_to_risk=reward_to_risk,
                volume=volume,
            )
        )

    candidates.sort(key=lambda item: item.reward_to_risk, reverse=True)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    if missing_symbols:
        warnings.append(
            "Excluded missing symbols: " + ", ".join(sorted(missing_symbols))
        )

    return VCPScanSummary(
        timeframe=timeframe,
        parameters=parameters,
        symbols_scanned=len(selected_symbols),
        candidates=candidates,
        warnings=warnings,
        analysis_timestamp=analysis_timestamp,
    )


def _scan_vcp_symbol_history(
    frame: pd.DataFrame,
    symbol: str,
    parameters: VCPParameters,
    *,
    max_detections: int,
) -> tuple[List[VCPPatternDetection], List[str]]:
    strategy = _build_vcp_scan_strategy(symbol, parameters)
    key = symbol.upper()
    close_history = strategy.close_history[key]
    high_history = strategy.high_history[key]
    low_history = strategy.low_history[key]
    volume_history = strategy.volume_history[key]

    timestamps = frame["timestamp"].tolist()
    closes = [float(value) for value in frame["close"].tolist()]
    highs = [float(value) for value in frame["high"].tolist()]
    lows = [float(value) for value in frame["low"].tolist()]
    volumes = [float(value) for value in frame["volume"].tolist()]

    detections: List[VCPPatternDetection] = []
    warnings: List[str] = []
    skip_until_idx = -1
    cooldown = max(parameters.pivot_lookback_days, 1)
    min_required = max(parameters.base_lookback_days,
                       parameters.trend_ma_period) + 5
    if len(closes) < min_required:
        warnings.append(
            f"Only {len(closes)} bars available; detections may be incomplete."
        )

    for idx, (ts, close, high, low, volume) in enumerate(
        zip(timestamps, closes, highs, lows, volumes)
    ):
        close_history.append(close)
        high_history.append(high)
        low_history.append(low)
        volume_history.append(volume)

        detection = strategy._detect_breakout(key, close, volume)
        if detection is None or idx <= skip_until_idx:
            continue

        entry_price, stop_price, target_price, resistance, base_low, risk = detection
        if risk <= 0:
            continue

        base_start_idx = max(0, idx - parameters.base_lookback_days + 1)
        base_end_idx = idx
        base_start_ts = timestamps[base_start_idx]
        base_end_ts = timestamps[base_end_idx]
        reward_to_risk = (target_price - entry_price) / \
            risk if risk else 0.0

        detections.append(
            VCPPatternDetection(
                breakout_timestamp=ts.to_pydatetime(),
                base_start=base_start_ts.to_pydatetime(),
                base_end=base_end_ts.to_pydatetime(),
                entry_price=float(entry_price),
                stop_price=float(stop_price),
                target_price=float(target_price),
                resistance=float(resistance),
                base_low=float(base_low),
                breakout_price=float(close),
                breakout_volume=float(volume),
                risk_per_share=float(risk),
                reward_to_risk=float(reward_to_risk),
            )
        )

        skip_until_idx = idx + cooldown
        if max_detections > 0 and len(detections) >= max_detections:
            break

    return detections, warnings


def scan_vcp_history(
    store_path: Path,
    *,
    symbols: Sequence[str],
    timeframe: str,
    overrides: Dict[str, float | int | None] | None = None,
    bar_size: str = "1d",
    lookback_years: float = 3.0,
    max_detections: int = 8,
) -> VCPPatternHistorySummary:
    if not symbols:
        raise ValueError(
            "At least one symbol must be provided for VCP history scan.")

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        candidate = symbol.strip().upper()
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)

    if not normalized:
        raise ValueError(
            "At least one valid symbol must be provided for history scan.")

    parameters = _resolve_scan_parameters(timeframe, overrides)
    store = ParquetBarStore(store_path)
    summary = VCPPatternHistorySummary()

    lookback_days = max(int(math.ceil(lookback_years * 365.0)),
                        parameters.base_lookback_days + 5)

    for symbol in normalized:
        try:
            frame = store.load(symbol, bar_size)
        except FileNotFoundError:
            summary.missing.append({
                "symbol": symbol,
                "reason": "missing_file",
            })
            continue

        if frame.empty:
            summary.missing.append({
                "symbol": symbol,
                "reason": "empty_data",
            })
            continue

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp")
        if lookback_days > 0:
            last_ts = frame["timestamp"].iloc[-1]
            cutoff = last_ts - pd.Timedelta(days=lookback_days)
            frame = frame.loc[frame["timestamp"] >= cutoff]

        frame = frame.reset_index(drop=True)
        if frame.empty:
            summary.warnings.append(
                f"{symbol}: no historical data within {lookback_years:.1f}-year window."
            )
            continue

        detections, symbol_warnings = _scan_vcp_symbol_history(
            frame,
            symbol,
            parameters,
            max_detections=max_detections,
        )

        if symbol_warnings:
            summary.warnings.extend(
                f"{symbol}: {message}" for message in symbol_warnings
            )

        analysis_start = frame["timestamp"].iloc[0].to_pydatetime()
        analysis_end = frame["timestamp"].iloc[-1].to_pydatetime()

        summary.results[symbol] = VCPPatternSeries(
            symbol=symbol,
            parameters=parameters,
            frame=frame,
            detections=detections,
            warnings=symbol_warnings,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )

    return summary


def run_vcp_experiment(config: VCPExperimentConfig) -> VCPExperimentResult:
    config.validate()
    store = ParquetBarStore(config.store_path)
    base_backtest = config.base_backtest_config()
    result = VCPExperimentResult()

    for params in config.parameters:
        label = params.label()
        for split in config.splits:
            split_config = split.as_backtest_config(base_backtest)
            strategy_cfg = config.strategy_config(params)
            strategy = VCPStrategy(strategy_cfg)
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
                VCPExperimentRow(
                    split=split.name,
                    parameter_label=label,
                    metrics=metrics,
                )
            )
    return result


def _optimize_vcp_parameters_annealing(
    *,
    store_path: Path,
    universe: Sequence[str],
    initial_cash: float,
    training_window: Tuple[date, date],
    paper_window: Tuple[date, date],
    parameter_spec: VCPParameterSpec,
    bar_size: str,
    calendar_name: str,
    commission_model: CommissionModel | None,
    slippage_model: SlippageModel | None,
    search_iterations: int,
    random_seed: int | None,
    initial_temperature: float,
    cooling_rate: float,
) -> VCPOptimizationOutcome:
    rng = random.Random(random_seed)
    options = _parameter_options_from_spec(parameter_spec)

    train_split = ExperimentSplit(
        name="train", start=training_window[0], end=training_window[1]
    )
    paper_split = ExperimentSplit(
        name="paper", start=paper_window[0], end=paper_window[1]
    )

    base_backtest = BacktestConfig(
        symbols=universe,
        start=None,
        end=None,
        bar_size=bar_size,
        initial_cash=initial_cash,
        commission_model=commission_model,
        slippage_model=slippage_model,
        calendar_name=calendar_name,
    )
    store = ParquetBarStore(store_path)
    train_config = train_split.as_backtest_config(base_backtest)
    paper_config = paper_split.as_backtest_config(base_backtest)

    current_params = _random_parameter_from_options(options, rng)
    current_metrics = _evaluate_vcp_training_metrics(
        store,
        train_config,
        current_params,
        universe,
        initial_cash,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )

    evaluations: List[Dict[str, float | int | str]] = []

    def record(label: str, metrics: Dict[str, float], iteration: int, source: str) -> None:
        payload: Dict[str, float | int | str] = {
            "split": "train",
            "params": label,
            "iteration": iteration,
            "source": source,
        }
        payload.update({key: float(value) for key, value in metrics.items()})
        evaluations.append(payload)

    current_label = current_params.label()
    record(current_label, current_metrics, 0, "seed")
    visited: Dict[str, Dict[str, float]] = {current_label: current_metrics}
    best_params = current_params
    best_metrics = current_metrics
    temperature = max(initial_temperature, 1e-3)
    total_iterations = max(search_iterations, 1)

    for iteration in range(1, total_iterations + 1):
        candidate = _mutate_parameter(current_params, options, rng)
        candidate_label = candidate.label()
        source = "cache"
        if candidate_label not in visited:
            metrics = _evaluate_vcp_training_metrics(
                store,
                train_config,
                candidate,
                universe,
                initial_cash,
                commission_model=commission_model,
                slippage_model=slippage_model,
            )
            visited[candidate_label] = metrics
            source = "evaluate"
        else:
            metrics = visited[candidate_label]

        record(candidate_label, metrics, iteration, source)

        candidate_score = _score_metrics(metrics)
        current_score = _score_metrics(current_metrics)

        if candidate_score > current_score:
            accept = True
        else:
            delta = candidate_score[0] - current_score[0]
            accept_prob = math.exp(
                delta / max(temperature, 1e-6)) if delta < 0 else 1.0
            accept = rng.random() < accept_prob

        if accept:
            current_params = candidate
            current_metrics = metrics
            current_label = candidate_label
            current_score = candidate_score

        if candidate_score > _score_metrics(best_metrics):
            best_params = candidate
            best_metrics = metrics

        temperature = max(temperature * cooling_rate, 1e-6)

    parameter_frame = (
        pd.DataFrame.from_records(evaluations)
        if evaluations
        else pd.DataFrame(columns=["split", "params", "iteration", "source"])
    )

    strategy_config = VCPConfig(
        symbols=universe,
        initial_cash=initial_cash,
        base_lookback_days=best_params.base_lookback_days,
        pivot_lookback_days=best_params.pivot_lookback_days,
        min_contractions=best_params.min_contractions,
        max_contraction_pct=best_params.max_contraction_pct,
        contraction_decay=best_params.contraction_decay,
        breakout_buffer_pct=best_params.breakout_buffer_pct,
        volume_squeeze_ratio=best_params.volume_squeeze_ratio,
        breakout_volume_ratio=best_params.breakout_volume_ratio,
        volume_lookback_days=best_params.volume_lookback_days,
        trend_ma_period=best_params.trend_ma_period,
        stop_loss_r_multiple=best_params.stop_loss_r_multiple,
        profit_target_r_multiple=best_params.profit_target_r_multiple,
        trailing_stop_r_multiple=best_params.trailing_stop_r_multiple,
        max_hold_days=best_params.max_hold_days,
        target_position_pct=best_params.target_position_pct,
        lot_size=best_params.lot_size,
        cash_reserve_pct=best_params.cash_reserve_pct,
        max_positions=max(len(universe), 5),
    )
    strategy = VCPStrategy(strategy_config)
    paper_engine = BacktestEngine(config=paper_config, store=store)
    paper_result = paper_engine.run(strategy)
    annotations = strategy.export_annotations()

    paper_metrics = asdict(
        compute_backtest_metrics(
            paper_result,
            initial_cash=paper_config.initial_cash,
            start=paper_config.start,
            end=paper_config.end,
        )
    )

    return VCPOptimizationOutcome(
        best_parameters=best_params,
        training_metrics={key: float(value)
                          for key, value in best_metrics.items()},
        paper_metrics={key: float(value)
                       for key, value in paper_metrics.items()},
        parameter_frame=parameter_frame,
        training_window=(train_split.start, train_split.end),
        paper_window=(paper_split.start, paper_split.end),
        paper_result=paper_result,
        paper_annotations=annotations,
    )


def optimize_vcp_parameters(
    store_path: Path,
    universe: Sequence[str],
    initial_cash: float,
    training_window: Tuple[date, date],
    paper_window: Tuple[date, date],
    *,
    bar_size: str = "1d",
    calendar_name: str = "XNYS",
    parameter_grid: Sequence[VCPParameters] | None = None,
    parameter_spec: VCPParameterSpec | None = None,
    commission_model: CommissionModel | None = None,
    slippage_model: SlippageModel | None = None,
    search_strategy: str = "grid",
    search_iterations: int = 150,
    random_seed: int | None = None,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.95,
) -> VCPOptimizationOutcome:
    if parameter_grid is not None and parameter_spec is not None:
        raise ValueError(
            "Specify either parameter_grid or parameter_spec, not both.")

    strategy_key = search_strategy.lower().strip()
    if strategy_key not in {"grid", "annealing"}:
        raise ValueError(
            f"Unsupported VCP optimization strategy '{search_strategy}'. Use 'grid' or 'annealing'."
        )

    if parameter_grid is not None:
        parameters = list(parameter_grid)
    elif parameter_spec is not None and strategy_key == "grid":
        parameters = generate_vcp_parameter_grid(parameter_spec)
    elif parameter_spec is not None and strategy_key == "annealing":
        return _optimize_vcp_parameters_annealing(
            store_path=store_path,
            universe=universe,
            initial_cash=initial_cash,
            training_window=training_window,
            paper_window=paper_window,
            parameter_spec=parameter_spec,
            bar_size=bar_size,
            calendar_name=calendar_name,
            commission_model=commission_model,
            slippage_model=slippage_model,
            random_seed=random_seed,
            search_iterations=search_iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
        )
    else:
        parameters = generate_vcp_parameter_grid(default_vcp_spec())

    if not parameters:
        raise ValueError(
            "Parameter grid must contain at least one configuration.")

    train_split = ExperimentSplit(
        name="train", start=training_window[0], end=training_window[1])
    paper_split = ExperimentSplit(
        name="paper", start=paper_window[0], end=paper_window[1])

    experiment_config = VCPExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        splits=[train_split],
        parameters=parameters,
        bar_size=bar_size,
        calendar_name=calendar_name,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )
    experiment_result = run_vcp_experiment(experiment_config)
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
    strategy = VCPStrategy(strategy_config)

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
    annotations = strategy.export_annotations()

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

    return VCPOptimizationOutcome(
        best_parameters=best_params,
        training_metrics=training_metrics,
        paper_metrics=paper_metrics,
        parameter_frame=frame,
        training_window=(train_split.start, train_split.end),
        paper_window=(paper_split.start, paper_split.end),
        paper_result=paper_result,
        paper_annotations=annotations,
    )


__all__ = [
    "VCPParameters",
    "VCPParameterSpec",
    "VCPExperimentConfig",
    "VCPExperimentResult",
    "VCPOptimizationOutcome",
    "VCPScanCandidate",
    "VCPScanSummary",
    "VCPPatternDetection",
    "VCPPatternSeries",
    "VCPPatternHistorySummary",
    "default_vcp_spec",
    "default_vcp_scan_universe",
    "refresh_technology_universe",
    "generate_vcp_parameter_grid",
    "optimize_vcp_parameters",
    "technology_universe_symbols",
    "scan_vcp_candidates",
    "scan_vcp_history",
    "run_vcp_experiment",
]
