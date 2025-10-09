"""FastAPI application exposing a candlestick UI for strategy backtests."""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import AppSettings
from ..data.stores.local import ParquetBarStore
from ..data.ingest import ingest_polygon_daily, update_polygon_daily_incremental
from ..data.universe import (
    fetch_snp100_members,
    ingest_universe_frame,
    latest_symbols,
    load_universe,
)
from ..experiments import (
    BreakoutParameterSpec,
    FloatRange,
    MomentumExperimentConfig,
    MomentumParameters,
    MomentumOptimizationSpec,
    MomentumOptimizationSummary,
    VCPParameterSpec,
    VCPParameters,
    VCPPatternDetection,
    VCPPatternSeries,
    DEFAULT_VCP_SCAN_CRITERIA,
    VCP_SCAN_CRITERIA_LABELS,
    default_vcp_spec,
    default_vcp_scan_universe,
    optimize_breakout_parameters,
    optimize_mean_reversion_parameters,
    optimize_momentum_parameters,
    optimize_vcp_parameters,
    generate_momentum_parameter_grid,
    run_momentum_experiment,
    VCPScanCandidate,
    scan_vcp_candidates,
    scan_vcp_history,
    liquid_universe_symbols,
)
from ..experiments.mean_reversion import HoldRange, OptimizationParameterSpec, ParameterRange
from ..strategies import BreakoutPattern

DEFAULT_STORE_PATH = Path(
    os.getenv("ALGO_TRADE_STORE_PATH", "data/raw/polygon/daily"))
DEFAULT_BAR_SIZE = "1d"
DEFAULT_SYMBOLS = ("AAPL", "MSFT")

NASDAQ_TRADER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
NASDAQ_TRADER_CACHE_TTL = timedelta(hours=12)

_VCP_PARAMETER_FIELDS = {field.name for field in fields(VCPParameters)}


class StrategyName(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VCP = "vcp"
    MOMENTUM = "momentum"


class VCPScanTimeframe(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class VCPScanCriterion(str, Enum):
    LIQUIDITY = "liquidity"
    UPTREND_BREAKOUT = "uptrend_breakout"
    HIGHER_LOWS = "higher_lows"
    VOLUME_CONTRACTION = "volume_contraction"


class VCPFetchRequest(BaseModel):
    force_refresh_universe: bool = Field(
        default=True,
        description="When true, refresh the liquid US equity universe before fetching data.",
    )
    lookback_years: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Years of history to seed when no cached data exists for a symbol.",
    )


class RangeRequest(BaseModel):
    """Schema describing an inclusive integer range."""

    minimum: int = Field(..., description="Inclusive lower bound")
    maximum: int = Field(..., description="Inclusive upper bound")
    step: int = Field(1, gt=0, description="Step size between values")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> "RangeRequest":
        if self.maximum < self.minimum:
            raise ValueError(
                "maximum must be greater than or equal to minimum")
        return self

    def to_range(self) -> ParameterRange:
        return ParameterRange(self.minimum, self.maximum, self.step)


class FloatRangeRequest(BaseModel):
    """Schema describing an inclusive floating point range."""

    minimum: float = Field(..., description="Inclusive lower bound")
    maximum: float = Field(..., description="Inclusive upper bound")
    step: float = Field(..., gt=0, description="Step size between values")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> "FloatRangeRequest":
        if self.maximum < self.minimum:
            raise ValueError(
                "maximum must be greater than or equal to minimum")
        return self

    def to_range(self) -> FloatRange:
        return FloatRange(self.minimum, self.maximum, self.step)


class HoldRangeRequest(RangeRequest):
    include_infinite: bool = Field(
        default=False, description="Include an infinite holding period (0 days)."
    )
    only_infinite: bool = Field(
        default=False,
        description="When true, ignore numeric bounds and use only the infinite holding period.",
    )

    model_config = ConfigDict(extra="forbid")

    def to_hold_range(self) -> HoldRange:
        if self.only_infinite:
            return HoldRange(minimum=0, maximum=0, step=1, include_infinite=True)
        return HoldRange(
            minimum=self.minimum,
            maximum=self.maximum,
            step=self.step,
            include_infinite=self.include_infinite,
        )


class MeanReversionParameterSpecRequest(BaseModel):
    entry_threshold: RangeRequest
    exit_threshold: RangeRequest
    max_hold_days: HoldRangeRequest
    target_position_pct: RangeRequest
    stop_loss_pct: RangeRequest | None = None
    include_no_stop_loss: bool = True
    lot_size: int = Field(10, gt=0)

    model_config = ConfigDict(extra="forbid")

    def to_spec(self) -> OptimizationParameterSpec:
        return OptimizationParameterSpec(
            entry_threshold=self.entry_threshold.to_range(),
            exit_threshold=self.exit_threshold.to_range(),
            max_hold_days=self.max_hold_days.to_hold_range(),
            target_position_pct=self.target_position_pct.to_range(),
            stop_loss_pct=self.stop_loss_pct.to_range() if self.stop_loss_pct else None,
            include_no_stop_loss=self.include_no_stop_loss,
            lot_size=self.lot_size,
        )


class BreakoutParameterSpecRequest(BaseModel):
    patterns: List[str] = Field(
        default_factory=lambda: [
            BreakoutPattern.TWENTY_DAY_HIGH.value,
            BreakoutPattern.DONCHIAN_CHANNEL.value,
        ]
    )
    lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=20, maximum=60, step=20)
    )
    breakout_buffer_pct: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.0, maximum=0.01, step=0.005)
    )
    volume_ratio_threshold: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=1.0, maximum=1.5, step=0.5)
    )
    volume_lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=20, maximum=20, step=1)
    )
    max_hold_days: HoldRangeRequest = Field(
        default_factory=lambda: HoldRangeRequest(
            minimum=10,
            maximum=20,
            step=10,
            include_infinite=True,
            only_infinite=False,
        )
    )
    target_position_pct: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=10, maximum=20, step=10)
    )
    stop_loss_pct: FloatRangeRequest | None = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.05, maximum=0.05, step=0.01)
    )
    trailing_stop_pct: FloatRangeRequest | None = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.08, maximum=0.08, step=0.01)
    )
    profit_target_pct: FloatRangeRequest | None = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.15, maximum=0.15, step=0.01)
    )
    include_no_stop_loss: bool = True
    include_no_trailing_stop: bool = True
    include_no_profit_target: bool = True
    lot_size: int = Field(10, gt=0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_patterns(self) -> "BreakoutParameterSpecRequest":
        if not self.patterns:
            raise ValueError("At least one breakout pattern must be provided.")
        normalized: list[str] = []
        seen: set[str] = set()
        for pattern in self.patterns:
            try:
                value = BreakoutPattern(pattern).value
            except ValueError as exc:  # pragma: no cover - request validation
                raise ValueError(
                    f"Unsupported breakout pattern '{pattern}'.") from exc
            if value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        self.patterns = normalized
        return self

    def to_spec(self) -> BreakoutParameterSpec:
        return BreakoutParameterSpec(
            patterns=[BreakoutPattern(value) for value in self.patterns],
            lookback_days=self.lookback_days.to_range(),
            breakout_buffer_pct=self.breakout_buffer_pct.to_range(),
            volume_ratio_threshold=self.volume_ratio_threshold.to_range(),
            volume_lookback_days=self.volume_lookback_days.to_range(),
            max_hold_days=self.max_hold_days.to_hold_range(),
            target_position_pct=self.target_position_pct.to_range(),
            stop_loss_pct=self.stop_loss_pct.to_range() if self.stop_loss_pct else None,
            trailing_stop_pct=self.trailing_stop_pct.to_range(
            ) if self.trailing_stop_pct else None,
            profit_target_pct=self.profit_target_pct.to_range(
            ) if self.profit_target_pct else None,
            include_no_stop_loss=self.include_no_stop_loss,
            include_no_trailing_stop=self.include_no_trailing_stop,
            include_no_profit_target=self.include_no_profit_target,
            lot_size=self.lot_size,
        )


class VCPParameterSpecRequest(BaseModel):
    base_lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=45, maximum=60, step=15)
    )
    pivot_lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=4, maximum=6, step=2)
    )
    min_contractions: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=3, maximum=3, step=1)
    )
    max_contraction_pct: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.12, maximum=0.16, step=0.04)
    )
    contraction_decay: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.6, maximum=0.8, step=0.2)
    )
    breakout_buffer_pct: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.001, maximum=0.003, step=0.002)
    )
    volume_squeeze_ratio: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.65, maximum=0.85, step=0.2)
    )
    breakout_volume_ratio: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=1.8, maximum=2.1, step=0.3)
    )
    volume_lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=18, maximum=24, step=6)
    )
    trend_ma_period: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=45, maximum=60, step=15)
    )
    stop_loss_r_multiple: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.9, maximum=1.1, step=0.2)
    )
    profit_target_r_multiple: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=2.0, maximum=2.5, step=0.5)
    )
    trailing_stop_r_multiple: FloatRangeRequest | None = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=1.5, maximum=1.5, step=0.1)
    )
    include_no_trailing_stop: bool = True
    max_hold_days: HoldRangeRequest = Field(
        default_factory=lambda: HoldRangeRequest(
            minimum=0,
            maximum=0,
            step=1,
            include_infinite=True,
            only_infinite=True,
        )
    )
    target_position_pct: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=15, maximum=15, step=1)
    )
    lot_size: int = Field(1, gt=0)
    cash_reserve_pct: float = Field(0.1, ge=0.0, lt=1.0)

    model_config = ConfigDict(extra="forbid")

    def to_spec(self) -> VCPParameterSpec:
        trailing = self.trailing_stop_r_multiple.to_range(
        ) if self.trailing_stop_r_multiple else None
        return VCPParameterSpec(
            base_lookback_days=self.base_lookback_days.to_range(),
            pivot_lookback_days=self.pivot_lookback_days.to_range(),
            min_contractions=self.min_contractions.to_range(),
            max_contraction_pct=self.max_contraction_pct.to_range(),
            contraction_decay=self.contraction_decay.to_range(),
            breakout_buffer_pct=self.breakout_buffer_pct.to_range(),
            volume_squeeze_ratio=self.volume_squeeze_ratio.to_range(),
            breakout_volume_ratio=self.breakout_volume_ratio.to_range(),
            volume_lookback_days=self.volume_lookback_days.to_range(),
            trend_ma_period=self.trend_ma_period.to_range(),
            stop_loss_r_multiple=self.stop_loss_r_multiple.to_range(),
            profit_target_r_multiple=self.profit_target_r_multiple.to_range(),
            trailing_stop_r_multiple=trailing,
            include_no_trailing_stop=self.include_no_trailing_stop,
            max_hold_days=self.max_hold_days.to_hold_range(),
            target_position_pct=self.target_position_pct.to_range(),
            lot_size=self.lot_size,
            cash_reserve_pct=self.cash_reserve_pct,
        )


class OptimizationRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    initial_cash: float = Field(10_000.0, gt=0)
    limit: int = Field(250, ge=0)
    bar_size: str = Field(DEFAULT_BAR_SIZE, description="Historical bar size")
    store_path: str | None = Field(default=None)
    paper_days: int = Field(360, ge=60)
    training_years: float = Field(2.0, gt=0)
    auto_fetch: bool = Field(
        False, description="Download missing Polygon history on demand"
    )
    strategy: StrategyName = Field(default=StrategyName.MEAN_REVERSION)
    mean_reversion_spec: MeanReversionParameterSpecRequest | None = Field(
        default=None, alias="parameter_spec"
    )
    breakout_spec: BreakoutParameterSpecRequest | None = None
    vcp_spec: VCPParameterSpecRequest | None = None
    vcp_search_strategy: str = Field(
        "grid", description="Optimization search strategy for VCP"
    )
    vcp_search_iterations: int = Field(150, ge=1)
    vcp_initial_temperature: float = Field(1.0, gt=0.0)
    vcp_cooling_rate: float = Field(0.95, gt=0.0, lt=1.0)
    vcp_random_seed: int | None = Field(default=None)
    include_warnings: bool = True

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _normalize(self) -> "OptimizationRequest":
        normalized_symbols: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            cleaned = symbol.strip().upper()
            if not cleaned or cleaned in seen:
                continue
            normalized_symbols.append(cleaned)
            seen.add(cleaned)
        if not normalized_symbols:
            raise ValueError("At least one symbol must be provided.")
        self.symbols = normalized_symbols

        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar_size is currently supported.")
        self.bar_size = normalized_bar_size

        if not isinstance(self.strategy, StrategyName):
            self.strategy = StrategyName(self.strategy)

        return self

    def resolve_mean_reversion_spec(self) -> OptimizationParameterSpec | None:
        if self.mean_reversion_spec is None:
            return None
        return self.mean_reversion_spec.to_spec()

    def resolve_breakout_spec(self) -> BreakoutParameterSpec | None:
        if self.breakout_spec is None:
            return None
        return self.breakout_spec.to_spec()

    def resolve_vcp_spec(self) -> VCPParameterSpec:
        if self.vcp_spec is None:
            return default_vcp_spec()
        return self.vcp_spec.to_spec()

    def resolve_vcp_search_kwargs(self) -> Dict[str, float | int | None | str]:
        strategy = self.vcp_search_strategy.strip().lower()
        if strategy not in {"grid", "annealing"}:
            raise ValueError(
                "Unsupported VCP search strategy. Choose 'grid' or 'annealing'."
            )
        if self.vcp_random_seed is not None and self.vcp_random_seed < 0:
            raise ValueError("Random seed must be non-negative.")
        return {
            "search_strategy": strategy,
            "search_iterations": int(self.vcp_search_iterations),
            "initial_temperature": float(self.vcp_initial_temperature),
            "cooling_rate": float(self.vcp_cooling_rate),
            "random_seed": self.vcp_random_seed,
        }


class VCPScanRequest(BaseModel):
    timeframe: VCPScanTimeframe = Field(default=VCPScanTimeframe.MEDIUM)
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE, description="Historical bar size")
    symbols: List[str] | None = Field(default=None)
    criteria: List[VCPScanCriterion] | None = Field(
        default=None,
        description="Subset of scan criteria to enforce",
    )
    max_candidates: int = Field(50, ge=0)
    include_warnings: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "VCPScanRequest":
        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        if self.symbols:
            cleaned: list[str] = []
            seen: set[str] = set()
            for symbol in self.symbols:
                candidate = symbol.strip().upper()
                if not candidate or candidate in seen:
                    continue
                cleaned.append(candidate)
                seen.add(candidate)
            self.symbols = cleaned or None
        else:
            self.symbols = None

        if self.criteria:
            requested: list[str] = []
            for criterion in self.criteria:
                value = criterion.value if isinstance(
                    criterion, VCPScanCriterion
                ) else str(criterion).strip().lower()
                if value not in VCP_SCAN_CRITERIA_LABELS:
                    raise ValueError(f"Unsupported criterion '{criterion}'.")
                requested.append(value)
            normalized: list[str] = []
            for key in DEFAULT_VCP_SCAN_CRITERIA:
                if key in requested and key not in normalized:
                    normalized.append(key)
            if not normalized:
                normalized = list(DEFAULT_VCP_SCAN_CRITERIA)
            self.criteria = [VCPScanCriterion(key)
                             for key in normalized]
        else:
            self.criteria = [VCPScanCriterion(key)
                             for key in DEFAULT_VCP_SCAN_CRITERIA]
        return self


class VCPPatternRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    timeframe: VCPScanTimeframe = Field(default=VCPScanTimeframe.MEDIUM)
    overrides: Dict[str, float | int | None] | None = Field(default=None)
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE, description="Historical bar size")
    lookback_years: float = Field(3.0, gt=0.0, le=10.0)
    max_detections: int = Field(8, ge=1, le=100)
    include_warnings: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "VCPPatternRequest":
        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            candidate = symbol.strip().upper()
            if not candidate or candidate in seen:
                continue
            cleaned.append(candidate)
            seen.add(candidate)
        if not cleaned:
            raise ValueError(
                "At least one symbol must be provided for VCP testing.")
        self.symbols = cleaned

        if self.overrides:
            normalized: Dict[str, float | int | None] = {}
            for key, value in self.overrides.items():
                key = key.strip()
                if key not in _VCP_PARAMETER_FIELDS:
                    raise ValueError(f"Unsupported override '{key}'.")
                normalized[key] = value
            self.overrides = normalized or None

        return self


class MomentumParameterRequest(BaseModel):
    lookback_days: int = Field(126, gt=0)
    skip_days: int = Field(21, ge=0)
    rebalance_days: int = Field(21, gt=0)
    max_positions: int = Field(5, gt=0)
    lot_size: int = Field(1, gt=0)
    cash_reserve_pct: float = Field(0.05, ge=0.0, lt=1.0)
    min_momentum: float = Field(0.0)
    volatility_window: int = Field(20, ge=0)
    volatility_exponent: float = Field(1.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    def to_parameters(self) -> MomentumParameters:
        return MomentumParameters(
            lookback_days=int(self.lookback_days),
            skip_days=int(self.skip_days),
            rebalance_days=int(self.rebalance_days),
            max_positions=int(self.max_positions),
            lot_size=int(self.lot_size),
            cash_reserve_pct=float(self.cash_reserve_pct),
            min_momentum=float(self.min_momentum),
            volatility_window=int(self.volatility_window),
            volatility_exponent=float(self.volatility_exponent),
        )


class MomentumRunRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    initial_cash: float = Field(100_000.0, gt=0)
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    parameters: List[MomentumParameterRequest] = Field(default_factory=list)
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE)
    auto_fetch: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "MomentumRunRequest":
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            candidate = symbol.strip().upper()
            if not candidate or candidate in seen:
                continue
            cleaned.append(candidate)
            seen.add(candidate)
        if not cleaned:
            raise ValueError("At least one symbol must be provided.")
        self.symbols = cleaned

        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        if self.training_window[0] >= self.training_window[1]:
            raise ValueError("training_window start must be before end")
        if self.paper_window[0] >= self.paper_window[1]:
            raise ValueError("paper_window start must be before end")

        if not self.parameters:
            self.parameters = [MomentumParameterRequest()]

        return self

    def training_period(self) -> Tuple[date, date]:
        return self.training_window

    def paper_period(self) -> Tuple[date, date]:
        return self.paper_window

    def to_config(self, store_path: Path) -> MomentumExperimentConfig:
        return MomentumExperimentConfig(
            store_path=store_path,
            universe=self.symbols,
            initial_cash=float(self.initial_cash),
            training_window=self.training_window,
            paper_window=self.paper_window,
            bar_size=self.bar_size,
        )

    def to_parameters(self) -> List[MomentumParameters]:
        return [item.to_parameters() for item in self.parameters]


class MomentumOptimizationParameterSpecRequest(BaseModel):
    lookback_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=84, maximum=189, step=21)
    )
    skip_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=5, maximum=21, step=8)
    )
    rebalance_days: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=10, maximum=30, step=10)
    )
    max_positions: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=4, maximum=10, step=2)
    )
    lot_size: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=1, maximum=1, step=1)
    )
    cash_reserve_pct: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.05, maximum=0.15, step=0.05)
    )
    min_momentum: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.0, maximum=0.15, step=0.05)
    )
    volatility_window: RangeRequest = Field(
        default_factory=lambda: RangeRequest(minimum=15, maximum=35, step=5)
    )
    volatility_exponent: FloatRangeRequest = Field(
        default_factory=lambda: FloatRangeRequest(
            minimum=0.8, maximum=1.2, step=0.2)
    )
    max_combinations: int = Field(60, ge=1, le=600)

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _range_values(range_request: RangeRequest) -> List[int]:
        values: List[int] = []
        current = int(range_request.minimum)
        while current <= range_request.maximum:
            values.append(current)
            current += range_request.step
        return values

    @staticmethod
    def _float_values(range_request: FloatRangeRequest) -> List[float]:
        values: List[float] = []
        current = float(range_request.minimum)
        step = float(range_request.step)
        epsilon = step / 10 if step else 1e-6
        while current <= range_request.maximum + epsilon:
            values.append(round(current, 6))
            current += step
        return values

    def to_spec(self) -> tuple[MomentumOptimizationSpec, int]:
        spec = MomentumOptimizationSpec(
            lookback_days=self._range_values(self.lookback_days),
            skip_days=self._range_values(self.skip_days),
            rebalance_days=self._range_values(self.rebalance_days),
            max_positions=self._range_values(self.max_positions),
            lot_size=self._range_values(self.lot_size),
            cash_reserve_pct=self._float_values(self.cash_reserve_pct),
            min_momentum=self._float_values(self.min_momentum),
            volatility_window=self._range_values(self.volatility_window),
            volatility_exponent=self._float_values(self.volatility_exponent),
        )
        return spec, int(self.max_combinations)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lookback_days": self._range_values(self.lookback_days),
            "skip_days": self._range_values(self.skip_days),
            "rebalance_days": self._range_values(self.rebalance_days),
            "max_positions": self._range_values(self.max_positions),
            "lot_size": self._range_values(self.lot_size),
            "cash_reserve_pct": self._float_values(self.cash_reserve_pct),
            "min_momentum": self._float_values(self.min_momentum),
            "volatility_window": self._range_values(self.volatility_window),
            "volatility_exponent": self._float_values(self.volatility_exponent),
            "max_combinations": int(self.max_combinations),
        }


class MomentumOptimizeRequest(BaseModel):
    initial_cash: float = Field(100_000.0, gt=0)
    training_window: Tuple[date, date]
    paper_window: Tuple[date, date]
    parameter_spec: MomentumOptimizationParameterSpecRequest = Field(
        default_factory=MomentumOptimizationParameterSpecRequest
    )
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE)
    symbols: List[str] | None = None
    use_snp100: bool = True
    auto_fetch: bool = True
    lookback_years: float = Field(3.0, gt=0.0, le=10.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "MomentumOptimizeRequest":
        normalized_symbols: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols or []:
            candidate = str(symbol).strip().upper()
            if candidate and candidate not in seen:
                normalized_symbols.append(candidate)
                seen.add(candidate)
        self.symbols = normalized_symbols or None

        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        train_start, train_end = self.training_window
        paper_start, paper_end = self.paper_window

        if train_start >= train_end:
            raise ValueError("training_window start must be before end")
        if paper_start >= paper_end:
            raise ValueError("paper_window start must be before end")
        if train_end >= paper_start:
            raise ValueError(
                "training_window must end before the paper_window starts")

        if self.lookback_years <= 0:
            raise ValueError("lookback_years must be positive")

        return self

    def resolved_symbols(self) -> List[str] | None:
        return self.symbols


class VCPScanExportRequest(BaseModel):
    symbols: List[str] = Field(
        default_factory=list,
        description="Symbols to include in the exported watchlist",
    )
    watchlist_name: str | None = Field(
        default=None,
        description="Optional display name used for the download filename.",
    )
    timeframe: str | None = Field(
        default=None,
        description="Optional timeframe label appended to the download filename when present.",
    )
    route: str = Field(
        default="SMART/AMEX",
        description="Routing or exchange specification for the watchlist rows (e.g. SMART/AMEX).",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "VCPScanExportRequest":
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            candidate = symbol.strip().upper()
            if not candidate or candidate in seen:
                continue
            cleaned.append(candidate)
            seen.add(candidate)
        if not cleaned:
            raise ValueError("At least one symbol must be provided.")
        self.symbols = cleaned

        if self.watchlist_name:
            normalized_name = " ".join(self.watchlist_name.strip().split())
            normalized_name = normalized_name.replace(",", " ")
            self.watchlist_name = normalized_name or None

        if self.timeframe:
            normalized_timeframe = " ".join(self.timeframe.strip().split())
            normalized_timeframe = normalized_timeframe.replace(",", " ")
            self.timeframe = normalized_timeframe or None

        route_value = (self.route or "SMART/AMEX").strip().upper()
        route_value = re.sub(r"\s+", "", route_value)
        self.route = route_value or "SMART/AMEX"

        return self


class UniverseFetchRequest(BaseModel):
    symbols: List[str] = Field(
        default_factory=list,
        description="Universe symbols to ensure are cached in Polygon history",
    )
    store_path: str | None = Field(
        default=None,
        description="Parquet store path used to verify cached bars.",
    )
    bar_size: str = Field(
        default=DEFAULT_BAR_SIZE,
        description="Historical bar size to fetch and validate.",
    )
    lookback_years: float = Field(
        default=3.0,
        gt=0.0,
        le=10.0,
        description="Years of history to backfill when downloading from Polygon.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "UniverseFetchRequest":
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            candidate = str(symbol).strip().upper()
            if candidate and candidate not in seen:
                cleaned.append(candidate)
                seen.add(candidate)
        if not cleaned:
            raise ValueError("At least one symbol must be provided.")
        self.symbols = cleaned

        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        self.lookback_years = float(self.lookback_years)
        return self


def create_app(templates_dir: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    root = Path(__file__).resolve().parent
    templates_path = templates_dir or (root / "templates")
    static_path = static_dir or (root / "static")

    app = FastAPI(title="Strategy Optimization Viewer", version="0.2.0")
    templates = Jinja2Templates(directory=str(templates_path))

    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)),
                  name="static")

    @app.get("/", response_class=HTMLResponse)
    async def render_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/mean-reversion", response_class=HTMLResponse)
    async def render_mean_reversion(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("mean_reversion.html", {"request": request})

    @app.get("/breakout", response_class=HTMLResponse)
    async def render_breakout(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("breakout.html", {"request": request})

    @app.get("/vcp", response_class=HTMLResponse)
    async def render_vcp(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("vcp.html", {"request": request})

    @app.get("/vcp-scan", response_class=HTMLResponse)
    async def render_vcp_scan(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("vcp_scan.html", {"request": request})

    @app.get("/momentum", response_class=HTMLResponse)
    async def render_momentum(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("momentum.html", {"request": request})

    @app.get("/api/symbols", response_class=JSONResponse)
    async def list_symbols(
        store_path: str = Query(
            default=str(DEFAULT_STORE_PATH),
            description="Path to the Parquet bar store.",
        ),
        bar_size: str = Query(
            default=DEFAULT_BAR_SIZE, description="Historical bar size to enumerate."
        ),
    ) -> JSONResponse:
        normalized_bar_size = bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise HTTPException(
                status_code=400, detail="Only 1d bar size is supported.")

        store_root = Path(store_path).expanduser().resolve()
        store = ParquetBarStore(store_root)
        symbols = store.list_symbols(normalized_bar_size)
        return JSONResponse({"symbols": symbols})

    @app.get("/api/vcp/universe", response_class=JSONResponse)
    async def get_vcp_universe(
        store_path: str = Query(
            default=str(DEFAULT_STORE_PATH),
            description="Path to the Parquet bar store to inspect.",
        ),
        bar_size: str = Query(
            default=DEFAULT_BAR_SIZE,
            description="Historical bar size to match the cached data.",
        ),
    ) -> JSONResponse:
        normalized_bar_size = bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise HTTPException(
                status_code=400, detail="Only 1d bar size is supported.")

        store_root = Path(store_path).expanduser().resolve()
        try:
            symbols, missing, warnings = default_vcp_scan_universe(
                store_root, bar_size=normalized_bar_size
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        payload: Dict[str, object] = {"symbols": symbols}
        if missing:
            payload["missing"] = missing
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

    @app.get("/api/universe/nasdaq", response_class=JSONResponse)
    async def list_nasdaq_universe(
        store_path: str = Query(
            default=str(DEFAULT_STORE_PATH),
            description="Path to the Parquet bar store used to check cached symbols.",
        ),
        bar_size: str = Query(
            default=DEFAULT_BAR_SIZE,
            description="Historical bar size to match when checking cache availability.",
        ),
    ) -> JSONResponse:
        normalized_bar_size = bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise HTTPException(
                status_code=400, detail="Only 1d bar size is supported.")

        settings = AppSettings()
        nasdaq_symbols, warnings = _load_latest_nasdaq_symbols(settings)
        if not nasdaq_symbols:
            detail: Dict[str, object] = {
                "message": "NASDAQ universe membership not available.",
            }
            if warnings:
                detail["warnings"] = warnings
            raise HTTPException(status_code=404, detail=detail)

        store_root = Path(store_path).expanduser().resolve()
        store = ParquetBarStore(store_root)
        missing = [
            symbol for symbol in nasdaq_symbols if not store.path_for(symbol, normalized_bar_size).exists()
        ]

        payload: Dict[str, object] = {"symbols": nasdaq_symbols}
        if missing:
            payload["missing"] = missing
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

    @app.get("/api/universe/snp100", response_class=JSONResponse)
    async def list_snp100_universe(
        store_path: str = Query(
            default=str(DEFAULT_STORE_PATH),
            description="Path to the Parquet bar store used to check cached symbols.",
        ),
        bar_size: str = Query(
            default=DEFAULT_BAR_SIZE,
            description="Historical bar size to match when checking cache availability.",
        ),
    ) -> JSONResponse:
        normalized_bar_size = bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise HTTPException(
                status_code=400, detail="Only 1d bar size is supported."
            )

        settings = AppSettings()
        snp_symbols, warnings = _load_latest_snp100_symbols(settings)
        if not snp_symbols:
            detail: Dict[str, object] = {
                "message": "S&P 100 universe membership not available.",
            }
            if warnings:
                detail["warnings"] = warnings
            raise HTTPException(status_code=404, detail=detail)

        store_root = Path(store_path).expanduser().resolve()
        store = ParquetBarStore(store_root)
        missing = [
            symbol
            for symbol in snp_symbols
            if not store.path_for(symbol, normalized_bar_size).exists()
        ]

        fetched_symbols: List[str] = []
        fetch_warnings: List[str] = []

        if missing:
            try:
                fetched_symbols, fetch_warnings = _fetch_polygon_history_for_symbols(
                    store,
                    missing,
                    lookback_years=3.0,
                )
            except Exception as exc:  # pragma: no cover - defensive fetch failure
                warnings.append(
                    f"Failed to download missing S&P 100 history from Polygon: {exc}"
                )
            else:
                if fetch_warnings:
                    warnings.extend(fetch_warnings)
                if fetched_symbols:
                    missing = [
                        symbol
                        for symbol in snp_symbols
                        if not store.path_for(symbol, normalized_bar_size).exists()
                    ]

        payload: Dict[str, object] = {"symbols": snp_symbols}
        if fetched_symbols:
            payload["fetched"] = fetched_symbols
        if missing:
            payload["missing"] = missing
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

    @app.post("/api/universe/import/fetch", response_class=JSONResponse)
    async def fetch_import_universe(request_body: UniverseFetchRequest) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH
        ).expanduser().resolve()
        store = ParquetBarStore(store_root)

        def _missing_symbols(symbols: Iterable[str]) -> list[str]:
            return [
                symbol
                for symbol in symbols
                if not store.path_for(symbol, request_body.bar_size).exists()
            ]

        missing_before = _missing_symbols(request_body.symbols)
        fetched: list[str] = []
        warnings: list[str] = []

        if missing_before:
            try:
                fetched, fetch_warnings = _fetch_polygon_history_for_symbols(
                    store,
                    missing_before,
                    lookback_years=request_body.lookback_years,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover - defensive fetch guard
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to download data from Polygon: {exc}",
                ) from exc
            if fetch_warnings:
                warnings.extend(fetch_warnings)

        missing_after = _missing_symbols(request_body.symbols)

        payload: Dict[str, Any] = {
            "requested": request_body.symbols,
            "bar_size": request_body.bar_size,
            "store_path": str(store_root),
            "fetched": fetched,
            "missing": missing_after,
        }
        if missing_before:
            payload["initial_missing"] = missing_before
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

    @app.post("/api/vcp/universe/fetch", response_class=JSONResponse)
    async def fetch_vcp_universe_data(
        request_body: VCPFetchRequest | None = Body(default=None),
    ) -> JSONResponse:
        params = request_body or VCPFetchRequest()

        symbols, universe_warnings = liquid_universe_symbols(
            force_refresh=params.force_refresh_universe
        )
        if not symbols:
            detail_payload: Dict[str, object] = {
                "message": "No liquid US equity symbols available from Polygon."
            }
            if universe_warnings:
                detail_payload["warnings"] = universe_warnings
            raise HTTPException(status_code=503, detail=detail_payload)

        try:
            report = update_polygon_daily_incremental(
                symbols,
                settings=AppSettings(),
                lookback_years=params.lookback_years,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        combined_warnings: list[str] = []
        combined_warnings.extend(universe_warnings)
        combined_warnings.extend(report.warnings)

        payload = {
            "total_symbols": len(symbols),
            "updated_symbols": report.updated,
            "updated_count": sum(report.updated.values()),
            "skipped_symbols": report.skipped,
            "written_paths": [str(path) for path in report.written_paths],
            "lookback_years": params.lookback_years,
            "force_refresh_universe": params.force_refresh_universe,
        }
        if combined_warnings:
            payload["warnings"] = combined_warnings
        return JSONResponse(payload)

    @app.post("/api/optimize", response_class=JSONResponse)
    async def optimize(request_body: OptimizationRequest) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH).expanduser().resolve()
        store = ParquetBarStore(store_root)

        missing_sources: List[Dict[str, str]] = []
        if request_body.auto_fetch:
            try:
                fetched = _maybe_fetch_polygon_data(
                    store,
                    request_body.symbols,
                    bar_size=request_body.bar_size,
                    paper_days=request_body.paper_days,
                    training_years=request_body.training_years,
                )
                if fetched:
                    missing_sources.extend(
                        {"symbol": symbol, "message": "Fetched from Polygon"}
                        for symbol in fetched
                    )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover - defensive fetch guard
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to download data from Polygon: {exc}",
                ) from exc

        frames, missing_frames = _load_symbol_frames(
            store, request_body.symbols, request_body.bar_size
        )
        missing_sources.extend(missing_frames)

        if not frames:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "No historical bars found for the requested symbols.",
                    "missing": missing_sources,
                },
            )

        results: Dict[str, Any] = {}
        symbol_warnings: List[Dict[str, str]] = []

        for symbol, frame in frames.items():
            try:
                training_window, paper_window = _determine_optimization_windows(
                    [frame],
                    paper_days=request_body.paper_days,
                    training_years=request_body.training_years,
                )
            except ValueError as exc:
                symbol_warnings.append({"symbol": symbol, "reason": str(exc)})
                continue

            try:
                if request_body.strategy == StrategyName.MEAN_REVERSION:
                    mean_reversion_spec = request_body.resolve_mean_reversion_spec()
                    optimization = optimize_mean_reversion_parameters(
                        store_path=store_root,
                        universe=[symbol],
                        initial_cash=request_body.initial_cash,
                        training_window=training_window,
                        paper_window=paper_window,
                        bar_size=request_body.bar_size,
                        parameter_spec=mean_reversion_spec,
                    )
                elif request_body.strategy == StrategyName.BREAKOUT:
                    breakout_spec = request_body.resolve_breakout_spec()
                    optimization = optimize_breakout_parameters(
                        store_path=store_root,
                        universe=[symbol],
                        initial_cash=request_body.initial_cash,
                        training_window=training_window,
                        paper_window=paper_window,
                        bar_size=request_body.bar_size,
                        parameter_spec=breakout_spec,
                    )
                else:
                    vcp_spec = request_body.resolve_vcp_spec()
                    optimization = optimize_vcp_parameters(
                        store_path=store_root,
                        universe=[symbol],
                        initial_cash=request_body.initial_cash,
                        training_window=training_window,
                        paper_window=paper_window,
                        bar_size=request_body.bar_size,
                        parameter_spec=vcp_spec,
                        **request_body.resolve_vcp_search_kwargs(),
                    )
            except ValueError as exc:
                symbol_warnings.append({"symbol": symbol, "reason": str(exc)})
                continue

            clipped = _clip_frame_to_window(
                frame, paper_window, request_body.limit)
            candles = _serialize_candles(clipped)
            signals_map = _serialize_signals(optimization.paper_result.trades)
            buy_signals = signals_map.get(symbol, [])
            equity_curve = _serialize_equity_curve(
                optimization.paper_result.equity_curve)
            metrics = _normalize_metrics(optimization.paper_metrics)
            annotations_map = _serialize_annotations(
                getattr(optimization, "paper_annotations", {})
            )
            annotations = annotations_map.get(symbol, [])

            results[symbol] = {
                "candles": candles,
                "buy_signals": buy_signals,
                "equity_curve": equity_curve,
                "metrics": metrics,
                "optimization": _serialize_optimization_summary(optimization),
                "strategy": request_body.strategy.value,
                "annotations": annotations,
            }

        if not results:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unable to optimize any symbols with the provided parameters.",
                    "missing": missing_sources + symbol_warnings,
                },
            )

        payload: Dict[str, Any] = {
            "requested_symbols": request_body.symbols,
            "symbols": list(results.keys()),
            "results": results,
            "strategy": request_body.strategy.value,
        }

        aggregated_warnings = missing_sources + symbol_warnings
        if request_body.include_warnings and aggregated_warnings:
            payload["warnings"] = {
                "message": "Some symbols were skipped because their data files were missing or lacked sufficient coverage.",
                "missing": aggregated_warnings,
            }

        return JSONResponse(payload)

    @app.post("/api/momentum/run", response_class=JSONResponse)
    async def run_momentum(request_body: MomentumRunRequest) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH
        ).expanduser().resolve()
        config = request_body.to_config(store_root)
        try:
            config.validate()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        store = ParquetBarStore(store_root)

        fetched_symbols: List[str] = []
        if request_body.auto_fetch:
            training_start, training_end = request_body.training_period()
            paper_start, paper_end = request_body.paper_period()
            paper_days = (paper_end - paper_start).days
            training_days = (training_end - training_start).days
            if paper_days <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="paper_window must span at least one day.",
                )
            if training_days <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="training_window must span at least one day.",
                )
            try:
                fetched_symbols = _maybe_fetch_polygon_data(
                    store,
                    request_body.symbols,
                    bar_size=config.bar_size,
                    paper_days=paper_days,
                    training_years=max(training_days / 365.25, 1 / 365.25),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover - defensive fetch guard
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to download data from Polygon: {exc}",
                ) from exc

        missing_files = [
            {
                "symbol": symbol,
                "path": str(store.path_for(symbol, config.bar_size)),
            }
            for symbol in config.universe
            if not store.path_for(symbol, config.bar_size).exists()
        ]
        if missing_files:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Missing historical bars for the requested symbols.",
                    "missing": missing_files,
                },
            )

        results: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        for parameter in request_body.to_parameters():
            try:
                outcome = run_momentum_experiment(config, parameter)
            except ValueError as exc:
                warnings.append({
                    "label": parameter.label(),
                    "reason": str(exc),
                })
                continue
            except Exception as exc:  # pragma: no cover - unexpected failure
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            results.append(_serialize_momentum_result(parameter, outcome))

        if not results:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No momentum results were produced for the provided parameters.",
                    "warnings": warnings,
                },
            )

        rankings = sorted(
            [
                {
                    "label": item["label"],
                    "paper_sharpe": item["paper_metrics"].get("sharpe_ratio", 0.0),
                    "paper_total_return": item["paper_metrics"].get("total_return", 0.0),
                    "paper_cagr": item["paper_metrics"].get("cagr", 0.0),
                    "paper_max_drawdown": item["paper_metrics"].get("max_drawdown", 0.0),
                }
                for item in results
            ],
            key=lambda entry: entry["paper_sharpe"],
            reverse=True,
        )

        payload: Dict[str, Any] = {
            "symbols": list(config.universe),
            "store_path": str(store_root),
            "initial_cash": float(config.initial_cash),
            "training_window": {
                "start": config.training_window[0].isoformat(),
                "end": config.training_window[1].isoformat(),
            },
            "paper_window": {
                "start": config.paper_window[0].isoformat(),
                "end": config.paper_window[1].isoformat(),
            },
            "bar_size": config.bar_size,
            "results": results,
            "rankings": rankings,
        }
        if fetched_symbols:
            payload["fetched_symbols"] = fetched_symbols
        if warnings:
            payload["warnings"] = warnings

        return JSONResponse(payload)

    @app.post("/api/momentum/optimize", response_class=JSONResponse)
    async def optimize_momentum_route(
        request_body: MomentumOptimizeRequest,
    ) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH
        ).expanduser().resolve()
        store = ParquetBarStore(store_root)

        warnings: List[str] = []
        fetched_symbols: List[str] = []

        custom_symbols = request_body.resolved_symbols()
        if custom_symbols:
            universe_symbols = list(custom_symbols)
        elif request_body.use_snp100:
            settings = AppSettings()
            universe_symbols, membership_warnings = _load_latest_snp100_symbols(
                settings)
            warnings.extend(membership_warnings)
        else:
            raise HTTPException(
                status_code=400,
                detail="No symbols provided for momentum optimization.",
            )

        if not universe_symbols:
            detail: Dict[str, object] = {
                "message": "No symbols available for momentum optimization.",
            }
            if warnings:
                detail["warnings"] = warnings
            raise HTTPException(status_code=503, detail=detail)

        universe_symbols = sorted(
            {str(symbol).strip().upper()
             for symbol in universe_symbols if str(symbol).strip()}
        )

        def _missing_symbols() -> List[str]:
            return [
                symbol
                for symbol in universe_symbols
                if not store.path_for(symbol, request_body.bar_size).exists()
            ]

        missing_before = _missing_symbols()
        if request_body.auto_fetch and missing_before:
            try:
                fetched_symbols, fetch_warnings = _fetch_polygon_history_for_symbols(
                    store,
                    missing_before,
                    lookback_years=request_body.lookback_years,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover - defensive fetch guard
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to download data from Polygon: {exc}",
                ) from exc
            else:
                warnings.extend(fetch_warnings)

        missing_after = _missing_symbols()
        if missing_after:
            detail: Dict[str, object] = {
                "message": "Missing historical bars for the requested symbols.",
                "missing": [
                    {
                        "symbol": symbol,
                        "path": str(store.path_for(symbol, request_body.bar_size)),
                    }
                    for symbol in missing_after
                ],
            }
            if warnings:
                detail["warnings"] = warnings
            if fetched_symbols:
                detail["fetched"] = fetched_symbols
            raise HTTPException(status_code=404, detail=detail)

        try:
            spec, max_combinations = request_body.parameter_spec.to_spec()
            parameter_grid = generate_momentum_parameter_grid(
                spec,
                max_evaluations=max_combinations,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            summary: MomentumOptimizationSummary = optimize_momentum_parameters(
                store_root,
                universe_symbols,
                float(request_body.initial_cash),
                request_body.training_window,
                request_body.paper_window,
                parameter_grid,
                bar_size=request_body.bar_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        evaluations = summary.evaluations
        results_payload = [
            _serialize_momentum_result(
                evaluation.parameters, evaluation.outcome)
            for evaluation in evaluations
        ]

        rankings = [
            {
                "label": evaluation.parameters.label(),
                "paper_sharpe": float(evaluation.outcome.paper_metrics.sharpe_ratio),
                "paper_total_return": float(evaluation.outcome.paper_metrics.total_return),
                "paper_cagr": float(evaluation.outcome.paper_metrics.cagr),
                "paper_max_drawdown": float(evaluation.outcome.paper_metrics.max_drawdown),
            }
            for evaluation in evaluations
        ]

        best_evaluation = summary.best()
        best_payload = {
            "label": best_evaluation.parameters.label(),
            "parameters": _serialize_parameters(best_evaluation.parameters),
            "training_metrics": _normalize_metrics(
                asdict(best_evaluation.outcome.training_metrics)
            ),
            "paper_metrics": _normalize_metrics(
                asdict(best_evaluation.outcome.paper_metrics)
            ),
        }

        payload: Dict[str, Any] = {
            "mode": "optimize",
            "symbols": universe_symbols,
            "store_path": str(store_root),
            "initial_cash": float(request_body.initial_cash),
            "training_window": {
                "start": request_body.training_window[0].isoformat(),
                "end": request_body.training_window[1].isoformat(),
            },
            "paper_window": {
                "start": request_body.paper_window[0].isoformat(),
                "end": request_body.paper_window[1].isoformat(),
            },
            "bar_size": request_body.bar_size,
            "results": results_payload,
            "rankings": rankings,
            "evaluated_count": len(evaluations),
            "candidate_count": len(parameter_grid),
            "best": best_payload,
            "parameter_spec": request_body.parameter_spec.as_dict(),
        }

        if request_body.auto_fetch:
            payload["auto_fetch"] = True
        if fetched_symbols:
            payload["fetched_symbols"] = fetched_symbols
        if warnings:
            payload["warnings"] = warnings
        if custom_symbols:
            payload["universe_source"] = "custom"
        else:
            payload["universe_source"] = "snp100"

        return JSONResponse(payload)

    @app.post("/api/vcp/scan", response_class=JSONResponse)
    async def scan_vcp(request_body: VCPScanRequest) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH).expanduser().resolve()
        try:
            summary = scan_vcp_candidates(
                store_path=store_root,
                timeframe=request_body.timeframe.value,
                bar_size=request_body.bar_size,
                symbols=request_body.symbols,
                max_candidates=request_body.max_candidates,
                criteria=[
                    criterion.value for criterion in request_body.criteria],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload: Dict[str, Any] = {
            "timeframe": summary.timeframe,
            "analysis_timestamp": summary.analysis_timestamp.isoformat()
            if summary.analysis_timestamp
            else None,
            "symbols_scanned": summary.symbols_scanned,
            "parameters": asdict(summary.parameters),
            "candidates": [
                _serialize_scan_candidate(candidate)
                for candidate in summary.candidates
            ],
            "store_path": str(store_root),
        }

        if request_body.symbols is not None:
            payload["requested_symbols"] = request_body.symbols

        if request_body.include_warnings and summary.warnings:
            payload["warnings"] = summary.warnings

        return JSONResponse(payload)

    @app.post("/api/vcp/scan/export", response_class=PlainTextResponse)
    async def export_vcp_watchlist(request_body: VCPScanExportRequest) -> PlainTextResponse:
        csv_body = _generate_ibkr_watchlist_csv(
            request_body.symbols,
            route=request_body.route,
        )

        filename_seed = request_body.watchlist_name or f"vcp_{request_body.timeframe or 'scan'}"
        filename = _build_watchlist_filename(filename_seed)

        response = PlainTextResponse(
            csv_body,
            media_type="text/csv; charset=utf-8",
        )
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/vcp/testing", response_class=JSONResponse)
    async def scan_vcp_testing(request_body: VCPPatternRequest) -> JSONResponse:
        store_root = Path(
            request_body.store_path or DEFAULT_STORE_PATH).expanduser().resolve()
        try:
            fetched_symbols, fetch_warnings = _ensure_vcp_history_cache(
                store_root,
                request_body.symbols,
                lookback_years=request_body.lookback_years,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive fetch guard
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download Polygon history: {exc}",
            ) from exc
        try:
            summary = scan_vcp_history(
                store_path=store_root,
                symbols=request_body.symbols,
                timeframe=request_body.timeframe.value,
                overrides=request_body.overrides,
                bar_size=request_body.bar_size,
                lookback_years=request_body.lookback_years,
                max_detections=request_body.max_detections,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload: Dict[str, Any] = {
            "requested_symbols": request_body.symbols,
            "symbols": list(summary.results.keys()),
            "timeframe": request_body.timeframe.value,
            "lookback_years": float(request_body.lookback_years),
            "results": {
                symbol: _serialize_vcp_pattern_series(series)
                for symbol, series in summary.results.items()
            },
            "store_path": str(store_root),
        }

        if summary.missing:
            payload["missing"] = summary.missing

        extra_warnings: list[str] = []
        if fetch_warnings:
            extra_warnings.extend(fetch_warnings)

        if request_body.include_warnings and summary.warnings:
            extra_warnings.extend(summary.warnings)

        if extra_warnings:
            payload["warnings"] = extra_warnings

        if fetched_symbols:
            payload["fetched"] = fetched_symbols

        return JSONResponse(payload)

    return app


def _generate_ibkr_watchlist_csv(
    symbols: Iterable[str],
    *,
    route: str,
    description_label: str | None = None,
) -> str:
    lines: list[str] = []
    route_value = route.strip().upper()
    label_suffix = ""
    if description_label:
        trimmed = description_label.strip()
        if trimmed:
            label_suffix = f" · {trimmed}"

    for symbol in symbols:
        entry = symbol.strip().upper()
        if not entry:
            continue
        label = f"{entry}{label_suffix}" if label_suffix else entry
        row_parts = ["SYM", label]
        if route_value:
            row_parts.append(route_value)
        lines.append(",".join(row_parts))

    return "\n".join(lines) + ("\n" if lines else "")


def _build_watchlist_filename(name: str) -> str:
    base = name.strip() or "watchlist"
    if base.lower().endswith(".csv"):
        base = base[:-4]
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", base)
    sanitized = sanitized.strip("_") or "watchlist"
    return f"{sanitized}.csv"


def _download_nasdaq_trader_symbols(url: str = NASDAQ_TRADER_URL) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure path
        warnings.append(f"Failed to download NASDAQ Trader universe: {exc}")
        return [], warnings

    lines = response.text.splitlines()
    if not lines:
        warnings.append("NASDAQ Trader feed returned an empty response.")
        return [], warnings

    header = [item.strip() for item in lines[0].split("|")]
    columns = {name: idx for idx, name in enumerate(header)}

    required_columns = [
        "Symbol",
        "Listing Exchange",
        "Test Issue",
        "NextShares",
    ]
    missing_columns = [
        column for column in required_columns if column not in columns]
    if missing_columns:
        warnings.append(
            "NASDAQ Trader feed missing expected columns: " +
            ", ".join(missing_columns)
        )
        return [], warnings

    symbol_idx = columns.get("Symbol")
    nasdaq_symbol_idx = columns.get("NASDAQ Symbol")
    listing_idx = columns["Listing Exchange"]
    test_idx = columns["Test Issue"]
    nextshares_idx = columns["NextShares"]
    financial_idx = columns.get("Financial Status")

    members: set[str] = set()
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line or line.startswith("File Creation Time"):
            continue
        parts = raw_line.split("|")
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        elif len(parts) > len(header):
            parts = parts[:len(header)]

        listing = parts[listing_idx].strip().upper()
        if listing != "Q":
            continue

        if parts[test_idx].strip().upper() == "Y":
            continue

        if parts[nextshares_idx].strip().upper() == "Y":
            continue

        if financial_idx is not None:
            status = parts[financial_idx].strip().upper()
            if status in {"D", "H", "S"}:  # delinquent, trading halted, suspended
                continue

        candidate = ""
        if nasdaq_symbol_idx is not None:
            candidate = parts[nasdaq_symbol_idx].strip().upper()
        if not candidate and symbol_idx is not None:
            candidate = parts[symbol_idx].strip().upper()

        if not candidate:
            continue

        members.add(candidate)

    if not members:
        warnings.append(
            "NASDAQ Trader feed did not yield any qualifying symbols.")
        return [], warnings

    return sorted(members), warnings


def _load_nasdaq_trader_symbols(settings: AppSettings) -> tuple[List[str], List[str]]:
    cache_path = settings.data_paths.cache / "nasdaq_trader.json"
    cached_symbols, cached_warnings = _load_cached_symbol_list(cache_path)
    warnings = list(cached_warnings)

    now = datetime.now(timezone.utc)
    cache_mtime: datetime | None = None
    try:
        cache_mtime = datetime.fromtimestamp(
            cache_path.stat().st_mtime, timezone.utc
        )
    except OSError:
        cache_mtime = None

    fetch_needed = cache_mtime is None or (
        now - cache_mtime) > NASDAQ_TRADER_CACHE_TTL

    fetched_symbols: List[str] = []
    if fetch_needed:
        fetched_symbols, fetch_warnings = _download_nasdaq_trader_symbols()
        warnings.extend(fetch_warnings)
        if fetched_symbols:
            try:
                settings.data_paths.ensure()
                payload = {
                    "symbols": fetched_symbols,
                    "source": "nasdaq_trader",
                    "fetched_at": now.isoformat(),
                }
                with cache_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
            except Exception as exc:  # pragma: no cover - cache write failure
                warnings.append(f"Failed to write NASDAQ Trader cache: {exc}")
            else:
                warnings.append(
                    "Downloaded NASDAQ Trader universe from nasdaqtrader.com.")
            return fetched_symbols, warnings
    elif cached_symbols:
        warnings.append(
            "NASDAQ Trader cache refreshed within the last 12 hours; using cached symbols."
        )
        return cached_symbols, warnings

    if cached_symbols:
        warnings.append(
            "Using cached NASDAQ Trader universe symbols after download failure.")
        return cached_symbols, warnings

    return [], warnings


def _load_cached_symbol_list(path: Path) -> tuple[List[str], List[str]]:
    if not path.exists():
        return [], []

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive cache parsing
        return [], [f"Failed to parse cached symbol list {path.name}: {exc}"]

    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        return [], [f"Cached symbol list {path.name} did not contain a 'symbols' array."]

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in raw_symbols:
        candidate = str(symbol).strip().upper()
        if candidate and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)

    if not normalized:
        return [], [f"Cached symbol list {path.name} did not contain any usable symbols."]

    return sorted(normalized), []


def _load_latest_nasdaq_symbols(settings: AppSettings) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    universe_path = settings.data_paths.raw / \
        "universe" / "nasdaq" / "membership.parquet"

    if universe_path.exists():
        try:
            snapshots = load_universe(universe_path)
        except Exception as exc:  # pragma: no cover - defensive parsing
            warnings.append(
                f"Failed to load NASDAQ universe membership: {exc}")
        else:
            if snapshots:
                symbols = sorted(latest_symbols(snapshots))
                return symbols, warnings
            warnings.append("NASDAQ universe membership is empty.")
    else:
        warnings.append("NASDAQ universe membership file not found.")

    trader_symbols, trader_warnings = _load_nasdaq_trader_symbols(settings)
    warnings.extend(trader_warnings)
    if trader_symbols:
        return trader_symbols, warnings

    fallback_candidates = [
        settings.data_paths.cache / "nasdaq_universe.json",
        settings.data_paths.cache / "nasdaq_technology_universe.json",
    ]

    for fallback_path in fallback_candidates:
        symbols, fallback_warnings = _load_cached_symbol_list(fallback_path)
        warnings.extend(fallback_warnings)
        if symbols:
            warnings.append(
                f"Using cached NASDAQ symbols from {fallback_path.name}.")
            return symbols, warnings

    return [], warnings


def _load_latest_snp100_symbols(settings: AppSettings) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    universe_path = settings.data_paths.raw / \
        "universe" / "snp100" / "membership.parquet"

    if universe_path.exists():
        try:
            snapshots = load_universe(universe_path)
        except Exception as exc:  # pragma: no cover - defensive parsing
            warnings.append(
                f"Failed to load S&P 100 universe membership: {exc}")
        else:
            if snapshots:
                symbols = sorted(latest_symbols(snapshots))
                return symbols, warnings
            warnings.append("S&P 100 universe membership is empty.")
    else:
        warnings.append("S&P 100 universe membership file not found.")

    try:
        frame = fetch_snp100_members()
    except Exception as exc:  # pragma: no cover - network failure path
        warnings.append(f"Failed to download S&P 100 membership: {exc}")
    else:
        if not frame.empty:
            symbols = sorted(frame["symbol"].astype(
                str).str.upper().str.strip().unique())
            try:
                ingest_universe_frame(
                    frame, settings=settings, universe_name="snp100")
            except Exception as exc:  # pragma: no cover - cache write failure
                warnings.append(
                    f"Failed to persist S&P 100 membership snapshot: {exc}")
            else:
                warnings.append(
                    "Downloaded S&P 100 membership from Wikipedia.")

            cache_path = settings.data_paths.cache / "snp100_universe.json"
            try:
                settings.data_paths.ensure()
                payload = {
                    "symbols": symbols,
                    "source": "snp100_wikipedia",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                with cache_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
            except Exception as exc:  # pragma: no cover - cache write failure
                warnings.append(f"Failed to write S&P 100 cache: {exc}")
            return symbols, warnings
        warnings.append("S&P 100 membership source returned no symbols.")

    fallback_paths = [settings.data_paths.cache / "snp100_universe.json"]

    for fallback_path in fallback_paths:
        symbols, fallback_warnings = _load_cached_symbol_list(fallback_path)
        warnings.extend(fallback_warnings)
        if symbols:
            warnings.append(
                f"Using cached S&P 100 symbols from {fallback_path.name}.")
            return symbols, warnings

    return [], warnings


def _fetch_polygon_history_for_symbols(
    store: ParquetBarStore,
    symbols: Sequence[str],
    *,
    lookback_years: float = 3.0,
) -> tuple[List[str], List[str]]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        candidate = str(symbol).strip().upper()
        if candidate and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)

    if not normalized:
        return [], []

    start, end = _compute_history_fetch_range(lookback_years)
    settings = AppSettings()
    try:
        ingest_polygon_daily(
            normalized,
            start,
            end,
            settings=settings,
            store=store,
            write=True,
        )
    except TypeError as exc:
        params = list(inspect.signature(ingest_polygon_daily).parameters)
        if "store" not in params and "store_obj" in params:
            ingest_polygon_daily(
                normalized,
                start,
                end,
                settings,
                store,
                True,
            )
        else:  # pragma: no cover - unexpected signature mismatch
            raise exc
    return normalized, []


def _ensure_vcp_history_cache(
    store_root: Path,
    symbols: Sequence[str],
    *,
    lookback_years: float,
) -> tuple[List[str], List[str]]:
    normalized: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        candidate = str(symbol).strip().upper()
        if candidate and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)

    if not normalized:
        return [], []

    store = ParquetBarStore(store_root)
    missing = [
        symbol
        for symbol in normalized
        if not store.path_for(symbol, DEFAULT_BAR_SIZE).exists()
    ]

    if not missing:
        return [], []

    start, end = _compute_history_fetch_range(lookback_years)
    settings = AppSettings()
    ingest_polygon_daily(
        missing,
        start,
        end,
        settings=settings,
        store=store,
        write=True,
    )
    return missing, []


def _compute_history_fetch_range(lookback_years: float) -> tuple[date, date]:
    effective_years = max(float(lookback_years or 0.0), 0.25)
    buffer_days = 45
    total_days = max(int(ceil(effective_years * 365.0)) + buffer_days, 120)
    end = date.today()
    start = end - timedelta(days=total_days)
    return start, end


def _maybe_fetch_polygon_data(
    store: ParquetBarStore,
    symbols: Iterable[str],
    *,
    bar_size: str,
    paper_days: int,
    training_years: float,
) -> List[str]:
    if bar_size != DEFAULT_BAR_SIZE:
        raise ValueError("Auto-fetch currently supports only 1d bar size.")

    missing_symbols = [
        symbol for symbol in symbols if not store.path_for(symbol, bar_size).exists()
    ]
    if not missing_symbols:
        return []

    settings = AppSettings()
    start, end = _compute_fetch_range(paper_days, training_years)
    ingest_polygon_daily(
        missing_symbols,
        start,
        end,
        settings=settings,
        store=store,
        write=True,
    )
    return missing_symbols


def _compute_fetch_range(paper_days: int, training_years: float) -> tuple[date, date]:
    end = date.today()
    buffer_days = 30
    total_days = max(int(ceil(training_years * 365)) +
                     paper_days + buffer_days, 1)
    start = end - timedelta(days=total_days)
    return start, end


def _load_symbol_frames(
    store: ParquetBarStore,
    symbols: Iterable[str],
    bar_size: str,
) -> tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    frames: Dict[str, pd.DataFrame] = {}
    missing: List[Dict[str, str]] = []
    for symbol in symbols:
        try:
            frame = _load_frame(store, symbol, bar_size)
        except FileNotFoundError as exc:
            missing.append(
                {"symbol": symbol, "reason": "missing_file", "path": str(exc)})
            continue
        if frame.empty:
            missing.append({"symbol": symbol, "reason": "empty_data"})
            continue
        frames[symbol] = frame.sort_values("timestamp")
    return frames, missing


def _clip_frame_to_window(
    frame: pd.DataFrame,
    window: tuple[date, date],
    limit: int,
) -> pd.DataFrame:
    mask = (frame["timestamp"].dt.date >= window[0]) & (
        frame["timestamp"].dt.date <= window[1]
    )
    clipped = frame.loc[mask]
    if limit and limit > 0:
        clipped = clipped.tail(limit)
    return clipped


def _load_frame(store: ParquetBarStore, symbol: str, bar_size: str) -> pd.DataFrame:
    frame = store.load(symbol, bar_size)
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _infer_start_date(frames: Iterable[pd.DataFrame]) -> date:
    candidates = [frame["timestamp"].min()
                  for frame in frames if not frame.empty]
    if not candidates:
        raise HTTPException(
            status_code=404, detail="No timestamps available for requested symbols.")
    earliest = min(candidates)
    ts = pd.Timestamp(earliest)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.date()


def _infer_end_date(frames: Iterable[pd.DataFrame]) -> date:
    candidates = [frame["timestamp"].max()
                  for frame in frames if not frame.empty]
    if not candidates:
        raise HTTPException(
            status_code=404, detail="No timestamps available for requested symbols.")
    latest = max(candidates)
    ts = pd.Timestamp(latest)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.date()


def _determine_optimization_windows(
    frames: Iterable[pd.DataFrame],
    *,
    paper_days: int = 360,
    training_years: float = 2,
) -> tuple[tuple[date, date], tuple[date, date]]:
    earliest = _infer_start_date(frames)
    latest = _infer_end_date(frames)
    today = date.today()

    paper_end = min(latest, today)
    paper_start_candidate = paper_end - timedelta(days=paper_days)
    paper_start = max(earliest, paper_start_candidate)
    if paper_start >= paper_end:
        raise ValueError(
            "Insufficient data to construct paper-testing window.")

    training_length_days = int(training_years * 365)
    train_end = paper_start - timedelta(days=1)
    if train_end <= earliest:
        raise ValueError(
            "Insufficient history to construct backtesting window.")
    train_start_candidate = paper_start - timedelta(days=training_length_days)
    train_start = max(earliest, train_start_candidate)
    if train_start >= train_end:
        raise ValueError(
            "Insufficient history to construct backtesting window.")

    return (train_start, train_end), (paper_start, paper_end)


def _serialize_momentum_result(
    parameters: MomentumParameters, outcome: Any
) -> Dict[str, Any]:
    training_equity_curve = _serialize_equity_curve(
        outcome.training_result.equity_curve
    )
    paper_equity_curve = _serialize_equity_curve(
        outcome.paper_result.equity_curve
    )

    training_signals = _serialize_signals(outcome.training_result.trades)
    paper_signals = _serialize_signals(outcome.paper_result.trades)

    return {
        "label": parameters.label(),
        "parameters": _serialize_parameters(parameters),
        "training_metrics": _normalize_metrics(asdict(outcome.training_metrics)),
        "paper_metrics": _normalize_metrics(asdict(outcome.paper_metrics)),
        "training_equity_curve": training_equity_curve,
        "paper_equity_curve": paper_equity_curve,
        "training_signals": training_signals,
        "paper_signals": paper_signals,
        "training_final_state": _serialize_portfolio_state(outcome.training_result.final_state),
        "paper_final_state": _serialize_portfolio_state(outcome.paper_result.final_state),
        "training_trades": _serialize_trades(outcome.training_result.trades, phase="training"),
        "paper_trades": _serialize_trades(outcome.paper_result.trades, phase="paper"),
    }


def _serialize_portfolio_state(state: Any) -> Dict[str, Any]:
    positions = [
        {
            "symbol": symbol,
            "quantity": int(position.quantity),
            "avg_price": float(position.avg_price),
        }
        for symbol, position in state.positions.items()
    ]
    positions.sort(key=lambda item: item["symbol"])
    return {
        "cash": float(state.cash),
        "positions": positions,
    }


def _serialize_trades(trades: Iterable[Any], *, phase: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for trade in trades:
        timestamp = _to_iso(getattr(trade, "timestamp", None))
        entries.append(
            {
                "timestamp": timestamp,
                "symbol": getattr(trade, "symbol", ""),
                "quantity": int(getattr(trade, "quantity", 0)),
                "price": float(getattr(trade, "price", 0.0)),
                "cash_after": float(getattr(trade, "cash_after", 0.0)),
                "commission": float(getattr(trade, "commission", 0.0)),
                "slippage": float(getattr(trade, "slippage", 0.0)),
                "phase": phase,
            }
        )
    entries.sort(key=lambda entry: entry["timestamp"] or "")
    return entries


def _serialize_scan_candidate(candidate: VCPScanCandidate) -> Dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "close_price": float(candidate.close_price),
        "market_cap": float(candidate.market_cap) if candidate.market_cap is not None else None,
        "monthly_dollar_volume": float(candidate.monthly_dollar_volume)
        if candidate.monthly_dollar_volume is not None
        else None,
        "rs_percentile": float(candidate.rs_percentile) if candidate.rs_percentile is not None else None,
        "liquidity_pass": bool(candidate.liquidity_pass),
        "market_cap_pass": bool(candidate.market_cap_pass),
        "close_above_sma20": bool(candidate.close_above_sma20),
        "rs_percentile_pass": bool(candidate.rs_percentile_pass),
        "uptrend_breakout_pass": bool(candidate.uptrend_breakout_pass),
        "daily_breakout_distance_pct": float(candidate.daily_breakout_distance_pct)
        if candidate.daily_breakout_distance_pct is not None
        else None,
        "weekly_breakout_distance_pct": float(candidate.weekly_breakout_distance_pct)
        if candidate.weekly_breakout_distance_pct is not None
        else None,
        "higher_lows_pass": bool(candidate.higher_lows_pass),
        "volume_contraction_pass": bool(candidate.volume_contraction_pass),
        "analysis_timestamp": _to_iso(candidate.analysis_timestamp),
    }


def _serialize_vcp_detection(detection: VCPPatternDetection) -> Dict[str, Any]:
    return {
        "breakout_timestamp": _to_iso(detection.breakout_timestamp),
        "base_start": _to_iso(detection.base_start),
        "base_end": _to_iso(detection.base_end),
        "entry_price": float(detection.entry_price),
        "stop_price": float(detection.stop_price),
        "target_price": float(detection.target_price),
        "resistance": float(detection.resistance),
        "base_low": float(detection.base_low),
        "breakout_price": float(detection.breakout_price),
        "breakout_volume": float(detection.breakout_volume),
        "risk_per_share": float(detection.risk_per_share),
        "reward_to_risk": float(detection.reward_to_risk),
    }


def _serialize_vcp_pattern_series(series: VCPPatternSeries) -> Dict[str, Any]:
    detections = [_serialize_vcp_detection(item)
                  for item in series.detections]
    payload: Dict[str, Any] = {
        "candles": _serialize_candles(series.frame),
        "detections": detections,
        "detection_count": len(detections),
        "parameters": asdict(series.parameters),
    }
    if series.analysis_start and series.analysis_end:
        payload["analysis_window"] = {
            "start": _to_iso(series.analysis_start),
            "end": _to_iso(series.analysis_end),
        }
    if series.warnings:
        payload["warnings"] = list(series.warnings)
    return payload


def _serialize_candles(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": _to_iso(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in frame.itertuples(index=False)
    ]


def _serialize_signals(trades: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    signals: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        if getattr(trade, "quantity", 0) > 0:
            symbol = getattr(trade, "symbol", "").upper()
            signals.setdefault(symbol, []).append(
                {
                    "timestamp": _to_iso(getattr(trade, "timestamp")),
                    "price": float(getattr(trade, "price", 0.0)),
                    "quantity": int(getattr(trade, "quantity", 0)),
                }
            )
    for entries in signals.values():
        entries.sort(key=lambda item: item["timestamp"])
    return signals


def _serialize_annotations(raw: Dict[str, Iterable[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    formatted: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, entries in raw.items():
        line_items: List[Dict[str, Any]] = []
        for entry in entries:
            payload: Dict[str, Any] = {
                "timestamp": entry.get("timestamp"),
                "entry": _coerce_float(entry.get("entry")),
                "stop": _coerce_float(entry.get("stop")),
                "target": _coerce_float(entry.get("target")),
                "resistance": _coerce_float(entry.get("resistance")),
                "base_low": _coerce_float(entry.get("base_low")),
                "risk_per_share": _coerce_float(entry.get("risk_per_share")),
            }
            line_items.append(payload)
        line_items.sort(key=lambda item: item.get("timestamp") or "")
        formatted[symbol] = line_items
    return formatted


def _serialize_equity_curve(curve: Iterable[tuple[datetime, float]]) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": _to_iso(timestamp),
            "equity": float(equity),
        }
        for timestamp, equity in curve
    ]


def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return normalized


def _serialize_optimization_summary(outcome: Any) -> Dict[str, Any]:
    rankings_frame = outcome.parameter_frame.sort_values(
        ["cagr", "final_equity"], ascending=[False, False]
    ).head(10)
    rankings = [_normalize_row(record)
                for record in rankings_frame.to_dict(orient="records")]

    return {
        "best_parameters": _serialize_parameters(outcome.best_parameters),
        "training": {
            "start": outcome.training_window[0].isoformat(),
            "end": outcome.training_window[1].isoformat(),
            "metrics": _normalize_metrics(outcome.training_metrics),
        },
        "paper": {
            "start": outcome.paper_window[0].isoformat(),
            "end": outcome.paper_window[1].isoformat(),
            "metrics": _normalize_metrics(outcome.paper_metrics),
        },
        "rankings": rankings,
    }


def _serialize_parameters(params: Any) -> Dict[str, Any]:
    if is_dataclass(params):
        data = asdict(params)
    elif isinstance(params, dict):
        data = dict(params)
    else:
        data = {
            key: getattr(params, key)
            for key in dir(params)
            if not key.startswith("_") and not callable(getattr(params, key))
        }

    serialized: Dict[str, Any] = {}
    for key, value in data.items():
        serialized[key] = _normalize_parameter_value(value)
    return serialized


def _normalize_parameter_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_parameter_value(item) for item in value]
    return value


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        elif isinstance(value, Enum):
            normalized[key] = value.value
        else:
            normalized[key] = value
    return normalized


def _to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
