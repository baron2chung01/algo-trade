"""FastAPI application exposing a candlestick UI for strategy backtests."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from uuid import uuid4

import pandas as pd
import requests
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import AppSettings
from ..data.stores.local import ParquetBarStore
from ..data.contracts import ContractSpec
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

DEFAULT_PAPER_TRADE_PARAMETERS: Dict[str, Any] = {
    "lookback_days": 126,
    "skip_days": 0,
    "rebalance_days": 5,
    "max_positions": 10,
    "lot_size": 1,
    "cash_reserve_pct": 0.05,
    "min_momentum": 0.04,
    "volatility_window": 20,
    "volatility_exponent": 1.25,
}
DEFAULT_PAPER_TRADE_TRAINING_YEARS = 2.0
DEFAULT_PAPER_TRADE_PAPER_DAYS = 365
MOMENTUM_LIVE_STATE_FILE = "momentum_live_state.json"
DEFAULT_LIVE_INTERVAL_SECONDS = 1_800

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


class MomentumPaperTradeRequest(BaseModel):
    symbols: List[str] | None = None
    initial_cash: float = Field(10_000.0, gt=0)
    training_window: Tuple[date, date] | None = None
    paper_window: Tuple[date, date] | None = None
    parameters: List[MomentumParameterRequest] | None = None
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE)
    auto_fetch: bool = False
    execute_orders: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "MomentumPaperTradeRequest":
        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        if self.parameters is None or not self.parameters:
            self.parameters = [MomentumParameterRequest(
                **DEFAULT_PAPER_TRADE_PARAMETERS
            )]
        elif len(self.parameters) != 1:
            raise ValueError(
                "Momentum paper trading accepts exactly one parameter set.")

        if self.training_window is None or self.paper_window is None:
            training_window, paper_window = _default_paper_trade_windows()
            self.training_window = training_window
            self.paper_window = paper_window

        if self.training_window[0] >= self.training_window[1]:
            raise ValueError("training_window start must be before end")
        if self.paper_window[0] >= self.paper_window[1]:
            raise ValueError("paper_window start must be before end")

        if self.symbols is not None:
            cleaned = _normalize_symbol_list(self.symbols)
            if not cleaned:
                raise ValueError(
                    "When provided, symbols must contain at least one ticker.")
            self.symbols = cleaned

        return self


class MomentumLiveTradeRequest(BaseModel):
    symbols: List[str] | None = None
    initial_cash: float | None = Field(default=None)
    interval_seconds: int = Field(
        DEFAULT_LIVE_INTERVAL_SECONDS, ge=60, le=86_400)
    auto_fetch: bool = False
    execute_orders: bool = True
    parameters: List[MomentumParameterRequest] | None = None
    store_path: str | None = None
    bar_size: str = Field(DEFAULT_BAR_SIZE)
    paper_days: int = Field(DEFAULT_PAPER_TRADE_PAPER_DAYS, ge=60, le=730)
    training_years: float = Field(
        DEFAULT_PAPER_TRADE_TRAINING_YEARS, ge=0.5, le=5.0)
    max_iterations: int | None = Field(default=None, ge=1, le=1_000)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "MomentumLiveTradeRequest":
        normalized_bar_size = self.bar_size.strip().lower()
        if normalized_bar_size != DEFAULT_BAR_SIZE:
            raise ValueError("Only 1d bar size is currently supported.")
        self.bar_size = normalized_bar_size

        if self.parameters is None or not self.parameters:
            self.parameters = [MomentumParameterRequest(
                **DEFAULT_PAPER_TRADE_PARAMETERS
            )]
        elif len(self.parameters) != 1:
            raise ValueError(
                "Momentum live trading accepts exactly one parameter set."
            )

        if self.symbols is not None:
            cleaned = _normalize_symbol_list(self.symbols)
            if not cleaned:
                raise ValueError(
                    "When provided, symbols must contain at least one ticker."
                )
            self.symbols = cleaned

        if self.initial_cash is not None:
            initial_cash_value = float(self.initial_cash)
            if initial_cash_value <= 0:
                raise ValueError(
                    "initial_cash must be positive when provided.")
            self.initial_cash = initial_cash_value

        self.interval_seconds = int(self.interval_seconds)
        if self.interval_seconds < 60:
            raise ValueError("interval_seconds must be at least 60 seconds.")

        self.paper_days = int(self.paper_days)
        self.training_years = float(self.training_years)
        return self


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

    app.state.momentum_live_trader = MomentumLiveTrader()

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

    @app.get("/momentum-paper", response_class=HTMLResponse)
    async def render_momentum_paper(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("momentum_paper.html", {"request": request})

    @app.get("/momentum-live", response_class=HTMLResponse)
    async def render_momentum_live(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("momentum_live.html", {"request": request})

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

    @app.post("/api/momentum/paper-trade", response_class=JSONResponse)
    async def run_momentum_paper_trade_route(
        request_body: MomentumPaperTradeRequest,
    ) -> JSONResponse:
        payload = _run_momentum_paper_trade(request_body)
        return JSONResponse(payload)

    @app.post("/api/momentum/live/start", response_class=JSONResponse)
    async def start_momentum_live_route(
        request_body: MomentumLiveTradeRequest,
    ) -> JSONResponse:
        trader: MomentumLiveTrader = app.state.momentum_live_trader
        try:
            payload = trader.start(request_body)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.post("/api/momentum/live/stop", response_class=JSONResponse)
    async def stop_momentum_live_route() -> JSONResponse:
        trader: MomentumLiveTrader = app.state.momentum_live_trader
        payload = trader.stop()
        return JSONResponse(payload)

    @app.get("/api/momentum/live/status", response_class=JSONResponse)
    async def get_momentum_live_status() -> JSONResponse:
        trader: MomentumLiveTrader = app.state.momentum_live_trader
        return JSONResponse(trader.status())

    @app.get("/api/momentum/live/history", response_class=JSONResponse)
    async def get_momentum_live_history() -> JSONResponse:
        trader: MomentumLiveTrader = app.state.momentum_live_trader
        return JSONResponse(trader.history())

    @app.post("/api/momentum/live/history/reset", response_class=JSONResponse)
    async def reset_momentum_live_history() -> JSONResponse:
        trader: MomentumLiveTrader = app.state.momentum_live_trader
        try:
            payload = trader.reset_history()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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


def _parse_trade_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.date()


def _filter_trades_for_trade_date(
    trades: Sequence[Dict[str, Any]],
    target_date: date | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for entry in trades:
        if not isinstance(entry, dict):
            continue
        entry_date = _parse_trade_date(entry.get("timestamp"))
        if target_date is None or entry_date == target_date:
            filtered.append(dict(entry))
        else:
            skipped.append(dict(entry))
    filtered.sort(key=lambda item: item.get("timestamp") or "")
    skipped.sort(key=lambda item: item.get("timestamp") or "")
    return filtered, skipped


def _run_momentum_paper_trade(
    request_body: MomentumPaperTradeRequest,
    *,
    settings: AppSettings | None = None,
    live_only: bool = False,
) -> Dict[str, Any]:
    runtime_settings = settings or AppSettings()
    warnings: List[str] = []
    actions: List[Dict[str, Any]] = []
    fetched_symbols: List[str] = []

    store_root = Path(
        request_body.store_path or DEFAULT_STORE_PATH
    ).expanduser().resolve()
    store = ParquetBarStore(store_root)

    if request_body.symbols:
        universe_symbols = sorted(request_body.symbols)
        actions.append(
            {
                "type": "universe_loaded",
                "source": "request",
                "symbol_count": len(universe_symbols),
                "symbols": universe_symbols,
            }
        )
    else:
        universe_symbols, universe_warnings, universe_source = _load_snp100_paper_trade_symbols(
            runtime_settings
        )
        warnings.extend(universe_warnings)
        if not universe_symbols:
            detail: Dict[str, object] = {
                "message": "S&P 100 universe is unavailable for paper trading.",
            }
            if warnings:
                detail["warnings"] = warnings
            raise HTTPException(status_code=503, detail=detail)
        actions.append(
            {
                "type": "universe_loaded",
                "source": universe_source,
                "symbol_count": len(universe_symbols),
                "symbols": universe_symbols,
            }
        )

    parameter_request = request_body.parameters[0]
    parameters = parameter_request.to_parameters()
    actions.append(
        {
            "type": "parameters_prepared",
            "parameters": _serialize_parameters(parameters),
        }
    )

    config = MomentumExperimentConfig(
        store_path=store_root,
        universe=universe_symbols,
        initial_cash=float(request_body.initial_cash),
        training_window=request_body.training_window,
        paper_window=request_body.paper_window,
        bar_size=request_body.bar_size,
    )

    try:
        config.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request_body.auto_fetch:
        training_start, training_end = request_body.training_window
        paper_start, paper_end = request_body.paper_window
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
                universe_symbols,
                bar_size=request_body.bar_size,
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
        else:
            actions.append(
                {
                    "type": "history_fetched",
                    "symbols": fetched_symbols,
                    "bar_size": request_body.bar_size,
                }
            )

    missing_files = [
        {
            "symbol": symbol,
            "path": str(store.path_for(symbol, request_body.bar_size)),
        }
        for symbol in universe_symbols
        if not store.path_for(symbol, request_body.bar_size).exists()
    ]
    if missing_files:
        detail = {
            "message": "Missing historical bars for the requested symbols.",
            "missing": missing_files,
        }
        if warnings:
            detail["warnings"] = warnings
        if fetched_symbols:
            detail["fetched"] = fetched_symbols
        raise HTTPException(status_code=404, detail=detail)

    actions.append(
        {
            "type": "experiment_run",
            "training_window": {
                "start": config.training_window[0].isoformat(),
                "end": config.training_window[1].isoformat(),
            },
            "paper_window": {
                "start": config.paper_window[0].isoformat(),
                "end": config.paper_window[1].isoformat(),
            },
        }
    )

    try:
        outcome = run_momentum_experiment(config, parameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary = _serialize_momentum_result(parameters, outcome)
    original_trades = summary.get("paper_trades", [])
    if not isinstance(original_trades, list):
        original_trades = []

    paper_window = request_body.paper_window if request_body.paper_window else None
    target_trade_date = paper_window[1] if (
        live_only and paper_window) else None
    paper_trades, skipped_trades = _filter_trades_for_trade_date(
        original_trades,
        target_trade_date,
    )
    summary["paper_trades"] = paper_trades

    if live_only:
        target_label = (
            target_trade_date.isoformat()
            if isinstance(target_trade_date, date)
            else "current trading day"
        )
        if skipped_trades:
            actions.append(
                {
                    "type": "live_trade_filter",
                    "status": "historical_trades_filtered",
                    "filtered_count": len(skipped_trades),
                    "target_date": target_label,
                }
            )
            warnings.append(
                f"Filtered {len(skipped_trades)} historical momentum trades outside {target_label}."
            )
        if not paper_trades:
            actions.append(
                {"type": "trade_signal", "status": "no_trades_today"})
            warnings.append(
                "No qualifying momentum trades were generated for the current trading day."
            )

    if paper_trades:
        for entry in paper_trades:
            quantity = int(entry.get("quantity", 0))
            if quantity == 0:
                continue
            direction = "buy" if quantity > 0 else "sell"
            actions.append(
                {
                    "type": "trade_signal",
                    "symbol": entry.get("symbol"),
                    "direction": direction,
                    "quantity": quantity,
                    "price": entry.get("price"),
                    "timestamp": entry.get("timestamp"),
                }
            )
    else:
        if not live_only:
            actions.append({"type": "trade_signal", "status": "no_trades"})

    quote_map: Dict[str, Dict[str, Any]] = {}
    if paper_trades:
        trade_symbols = [entry.get("symbol") for entry in paper_trades]
        quotes, quote_actions, quote_warnings = _fetch_polygon_last_trade_quotes(
            trade_symbols, settings=runtime_settings
        )
        quote_map = quotes
        if quote_actions:
            actions.extend(quote_actions)
        if quote_warnings:
            warnings.extend(quote_warnings)

        for entry in paper_trades:
            symbol = str(entry.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            quote = quote_map.get(symbol)
            if not quote:
                continue
            quote_price = quote.get("price")
            if quote_price is None:
                continue
            original_price = entry.get("price")
            if original_price is not None and "paper_price" not in entry:
                try:
                    entry["paper_price"] = float(original_price)
                except (TypeError, ValueError):
                    pass
            entry["price"] = float(quote_price)
            entry["quote_price"] = float(quote_price)
            if quote.get("timestamp"):
                entry["quote_timestamp"] = quote["timestamp"]
            if quote.get("size") is not None:
                try:
                    entry["quote_size"] = int(quote["size"])
                except (TypeError, ValueError):
                    pass
            if quote.get("exchange") is not None:
                entry["quote_exchange"] = quote["exchange"]

            if original_price is not None:
                try:
                    original_float = float(original_price)
                except (TypeError, ValueError):
                    original_float = None
                if (
                    original_float is not None
                    and abs(float(quote_price) - original_float) > 0.01
                ):
                    warnings.append(
                        f"Polygon quote for {symbol} ({float(quote_price):.2f}) differs from simulated price ({original_float:.2f})."
                    )

    if request_body.execute_orders and paper_trades and not live_only:
        execution_actions, execution_warnings, execution_reports = _execute_ibkr_paper_orders(
            paper_trades, settings=runtime_settings
        )
        actions.extend(execution_actions)
        warnings.extend(execution_warnings)
        if execution_reports:
            _apply_ibkr_execution_reports(
                paper_trades,
                execution_reports,
                actions=actions,
                warnings=warnings,
            )
    elif request_body.execute_orders and live_only and paper_trades:
        actions.append(
            {
                "type": "ibkr_execution_deferred",
                "reason": "live_trade_post_processing",
            }
        )

    payload: Dict[str, Any] = {
        "mode": "paper_trade",
        "symbols": universe_symbols,
        "store_path": str(store_root),
        "initial_cash": float(request_body.initial_cash),
        "training_window": {
            "start": config.training_window[0].isoformat(),
            "end": config.training_window[1].isoformat(),
        },
        "paper_window": {
            "start": config.paper_window[0].isoformat(),
            "end": config.paper_window[1].isoformat(),
        },
        "parameters": summary.get("parameters"),
        "evaluation": summary,
        "actions": actions,
    }

    if fetched_symbols:
        payload["fetched_symbols"] = fetched_symbols
    if warnings:
        payload["warnings"] = warnings
    payload["paper_trades"] = paper_trades
    payload["execution_requested"] = bool(request_body.execute_orders)
    return payload


def _run_momentum_realtime_trade(
    request_body: MomentumPaperTradeRequest,
    *,
    settings: AppSettings,
    positions_snapshot: Dict[str, Dict[str, Any]],
    liquidity_snapshot: Dict[str, Any],
    extra_actions: Sequence[Dict[str, Any]] | None = None,
    extra_warnings: Sequence[str] | None = None,
    previous_quotes: Mapping[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    if extra_actions:
        actions.extend(dict(action)
                       for action in extra_actions if isinstance(action, dict))

    warnings: List[str] = []
    if extra_warnings:
        warnings.extend(str(message) for message in extra_warnings if message)

    initial_cash = float(request_body.initial_cash or 0.0)
    if initial_cash <= 0:
        raise RuntimeError(
            "Real-time momentum run requires a positive initial cash balance.")

    store_root = Path(
        request_body.store_path or DEFAULT_STORE_PATH).expanduser().resolve()
    store = ParquetBarStore(store_root)

    if request_body.parameters and len(request_body.parameters) >= 1:
        parameters = request_body.parameters[0].to_parameters()
    else:
        parameters = MomentumParameterRequest(
            **DEFAULT_PAPER_TRADE_PARAMETERS).to_parameters()

    if request_body.symbols:
        universe_symbols = sorted({symbol.upper()
                                  for symbol in request_body.symbols})
        actions.append(
            {
                "type": "universe_loaded",
                "source": "request",
                "symbol_count": len(universe_symbols),
                "symbols": universe_symbols,
            }
        )
    else:
        universe_symbols, universe_warnings, universe_source = _load_snp100_paper_trade_symbols(
            settings)
        warnings.extend(universe_warnings)
        if not universe_symbols:
            raise RuntimeError(
                "Unable to load S&P 100 universe for real-time momentum scan.")
        actions.append(
            {
                "type": "universe_loaded",
                "source": universe_source,
                "symbol_count": len(universe_symbols),
                "symbols": universe_symbols,
            }
        )

    holdings_symbols = sorted({symbol.upper()
                              for symbol in positions_snapshot.keys()})
    scoring_symbols = sorted({*universe_symbols, *holdings_symbols})
    quote_symbols = scoring_symbols

    polygon_quotes, quote_actions, quote_warnings = _fetch_polygon_live_quotes(
        quote_symbols,
        settings=settings,
    )
    actions.extend(quote_actions)
    warnings.extend(quote_warnings)

    ibkr_quotes, ibkr_quote_actions, ibkr_quote_warnings = _fetch_ibkr_live_quotes(
        quote_symbols,
        settings=settings,
    )
    actions.extend(ibkr_quote_actions)
    warnings.extend(ibkr_quote_warnings)

    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol, payload in polygon_quotes.items():
        if isinstance(payload, dict):
            quotes[symbol] = dict(payload)

    if ibkr_quotes:
        for symbol, payload in ibkr_quotes.items():
            if not isinstance(payload, dict):
                continue
            prior_entry = quotes.get(symbol)
            prior_source = prior_entry.get("source") if isinstance(
                prior_entry, dict) else None
            merged_entry: Dict[str, Any] = {}
            if isinstance(prior_entry, dict):
                merged_entry.update(prior_entry)
            merged_entry.update(payload)
            quotes[symbol] = merged_entry
            actions.append(
                {
                    "type": "quote_source_merge",
                    "symbol": symbol,
                    "preferred_source": payload.get("source") or "ibkr_snapshot",
                    "prior_source": prior_source,
                }
            )

    cached_hits: list[str] = []
    if previous_quotes:
        for symbol in quote_symbols:
            normalized = str(symbol).strip().upper()
            if not normalized:
                continue
            current_entry = quotes.get(normalized)
            current_price: float | None = None
            if isinstance(current_entry, dict):
                raw_price = current_entry.get("price")
                try:
                    if raw_price is not None:
                        current_price = float(raw_price)
                except (TypeError, ValueError):
                    current_price = None
            if current_price is not None and current_price > 0:
                continue

            cached_entry = previous_quotes.get(normalized)
            if not isinstance(cached_entry, Mapping):
                continue
            cached_price_raw = cached_entry.get("price")
            try:
                cached_price = float(cached_price_raw)
            except (TypeError, ValueError, AttributeError):
                cached_price = None
            if cached_price is None or cached_price <= 0:
                continue

            timestamp_value = cached_entry.get("timestamp")
            if not isinstance(timestamp_value, str) or not timestamp_value.strip():
                timestamp_value = datetime.now(timezone.utc).isoformat()

            fallback_payload: Dict[str, Any] = {
                "price": float(cached_price),
                "timestamp": timestamp_value,
                "source": cached_entry.get("source") or "cached_quote",
            }
            for key in ("bid", "ask", "last", "midpoint", "close"):
                if key in cached_entry and cached_entry[key] is not None:
                    fallback_payload[key] = cached_entry[key]

            quotes[normalized] = fallback_payload
            cached_hits.append(normalized)
            actions.append(
                {
                    "type": "quote_cache_fallback",
                    "symbol": normalized,
                    "status": "applied",
                    "source": fallback_payload.get("source"),
                    "timestamp": fallback_payload.get("timestamp"),
                }
            )
            warnings.append(
                f"Using cached quote for {normalized}; live data unavailable."
            )

    scored, scores_map, score_warnings = _compute_realtime_momentum_scores(
        scoring_symbols,
        store=store,
        bar_size=request_body.bar_size,
        quotes=quotes,
        parameters=parameters,
    )
    warnings.extend(score_warnings)

    selected: List[str] = []
    if scored:
        for symbol, _ in scored:
            if symbol in universe_symbols:
                selected.append(symbol)
            if len(selected) >= parameters.max_positions:
                break

    selection_timestamp = datetime.now(timezone.utc).isoformat()
    actions.append(
        {
            "type": "momentum_selection_snapshot",
            "timestamp": selection_timestamp,
            "selected_symbols": selected[:20],
            "candidate_count": len(universe_symbols),
        }
    )

    if not quotes:
        warnings.append(
            "No real-time Polygon quotes available; skipping trade generation.")
        payload = {
            "mode": "live",
            "symbols": universe_symbols,
            "initial_cash": initial_cash,
            "paper_trades": [],
            "actions": actions,
            "warnings": warnings,
            "portfolio": {
                "positions": positions_snapshot,
                "liquidity": liquidity_snapshot,
                "cash": initial_cash,
            },
            "ibkr_positions": positions_snapshot,
            "liquidity": liquidity_snapshot,
            "execution_requested": bool(request_body.execute_orders),
            "trade_date": date.today().isoformat(),
        }
        return payload

    cash_available = float(initial_cash)
    live_positions: Dict[str, Dict[str, Any]] = {}
    holdings_value = 0.0
    position_count = 0

    for symbol, info in positions_snapshot.items():
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue
        quantity_raw = info.get("quantity", 0)
        try:
            quantity = int(quantity_raw)
        except (TypeError, ValueError):
            continue
        if quantity == 0:
            continue

        quote = quotes.get(normalized)
        price = quote.get("price") if isinstance(quote, dict) else None
        if price is None:
            fallback_price = info.get("market_price") or info.get("avg_cost")
            if fallback_price is None:
                warnings.append(
                    f"Missing price for held position {normalized}; skipping valuation."
                )
                continue
            price = float(fallback_price)

        position_value = float(quantity) * float(price)
        live_positions[normalized] = {
            "quantity": quantity,
            "market_price": float(price),
            "market_value": position_value,
            "momentum_score": scores_map.get(normalized),
        }
        holdings_value += position_value
        position_count += 1

    total_assets = cash_available + holdings_value
    live_liquidity: Dict[str, Any] = {
        "cash_available": cash_available,
        "holdings_value": holdings_value,
        "total_assets": total_assets,
        "positions_value": {symbol: data.get("market_value") for symbol, data in live_positions.items()},
        "position_count": position_count,
    }
    cash_breakdown = liquidity_snapshot.get(
        "cash_breakdown") if isinstance(liquidity_snapshot, dict) else None
    if isinstance(cash_breakdown, dict):
        live_liquidity["cash_breakdown"] = cash_breakdown

    investable_equity = total_assets * \
        (1.0 - float(parameters.cash_reserve_pct))
    selected_set = set(selected)
    paper_trades: List[Dict[str, Any]] = []

    timestamp_now = datetime.now(timezone.utc).isoformat()

    for symbol, details in live_positions.items():
        if symbol in selected_set:
            continue
        quote = quotes.get(symbol)
        price = quote.get("price") if isinstance(quote, dict) else None
        if price is None:
            warnings.append(
                f"Cannot compute exit for {symbol}; missing real-time price.")
            continue
        quantity = int(details.get("quantity", 0))
        if quantity == 0:
            continue
        trade = {
            "symbol": symbol,
            "quantity": -quantity,
            "price": float(price),
            "quote_price": float(price),
            "quote_timestamp": quote.get("timestamp") if isinstance(quote, dict) else timestamp_now,
            "quote_source": quote.get("source") if isinstance(quote, dict) else "unknown",
            "timestamp": timestamp_now,
            "phase": "live",
            "reason": "momentum_exit",
            "momentum_score": scores_map.get(symbol),
        }
        paper_trades.append(trade)

    selected_count = len(selected)
    lot_size = max(int(parameters.lot_size), 1)
    per_position_value = investable_equity / \
        selected_count if selected_count else 0.0

    for symbol in selected:
        quote = quotes.get(symbol)
        price = quote.get("price") if isinstance(quote, dict) else None
        if price is None or price <= 0:
            warnings.append(
                f"Skipping {symbol} buy evaluation; invalid real-time price.")
            continue

        current_quantity = int(live_positions.get(
            symbol, {}).get("quantity", 0))
        target_quantity = int(per_position_value /
                              float(price)) if per_position_value > 0 else 0
        target_quantity = (target_quantity // lot_size) * lot_size
        if target_quantity < 0:
            target_quantity = 0
        delta = target_quantity - current_quantity
        if delta == 0:
            continue

        trade = {
            "symbol": symbol,
            "quantity": delta,
            "price": float(price),
            "quote_price": float(price),
            "quote_timestamp": quote.get("timestamp") if isinstance(quote, dict) else timestamp_now,
            "quote_source": quote.get("source") if isinstance(quote, dict) else "unknown",
            "timestamp": timestamp_now,
            "phase": "live",
            "reason": "momentum_rebalance",
            "momentum_score": scores_map.get(symbol),
        }
        paper_trades.append(trade)

    if paper_trades:
        _project_live_trade_plan(
            paper_trades,
            initial_cash=cash_available,
            actions=actions,
            liquidity=live_liquidity,
            buy_universe_label="S&P 100" if not request_body.symbols else "Configured universe",
        )

        for entry in paper_trades:
            quantity = int(entry.get("quantity", 0))
            if quantity == 0:
                continue
            direction = "buy" if quantity > 0 else "sell"
            actions.append(
                {
                    "type": "trade_signal",
                    "symbol": entry.get("symbol"),
                    "direction": direction,
                    "quantity": quantity,
                    "price": entry.get("price"),
                    "timestamp": entry.get("timestamp", timestamp_now),
                }
            )
    else:
        actions.append({"type": "trade_signal", "status": "no_trades_today"})
        warnings.append(
            "No qualifying momentum trades were generated for the current trading interval."
        )

    payload_positions = {
        symbol: dict(details)
        for symbol, details in live_positions.items()
    }
    ibkr_positions_payload = {
        str(symbol).strip().upper(): dict(info)
        for symbol, info in positions_snapshot.items()
        if isinstance(info, dict)
    }

    quote_snapshot: Dict[str, Dict[str, Any]] = {}
    for symbol, entry in quotes.items():
        if not isinstance(entry, dict):
            continue
        price_value = _coerce_float(entry.get("price"))
        if price_value is None or price_value <= 0:
            continue
        snapshot_entry = dict(entry)
        snapshot_entry["price"] = float(price_value)
        quote_snapshot[str(symbol).strip().upper()] = snapshot_entry

    payload = {
        "mode": "live",
        "symbols": universe_symbols,
        "initial_cash": initial_cash,
        "paper_trades": paper_trades,
        "actions": actions,
        "warnings": warnings,
        "portfolio": {
            "positions": payload_positions,
            "liquidity": live_liquidity,
            "cash": cash_available,
        },
        "liquidity": live_liquidity,
        "ibkr_positions": ibkr_positions_payload,
        "execution_requested": bool(request_body.execute_orders),
        "trade_date": date.today().isoformat(),
    }

    if quote_snapshot:
        payload["quotes_snapshot"] = quote_snapshot
    if cached_hits:
        payload["quote_cache_hits"] = cached_hits

    if scored:
        payload["momentum_scores"] = [
            {"symbol": symbol, "score": score}
            for symbol, score in scored[: min(50, len(scored))]
        ]

    return payload


def _normalize_polygon_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric > 1e18:  # nanoseconds
        seconds = numeric / 1_000_000_000
    elif numeric > 1e15:  # microseconds
        seconds = numeric / 1_000_000
    elif numeric > 1e12:  # milliseconds
        seconds = numeric / 1_000
    else:
        seconds = numeric

    try:
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.isoformat()


def _extract_cash_breakdown_from_actions(
    actions: Sequence[Dict[str, Any]]
) -> Dict[str, float]:
    for action in reversed(actions):
        if not isinstance(action, dict):
            continue
        if action.get("type") != "ibkr_cash_snapshot":
            continue
        candidates = action.get("candidates")
        if not isinstance(candidates, dict):
            continue
        breakdown: Dict[str, float] = {}
        for currency, info in candidates.items():
            if not isinstance(info, dict):
                continue
            value = info.get("value")
            if value is None:
                continue
            try:
                breakdown[str(currency)] = float(value)
            except (TypeError, ValueError):
                continue
        if breakdown:
            return breakdown
    return {}


def _calculate_portfolio_liquidity(
    initial_cash: float,
    positions: Dict[str, Dict[str, Any]] | None,
    actions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    holdings_value = 0.0
    position_values: Dict[str, float] = {}
    position_count = 0
    if positions:
        for symbol, info in positions.items():
            if not isinstance(info, dict):
                continue
            try:
                quantity = int(info.get("quantity", 0))
            except (TypeError, ValueError):
                quantity = 0
            if quantity == 0:
                continue
            value: float | None = None
            market_value = info.get("market_value")
            if market_value is not None:
                try:
                    value = float(market_value)
                except (TypeError, ValueError):
                    value = None
            if value is None:
                market_price = info.get("market_price")
                if market_price is not None:
                    try:
                        value = float(market_price) * abs(quantity)
                    except (TypeError, ValueError):
                        value = None
            if value is None:
                avg_cost = info.get("avg_cost")
                if avg_cost is not None:
                    try:
                        value = float(avg_cost) * abs(quantity)
                    except (TypeError, ValueError):
                        value = None
            if value is None:
                value = 0.0

            position_values[str(symbol)] = float(value)
            holdings_value += float(value)
            position_count += 1

    total_assets = float(initial_cash) + holdings_value
    cash_breakdown = _extract_cash_breakdown_from_actions(actions)
    payload = {
        "cash_available": float(initial_cash),
        "holdings_value": float(holdings_value),
        "total_assets": float(total_assets),
        "positions_value": position_values,
        "position_count": position_count,
    }
    if cash_breakdown:
        payload["cash_breakdown"] = cash_breakdown
    return payload


def _extract_trade_price(entry: Dict[str, Any]) -> float | None:
    price_candidates = [
        entry.get("price"),
        entry.get("quote_price"),
        entry.get("paper_price"),
    ]
    for candidate in price_candidates:
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _partition_trades_by_direction(
    trades: Sequence[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sells: List[Dict[str, Any]] = []
    buys: List[Dict[str, Any]] = []
    neutral: List[Dict[str, Any]] = []
    for entry in trades:
        if not isinstance(entry, dict):
            continue
        try:
            quantity = int(entry.get("quantity", 0))
        except (TypeError, ValueError):
            neutral.append(entry)
            continue
        if quantity < 0:
            sells.append(entry)
        elif quantity > 0:
            buys.append(entry)
        else:
            neutral.append(entry)
    return sells, buys, neutral


def _estimate_trade_value(trades: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    for entry in trades:
        if not isinstance(entry, dict):
            continue
        price = _extract_trade_price(entry)
        if price is None:
            continue
        try:
            quantity = abs(int(entry.get("quantity", 0)))
        except (TypeError, ValueError):
            continue
        total += float(quantity) * price
    return total


def _project_live_trade_plan(
    trades: List[Dict[str, Any]] | None,
    *,
    initial_cash: float,
    actions: List[Dict[str, Any]],
    liquidity: Dict[str, Any],
    buy_universe_label: str,
) -> None:
    if not trades or not isinstance(trades, list):
        return

    sells, buys, neutral = _partition_trades_by_direction(trades)
    if not sells and not buys:
        return

    reordered = sells + buys + neutral
    trades[:] = reordered

    sell_symbols = [
        str(entry.get("symbol", "")).strip().upper()
        for entry in sells
        if entry.get("symbol")
    ]
    buy_symbols = [
        str(entry.get("symbol", "")).strip().upper()
        for entry in buys
        if entry.get("symbol")
    ]

    total_sell_value = _estimate_trade_value(sells)
    total_buy_value = _estimate_trade_value(buys)
    available_cash_after_sells = float(initial_cash) + total_sell_value

    liquidity["estimated_sell_value"] = total_sell_value
    liquidity["estimated_buy_value"] = total_buy_value
    liquidity["projected_cash_after_sells"] = available_cash_after_sells

    actions.append(
        {
            "type": "momentum_live_sequence",
            "sell_count": len(sells),
            "sell_symbols": sell_symbols[:20],
            "buy_count": len(buys),
            "buy_symbols": buy_symbols[:20],
            "sequence": ["sell", "buy"],
            "available_cash_before_buys": float(initial_cash),
            "projected_cash_after_sells": available_cash_after_sells,
            "buy_universe": buy_universe_label,
        }
    )


def _fetch_polygon_prev_close_quote(
    session: requests.Session,
    symbol: str,
    api_key: str,
) -> tuple[Dict[str, Any] | None, str | None]:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
    try:
        response = session.get(
            url,
            params={"apiKey": api_key, "adjusted": "true"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure path
        return None, f"Polygon previous close request for {symbol} failed: {exc}"

    payload = response.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return None, f"Polygon previous close response for {symbol} did not contain results."

    result = results[0]
    price = result.get("c") or result.get("close")
    if price is None:
        return None, f"Polygon previous close for {symbol} missing price field."

    timestamp = result.get("t")
    quote = {
        "price": float(price),
        "timestamp": _normalize_polygon_timestamp(timestamp),
        "source": "previous_close",
    }
    volume = result.get("v") or result.get("volume")
    if volume is not None:
        try:
            quote["size"] = int(volume)
        except (TypeError, ValueError):
            pass
    return quote, None


def _fetch_polygon_last_trade_quotes(
    symbols: Sequence[str],
    *,
    settings: AppSettings,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    normalized_symbols = [str(symbol).strip().upper()
                          for symbol in symbols if str(symbol).strip()]
    unique_symbols: List[str] = []
    seen: set[str] = set()
    for symbol in normalized_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)

    if not unique_symbols:
        return {}, actions, warnings

    api_key = settings.polygon_api_key
    if not api_key:
        warnings.append(
            "Polygon API key not configured; cannot fetch real-time quotes.")
        actions.append(
            {
                "type": "polygon_quote_fetch",
                "status": "skipped",
                "reason": "missing_api_key",
                "symbol_count": len(unique_symbols),
            }
        )
        return {}, actions, warnings

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    quotes: Dict[str, Dict[str, Any]] = {}
    try:
        for symbol in unique_symbols:
            url = f"https://api.polygon.io/v2/last/trade/{symbol}"
            try:
                response = session.get(
                    url,
                    params={"apiKey": api_key},
                    timeout=10,
                )
                if response.status_code == 403:
                    fallback_quote, fallback_warning = _fetch_polygon_prev_close_quote(
                        session, symbol, api_key
                    )
                    if fallback_quote:
                        quotes[symbol] = fallback_quote
                        actions.append(
                            {
                                "type": "polygon_quote",
                                "symbol": symbol,
                                "status": "fallback_prev_close",
                                "price": fallback_quote.get("price"),
                                "timestamp": fallback_quote.get("timestamp"),
                            }
                        )
                        warnings.append(
                            f"Polygon real-time quote for {symbol} returned 403; using previous close price."
                        )
                        if fallback_warning:
                            warnings.append(fallback_warning)
                        continue

                    message = fallback_warning or "Unable to retrieve previous close price."
                    warnings.append(
                        f"Polygon quote for {symbol} returned 403 Forbidden. {message}"
                    )
                    actions.append(
                        {
                            "type": "polygon_quote",
                            "symbol": symbol,
                            "status": "failed",
                            "message": "403 Forbidden",
                        }
                    )
                    continue

                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - network failure path
                warnings.append(
                    f"Failed to fetch Polygon quote for {symbol}: {exc}")
                actions.append(
                    {
                        "type": "polygon_quote",
                        "symbol": symbol,
                        "status": "failed",
                        "message": str(exc),
                    }
                )
                continue

            last_trade = None
            if isinstance(payload, dict):
                last_trade = payload.get("last") or payload.get("results")

            if not isinstance(last_trade, dict):
                warnings.append(
                    f"Polygon quote for {symbol} returned no trade data."
                )
                actions.append(
                    {
                        "type": "polygon_quote",
                        "symbol": symbol,
                        "status": "empty",
                    }
                )
                continue

            price = last_trade.get("price") or last_trade.get("p")
            if price is None:
                warnings.append(
                    f"Polygon quote for {symbol} missing price field."
                )
                actions.append(
                    {
                        "type": "polygon_quote",
                        "symbol": symbol,
                        "status": "invalid",
                    }
                )
                continue

            timestamp = (
                last_trade.get("timestamp")
                or last_trade.get("t")
                or payload.get("timestamp")
            )
            timestamp_iso = _normalize_polygon_timestamp(timestamp)

            quote = {
                "price": float(price),
                "timestamp": timestamp_iso,
            }
            size = last_trade.get("size") or last_trade.get("s")
            exchange = last_trade.get("exchange") or last_trade.get("x")
            if size is not None:
                quote["size"] = int(size)
            if exchange is not None:
                quote["exchange"] = exchange

            quotes[symbol] = quote
            actions.append(
                {
                    "type": "polygon_quote",
                    "symbol": symbol,
                    "status": "ok",
                    "price": quote["price"],
                    "timestamp": timestamp_iso,
                }
            )
    finally:
        session.close()

    return quotes, actions, warnings


def _is_recent_timestamp(value: Any, *, max_age: timedelta) -> bool:
    if value in (None, ""):
        return False
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    except Exception:
        return False
    now = datetime.now(timezone.utc)
    return now - ts <= max_age


def _normalize_symbol_for_ibkr(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if "." in normalized:
        normalized = normalized.replace(".", " ")
    return normalized


def _extract_first_valid_price(*values: Any) -> float | None:
    for raw in values:
        price = _coerce_float(raw)
        if price is None:
            continue
        if pd.isna(price) or price <= 0:
            continue
        return float(price)
    return None


def _fetch_ibkr_live_quotes(
    symbols: Sequence[str],
    *,
    settings: AppSettings,
    snapshot_timeout: float = 5.0,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    quotes: Dict[str, Dict[str, Any]] = {}
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    unique_symbols: Dict[str, str] = {}
    for symbol in symbols:
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue
        if normalized not in unique_symbols:
            unique_symbols[normalized] = _normalize_symbol_for_ibkr(normalized)

    if not unique_symbols:
        return quotes, actions, warnings

    try:
        from ib_insync import IB, util  # type: ignore[import]
        from ib_insync.contract import Contract  # type: ignore[import]
    except ImportError:  # pragma: no cover - runtime guard
        warnings.append(
            "ib-insync is not available; cannot fetch IBKR live quotes."
        )
        actions.append(
            {
                "type": "ibkr_quote_fetch",
                "status": "skipped",
                "reason": "missing_dependency",
            }
        )
        return quotes, actions, warnings

    util.startLoop()

    ib: Any | None = None

    try:

        connection_attempts = _ibkr_connection_candidates(settings)
        last_error: BaseException | None = None
        for attempt_index, (host, port, client_id) in enumerate(connection_attempts, start=1):
            connect_action = {
                "type": "ibkr_connect",
                "host": host,
                "port": port,
                "client_id": client_id,
                "status": "pending",
                "purpose": "quote_fetch",
                "attempt": attempt_index,
            }
            actions.append(connect_action)

            candidate_ib = IB()
            try:
                candidate_ib.connect(
                    host, port, clientId=client_id, timeout=10)
                if not candidate_ib.isConnected():
                    raise RuntimeError("Failed to establish IBKR connection")
                connect_action["status"] = "connected"
                ib = candidate_ib
                break
            except Exception as exc:  # pragma: no cover - connection failure path
                formatted = _format_exception_message(exc)
                connect_action["status"] = "failed"
                connect_action["error"] = formatted
                last_error = exc
                try:
                    if candidate_ib.isConnected():
                        candidate_ib.disconnect()
                except Exception:
                    pass
                continue

        if ib is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Unable to connect to IBKR API.")

        for fetch_index, (symbol, ib_symbol) in enumerate(unique_symbols.items(), start=1):
            fetch_action = {
                "type": "ibkr_quote_fetch",
                "symbol": symbol,
                "ib_symbol": ib_symbol,
                "status": "pending",
                "attempt": fetch_index,
            }
            actions.append(fetch_action)

            try:
                spec = ContractSpec(symbol=ib_symbol)
                contract_kwargs = spec.as_ibkr_kwargs()
                contract = Contract(**contract_kwargs)
                ib.qualifyContracts(contract)

                request_started = time.time()
                try:
                    ticker = ib.reqMktData(
                        contract,
                        "",
                        snapshot=True,
                        regulatorySnapshot=False,
                    )
                except TypeError:
                    # Older ib_insync versions require keyword for snapshot flag.
                    ticker = ib.reqMktData(contract, "", True, False)

                deadline = time.time() + float(snapshot_timeout)
                price: float | None = None
                last_value: float | None = None
                bid_value: float | None = None
                ask_value: float | None = None

                while time.time() < deadline:
                    ib.waitOnUpdate(timeout=0.25)
                    last_value = _extract_first_valid_price(
                        getattr(ticker, "last", None)) or last_value
                    bid_value = _extract_first_valid_price(
                        getattr(ticker, "bid", None)) or bid_value
                    ask_value = _extract_first_valid_price(
                        getattr(ticker, "ask", None)) or ask_value
                    market_price = None
                    try:
                        market_price = ticker.marketPrice()
                    except Exception:
                        market_price = None
                    midpoint = getattr(ticker, "midpoint", None)
                    close_value = getattr(ticker, "close", None)
                    price = _extract_first_valid_price(
                        last_value,
                        market_price,
                        midpoint,
                        close_value,
                        bid_value,
                        ask_value,
                    )
                    if price is not None:
                        break

                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass

                if price is None:
                    fetch_action["status"] = "stale"
                    warnings.append(
                        f"IBKR snapshot did not return a tradeable price for {symbol}."
                    )
                    continue

                timestamp_raw = getattr(ticker, "time", None)
                if not timestamp_raw:
                    last_trade = getattr(ticker, "lastTrade", None)
                    if last_trade is not None:
                        timestamp_raw = getattr(last_trade, "time", None)
                if not timestamp_raw:
                    timestamp_raw = getattr(ticker, "lastTimestamp", None)
                if not timestamp_raw:
                    timestamp_raw = datetime.now(timezone.utc)

                quote_payload: Dict[str, Any] = {
                    "price": float(price),
                    "timestamp": _to_iso(timestamp_raw),
                    "source": "ibkr_snapshot",
                }
                if last_value is not None:
                    quote_payload["last"] = float(last_value)
                if bid_value is not None:
                    quote_payload["bid"] = float(bid_value)
                if ask_value is not None:
                    quote_payload["ask"] = float(ask_value)

                quotes[symbol] = quote_payload
                fetch_action["status"] = "ok"
                fetch_action["price"] = float(price)
                fetch_action["elapsed_ms"] = int(
                    (time.time() - request_started) * 1_000)
            except Exception as exc:  # pragma: no cover - defensive guard
                formatted = _format_exception_message(exc)
                fetch_action["status"] = "failed"
                fetch_action["error"] = formatted
                warnings.append(
                    f"Failed to fetch IBKR quote for {symbol}: {formatted}")

    except Exception as exc:  # pragma: no cover - defensive guard
        formatted = _format_exception_message(exc)
        warnings.append(f"Failed to fetch IBKR quotes: {formatted}")
        actions.append(
            {
                "type": "ibkr_quote_fetch",
                "status": "failed",
                "message": formatted,
            }
        )
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            actions.append(
                {
                    "type": "ibkr_disconnect",
                    "status": "disconnected",
                    "purpose": "quote_fetch",
                }
            )

    return quotes, actions, warnings


def _fetch_polygon_live_quotes(
    symbols: Sequence[str],
    *,
    settings: AppSettings,
    max_age_minutes: int = 15,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    quotes, actions, warnings = _fetch_polygon_last_trade_quotes(
        symbols,
        settings=settings,
    )

    if not quotes:
        return quotes, actions, warnings

    max_age = timedelta(minutes=max_age_minutes)
    api_key = settings.polygon_api_key

    session: requests.Session | None = None
    try:
        stale_symbols: list[str] = []
        for symbol, quote in list(quotes.items()):
            timestamp = quote.get("timestamp")
            if _is_recent_timestamp(timestamp, max_age=max_age):
                quote.setdefault("source", "last_trade")
                continue

            if api_key:
                if session is None:
                    session = requests.Session()
                    session.headers.update({
                        "Authorization": f"Bearer {api_key}",
                    })
                fallback_quote, fallback_warning = _fetch_polygon_prev_close_quote(
                    session,
                    symbol,
                    api_key,
                )
                if fallback_warning:
                    warnings.append(fallback_warning)
                if fallback_quote:
                    fallback_quote.setdefault("source", "previous_close")
                    actions.append(
                        {
                            "type": "polygon_quote_fallback",
                            "symbol": symbol,
                            "status": "stale_last_trade",
                            "fallback": "previous_close",
                        }
                    )
                    quotes[symbol] = fallback_quote
                    continue

            actions.append(
                {
                    "type": "polygon_quote_fetch",
                    "symbol": symbol,
                    "status": "stale",
                    "reason": "unavailable_recent_quote",
                }
            )
            warnings.append(
                f"No recent Polygon quote (≤{max_age_minutes} minutes) available for {symbol}; skipping symbol."
            )
            stale_symbols.append(symbol)

        for symbol in stale_symbols:
            quotes.pop(symbol, None)
    finally:
        if session is not None:
            session.close()

    return quotes, actions, warnings


def _compute_realtime_momentum_scores(
    symbols: Sequence[str],
    *,
    store: ParquetBarStore,
    bar_size: str,
    quotes: Dict[str, Dict[str, Any]],
    parameters: MomentumParameters,
) -> tuple[List[tuple[str, float]], Dict[str, float], List[str]]:
    scored: List[tuple[str, float]] = []
    scores_map: Dict[str, float] = {}
    warnings: List[str] = []

    if not symbols:
        return scored, scores_map, warnings

    required_history = (
        int(parameters.lookback_days)
        + int(parameters.skip_days)
        + 1
    )

    for symbol in symbols:
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue

        quote = quotes.get(normalized)
        price = quote.get("price") if isinstance(quote, dict) else None
        if price is None or price <= 0:
            warnings.append(
                f"Skipping {normalized}: missing or invalid real-time price."
            )
            continue

        try:
            frame = store.load(normalized, bar_size)
        except FileNotFoundError:
            warnings.append(
                f"Historical data for {normalized} not available in store; skipping symbol."
            )
            continue
        except Exception as exc:
            warnings.append(
                f"Failed to load history for {normalized}: {_format_exception_message(exc)}"
            )
            continue

        closes = frame.get("close")
        if closes is None:
            warnings.append(
                f"Historical bars for {normalized} missing close column.")
            continue

        close_series = [float(value)
                        for value in closes.to_list() if value is not None]
        close_series.append(float(price))

        if len(close_series) <= required_history:
            warnings.append(
                f"Insufficient history for {normalized}; need at least {required_history + 1} closes."
            )
            continue

        base_index = len(
            close_series) - int(parameters.lookback_days) - int(parameters.skip_days) - 1
        if base_index < 0 or base_index >= len(close_series):
            warnings.append(
                f"History window for {normalized} is too short for configured lookback/skip settings."
            )
            continue

        base_price = close_series[base_index]
        recent_price = close_series[-1]
        if base_price <= 0 or recent_price <= 0:
            continue

        momentum = (recent_price / base_price) - 1.0
        if momentum < float(parameters.min_momentum):
            continue

        score = momentum
        volatility_window = int(parameters.volatility_window)
        if volatility_window > 1 and len(close_series) >= volatility_window + 1:
            window = close_series[-(volatility_window + 1):]
            returns = [
                (window[idx] / window[idx - 1]) - 1.0
                for idx in range(1, len(window))
                if window[idx - 1] > 0
            ]
            if returns and float(parameters.volatility_exponent) > 0:
                vol = statistics.pstdev(returns)
                if vol > 0:
                    score = momentum / \
                        (vol ** float(parameters.volatility_exponent))

        scored.append((normalized, score))
        scores_map[normalized] = score

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored, scores_map, warnings


def _read_live_state(store_path: Path | None) -> tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if not store_path:
        return {}, warnings

    try:
        base_path = Path(store_path)
    except TypeError:
        warnings.append(
            "Momentum live store path is invalid; using empty state.")
        return {}, warnings

    state_path = base_path / MOMENTUM_LIVE_STATE_FILE
    if not state_path.exists():
        return {}, warnings

    try:
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        warnings.append(f"Failed to read momentum live state: {exc}")
        return {}, warnings

    if isinstance(payload, dict):
        return payload, warnings

    warnings.append(
        "Momentum live state file contained unexpected content; ignoring.")
    return {}, warnings


def _write_live_state(store_path: Path | None, state: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if not store_path:
        return warnings

    try:
        base_path = Path(store_path)
    except TypeError:
        warnings.append(
            "Momentum live store path is invalid; skipping state persistence.")
        return warnings

    try:
        base_path.mkdir(parents=True, exist_ok=True)
        state_path = base_path / MOMENTUM_LIVE_STATE_FILE
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(state_path)
    except Exception as exc:
        warnings.append(f"Failed to persist momentum live state: {exc}")

    return warnings


def _load_persisted_initial_cash(store_path: Path | None) -> tuple[float | None, List[str]]:
    state, warnings = _read_live_state(store_path)

    raw_value = state.get("last_initial_cash") if isinstance(
        state, dict) else None
    if raw_value is None:
        return None, warnings

    try:
        return float(raw_value), warnings
    except (TypeError, ValueError):
        warnings.append(
            "Momentum live state file contained an invalid cash value.")
        return None, warnings


def _persist_initial_cash_to_disk(store_path: Path | None, value: float) -> List[str]:
    warnings: List[str] = []
    state, read_warnings = _read_live_state(store_path)
    warnings.extend(read_warnings)

    state["last_initial_cash"] = float(value)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_warnings = _write_live_state(store_path, state)
    warnings.extend(write_warnings)
    return warnings


def _load_persisted_trade_history(
    store_path: Path | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    state, warnings = _read_live_state(store_path)

    trade_history: List[Dict[str, Any]] = []
    raw_history = state.get("trade_history") if isinstance(
        state, dict) else None
    if isinstance(raw_history, list):
        for entry in raw_history:
            if isinstance(entry, dict):
                trade_history.append(dict(entry))
    elif raw_history not in (None, []):
        warnings.append(
            "Momentum live state contained malformed trade history.")

    equity_curve: List[Dict[str, Any]] = []
    raw_equity = state.get("equity_curve") if isinstance(state, dict) else None
    if isinstance(raw_equity, list):
        for entry in raw_equity:
            if isinstance(entry, dict):
                equity_curve.append(dict(entry))
    elif raw_equity not in (None, []):
        warnings.append(
            "Momentum live state contained malformed equity curve.")

    trade_history.sort(key=lambda item: str(item.get("timestamp", "")))
    equity_curve.sort(key=lambda item: str(item.get("timestamp", "")))

    return trade_history, equity_curve, warnings


def _persist_trade_history_to_disk(
    store_path: Path | None,
    trade_history: Sequence[Dict[str, Any]],
    equity_curve: Sequence[Dict[str, Any]],
) -> List[str]:
    state, warnings = _read_live_state(store_path)

    state["trade_history"] = [dict(entry) for entry in trade_history]
    state["equity_curve"] = [dict(entry) for entry in equity_curve]
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_warnings = _write_live_state(store_path, state)
    warnings.extend(write_warnings)
    return warnings


def _load_persisted_quote_cache(
    store_path: Path | None,
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    cache: Dict[str, Dict[str, Any]] = {}
    state, warnings = _read_live_state(store_path)

    raw_cache = state.get("quote_cache") if isinstance(state, dict) else None
    if isinstance(raw_cache, dict):
        for symbol, entry in raw_cache.items():
            if not isinstance(entry, dict):
                continue
            normalized = str(symbol).strip().upper()
            price_value = _coerce_float(entry.get("price"))
            if price_value is None or price_value <= 0:
                continue
            snapshot = dict(entry)
            snapshot["price"] = float(price_value)
            if "timestamp" in snapshot and snapshot["timestamp"] is None:
                snapshot.pop("timestamp")
            cache[normalized] = snapshot
    elif raw_cache not in (None, {}):
        warnings.append("Momentum live state contained malformed quote cache.")

    return cache, warnings


def _persist_quote_cache_to_disk(
    store_path: Path | None,
    quote_cache: Mapping[str, Dict[str, Any]],
) -> List[str]:
    state, warnings = _read_live_state(store_path)

    serialized_cache: Dict[str, Dict[str, Any]] = {}
    for symbol, entry in quote_cache.items():
        if not isinstance(entry, dict):
            continue
        price_value = _coerce_float(entry.get("price"))
        if price_value is None or price_value <= 0:
            continue
        normalized = str(symbol).strip().upper()
        snapshot = dict(entry)
        snapshot["price"] = float(price_value)
        serialized_cache[normalized] = snapshot

    if serialized_cache:
        state["quote_cache"] = serialized_cache
    else:
        state.pop("quote_cache", None)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_warnings = _write_live_state(store_path, state)
    warnings.extend(write_warnings)
    return warnings


def _reset_persisted_trade_history(store_path: Path | None) -> List[str]:
    state, warnings = _read_live_state(store_path)

    if isinstance(state, dict):
        state.pop("trade_history", None)
        state.pop("equity_curve", None)
        state.pop("quote_cache", None)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_warnings = _write_live_state(store_path, state)
    warnings.extend(write_warnings)
    return warnings


def _summarize_ibkr_cash_failure(actions: Sequence[Dict[str, Any]], warnings: Sequence[str]) -> str:
    snapshot: Dict[str, Any] | None = None
    for action in reversed(actions):
        if isinstance(action, dict) and action.get("type") == "ibkr_cash_snapshot":
            snapshot = action
            break

    parts: list[str] = []
    if snapshot:
        status = snapshot.get("status")
        if status and status != "ok":
            parts.append(f"status={status}")
        account = snapshot.get("account")
        if account:
            parts.append(f"account={account}")
        candidates = snapshot.get("candidates")
        if isinstance(candidates, dict) and candidates:
            preview_entries: list[str] = []
            for idx, (currency, info) in enumerate(candidates.items()):
                if idx >= 5:
                    preview_entries.append("…")
                    break
                value = info.get("value") if isinstance(info, dict) else None
                preview_entries.append(f"{currency}:{value}")
            if preview_entries:
                parts.append("candidates=" + ", ".join(preview_entries))
        summary = snapshot.get("summary")
        if isinstance(summary, list) and summary:
            summary_entries: list[str] = []
            for idx, entry in enumerate(summary):
                if idx >= 5:
                    summary_entries.append("…")
                    break
                if isinstance(entry, dict):
                    tag = entry.get("tag") or entry.get(
                        "parsed_currency") or "∅"
                    value = entry.get("value") or entry.get("parsed_value")
                    summary_entries.append(f"{tag}={value}")
            if summary_entries:
                parts.append("summary=" + ", ".join(summary_entries))
        attempts = snapshot.get("summary_attempts")
        if isinstance(attempts, list) and attempts:
            attempt_entries: list[str] = []
            for idx, attempt in enumerate(attempts):
                if idx >= 5:
                    attempt_entries.append("…")
                    break
                if isinstance(attempt, dict):
                    signature = attempt.get("signature") or "?"
                    error = attempt.get("error")
                    status = attempt.get("status")
                    if error:
                        attempt_entries.append(f"{signature}:{error}")
                    elif status:
                        attempt_entries.append(f"{signature}:{status}")
                    else:
                        attempt_entries.append(signature)
            if attempt_entries:
                parts.append("attempts=" + ", ".join(attempt_entries))

    for warning in warnings:
        if warning and isinstance(warning, str):
            parts.append(warning)

    return "Diagnostics: " + "; ".join(parts) if parts else ""


def _ibkr_connection_success(actions: Sequence[Dict[str, Any]], *, purpose: str) -> bool:
    connected = False
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("type") != "ibkr_connect":
            continue
        if purpose and action.get("purpose") not in {purpose, None}:
            continue
        status = str(action.get("status", "")).lower()
        if status == "connected":
            connected = True
        elif status in {"failed", "error", "skipped"}:
            return False
    return connected


def _format_exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _ibkr_connection_candidates(settings: AppSettings) -> List[tuple[str, int, int]]:
    host = settings.ibkr_host
    base_port = int(settings.ibkr_port)
    base_client_id = int(settings.ibkr_client_id)

    port_candidates: list[int] = [base_port]
    for fallback in (7497, 7496, 4002, 4001):
        if fallback not in port_candidates:
            port_candidates.append(fallback)

    client_candidates: list[int] = [base_client_id]
    for offset in range(1, 4):
        candidate = base_client_id + offset
        if candidate not in client_candidates:
            client_candidates.append(candidate)

    attempts: list[tuple[str, int, int]] = []
    for port in port_candidates:
        for client_id in client_candidates:
            attempts.append((host, port, client_id))
    return attempts


def _fetch_ibkr_account_cash(*, settings: AppSettings) -> tuple[float | None, List[Dict[str, Any]], List[str]]:
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    try:
        from ib_insync import IB, util
    except ImportError:  # pragma: no cover - runtime guard
        warnings.append(
            "ib-insync is not available; cannot read IBKR cash balance."
        )
        actions.append(
            {
                "type": "ibkr_cash_snapshot",
                "status": "skipped",
                "reason": "missing_dependency",
            }
        )
        return None, actions, warnings

    util.startLoop()

    ib: Any | None = None

    try:

        connection_attempts = _ibkr_connection_candidates(settings)
        last_error: BaseException | None = None
        for attempt_index, (host, port, client_id) in enumerate(connection_attempts, start=1):
            connect_action = {
                "type": "ibkr_connect",
                "host": host,
                "port": port,
                "client_id": client_id,
                "status": "pending",
                "purpose": "cash_fetch",
                "attempt": attempt_index,
            }
            actions.append(connect_action)

            candidate_ib = IB()
            try:
                candidate_ib.connect(
                    host, port, clientId=client_id, timeout=10)
                if not candidate_ib.isConnected():
                    raise RuntimeError("Failed to establish IBKR connection")
                connect_action["status"] = "connected"
                ib = candidate_ib
                break
            except Exception as exc:  # pragma: no cover - connection failure path
                formatted = _format_exception_message(exc)
                connect_action["status"] = "failed"
                connect_action["error"] = formatted
                last_error = exc
                try:
                    if candidate_ib.isConnected():
                        candidate_ib.disconnect()
                except Exception:
                    pass
                continue

        if ib is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Unable to connect to IBKR API.")

        summary_tags = "TotalCashValue,AvailableFunds,NetLiquidation,CashBalance,TotalCashBalance,ExcessLiquidity,SettledCash,EquityWithLoanValue"
        summary_group: str | None = "All"
        summary_attempts: list[dict[str, Any]] = []
        summary_items: list[Any] = []
        summary_loaded = False
        summary_request: Any | None = None

        def _record_summary_attempt(status: str, *, signature: str, error: Exception | None = None) -> None:
            entry: dict[str, Any] = {"status": status, "signature": signature}
            if error is not None:
                entry["error"] = str(error)
            summary_attempts.append(entry)

        def _format_attempts() -> str:
            if not summary_attempts:
                return "no attempts recorded"
            return "; ".join(
                f"{attempt['signature']} => {attempt.get('error', attempt['status'])}"
                for attempt in summary_attempts
            )

        account_summary_fn = getattr(ib, "accountSummary", None)
        if callable(account_summary_fn):
            try:
                summary_items = list(account_summary_fn())
                summary_group = "All"
                summary_loaded = True
                _record_summary_attempt("ok", signature="account_summary")
            except Exception as exc:
                _record_summary_attempt(
                    "failed", signature="account_summary", error=exc)

        if not summary_loaded:
            req_account_summary_fn = getattr(ib, "reqAccountSummary", None)
            if not callable(req_account_summary_fn):
                _record_summary_attempt(
                    "failed",
                    signature="req_account_summary",
                    error=RuntimeError("reqAccountSummary not available"),
                )
                raise RuntimeError(
                    f"reqAccountSummary invocation failed: {_format_attempts()}")

            try:
                summary_request = req_account_summary_fn("All", summary_tags)
                summary_group = "All"
                summary_loaded = True
                _record_summary_attempt("ok", signature="group_and_tags")
            except TypeError as exc:
                _record_summary_attempt(
                    "failed", signature="group_and_tags", error=exc)
                summary_request = None
                summary_loaded = False
            except Exception as exc:
                _record_summary_attempt(
                    "failed", signature="group_and_tags", error=exc)
                summary_request = None
                summary_loaded = False

            if not summary_loaded:
                try:
                    summary_request = req_account_summary_fn(summary_tags)
                    summary_group = None
                    summary_loaded = True
                    _record_summary_attempt("ok", signature="tags_only")
                except TypeError as exc:
                    _record_summary_attempt(
                        "failed", signature="tags_only", error=exc)
                    summary_request = None
                    summary_loaded = False
                except Exception as exc:
                    _record_summary_attempt(
                        "failed", signature="tags_only", error=exc)
                    summary_request = None
                    summary_loaded = False

            if not summary_loaded or summary_request is None:
                raise RuntimeError(
                    f"reqAccountSummary invocation failed: {_format_attempts()}")

            deadline = time.time() + 5.0
            while not summary_request and time.time() < deadline:
                time.sleep(0.1)

            summary_items = list(summary_request)
            cancel_fn = getattr(ib, "cancelAccountSummary", None)
            if callable(cancel_fn):  # pragma: no branch - defensive cleanup
                try:
                    cancel_fn(summary_request)
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    _record_summary_attempt(
                        "failed", signature="cancel", error=exc)

        priority_order = {
            "totalcashvalue": 0,
            "cashbalance": 1,
            "totalcashbalance": 1,
            "settledcash": 2,
            "availablefunds": 3,
            "excessliquidity": 4,
            "equitywithloanvalue": 5,
            "netliquidation": 6,
        }

        base_currency = getattr(settings, "ibkr_base_currency", "USD")
        base_currency_upper = str(base_currency or "").upper()
        base_tokens = {
            token for token in re.split(r"[^A-Z]", base_currency_upper) if token
        }
        if base_currency_upper:
            base_tokens.add(base_currency_upper)
        base_tokens.add("BASE")

        numeric_pattern = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

        def _parse_numeric(raw: Any) -> float | None:
            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return None
                match = numeric_pattern.search(text)
                if not match:
                    text_no_space = text.replace(" ", "")
                    match = numeric_pattern.search(text_no_space)
                if not match:
                    return None
                number_text = match.group(0).replace(",", "")
                try:
                    return float(number_text)
                except ValueError:
                    return None
            return None

        def _currency_tokens(currency: Any) -> tuple[str, set[str]]:
            if not isinstance(currency, str):
                return "", set()
            currency_upper = currency.upper().strip()
            tokens = {
                token for token in re.split(r"[^A-Z]", currency_upper) if token
            }
            if currency_upper and not tokens:
                tokens = {currency_upper}
            return currency_upper, tokens

        summary_snapshot: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        configured_account = None
        if hasattr(settings, "ibkr_account") and settings.ibkr_account:
            configured_account = str(settings.ibkr_account).strip().upper()
        for item in summary_items:
            tag = str(getattr(item, "tag", "") or "")
            tag_lower = tag.lower()
            currency_raw = getattr(item, "currency", "")
            currency_upper, currency_token_set = _currency_tokens(currency_raw)
            value_raw = getattr(item, "value", None)
            parsed_value = _parse_numeric(value_raw)
            account_raw = getattr(item, "account", None)
            account_normalized = None
            if isinstance(account_raw, str):
                account_normalized = account_raw.strip().upper() or None
            summary_snapshot.append(
                {
                    "tag": tag or None,
                    "currency": currency_raw if currency_raw != "" else None,
                    "parsed_currency": currency_upper or None,
                    "value": value_raw,
                    "parsed_value": parsed_value,
                    "account": account_raw,
                    "parsed_account": account_normalized,
                }
            )

            if tag_lower not in priority_order:
                continue
            if parsed_value is None:
                continue
            if configured_account and account_normalized and account_normalized != configured_account:
                continue
            candidates.append(
                {
                    "tag": tag,
                    "tag_lower": tag_lower,
                    "priority": priority_order[tag_lower],
                    "value": parsed_value,
                    "currency_upper": currency_upper,
                    "currency_tokens": currency_token_set,
                    "account": account_raw,
                }
            )

        chosen_candidate: dict[str, Any] | None = None
        if candidates:
            candidates.sort(key=lambda entry: entry["priority"])

            for candidate in candidates:
                candidate_tokens = candidate["currency_tokens"]
                currency_upper = candidate["currency_upper"]
                if not candidate_tokens and not currency_upper:
                    chosen_candidate = candidate
                    break
                if candidate_tokens.intersection(base_tokens) or (
                    currency_upper and currency_upper in base_tokens
                ):
                    chosen_candidate = candidate
                    break

            if chosen_candidate is None and base_currency_upper:
                for candidate in candidates:
                    tokens = candidate["currency_tokens"]
                    if any(base_currency_upper in token for token in tokens):
                        chosen_candidate = candidate
                        break

            if chosen_candidate is None:
                chosen_candidate = candidates[0]

        currency_values: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            label = candidate["currency_upper"] or base_currency_upper or "BASE"
            existing = currency_values.get(label)
            if existing is None or candidate["priority"] < existing["priority"]:
                currency_values[label] = {
                    "value": candidate["value"],
                    "tag": candidate["tag"],
                    "priority": candidate["priority"],
                }

        if chosen_candidate is not None:
            chosen_value = float(chosen_candidate["value"])
            chosen_tag = chosen_candidate["tag"]
            snapshot_payload = {
                "type": "ibkr_cash_snapshot",
                "tag": chosen_tag,
                "value": chosen_value,
                "currency": chosen_candidate.get("currency_upper") or base_currency_upper,
                "account": chosen_candidate.get("account"),
                "candidates": currency_values,
                "summary_attempts": summary_attempts,
                "summary_group": summary_group,
            }
            if chosen_value > 0:
                snapshot_payload["status"] = "ok"
                chosen_value_float = chosen_value
            else:
                snapshot_payload["status"] = "non_positive"
                warnings.append(
                    "IBKR account summary returned a non-positive cash balance; will use fallback value."
                )
                chosen_value_float = None
            actions.append(snapshot_payload)
        else:
            summary_preview = ", ".join(
                f"{(entry.get('tag') or entry.get('parsed_currency') or '∅')}={entry.get('value') or entry.get('parsed_value')}"
                for entry in summary_snapshot[:5]
                if isinstance(entry, dict)
            )
            candidate_preview = ", ".join(
                f"{currency}:{info.get('value')}"
                for currency, info in list(currency_values.items())[:5]
                if isinstance(info, dict)
            )
            details: list[str] = [
                "Could not determine IBKR cash balance from account summary.",
            ]
            if candidate_preview:
                details.append(f"Candidates => {candidate_preview}")
            if summary_preview:
                details.append(f"Summary => {summary_preview}")
            warnings.append(" ".join(details))
            actions.append(
                {
                    "type": "ibkr_cash_snapshot",
                    "status": "unavailable",
                    "summary": summary_snapshot[:15],
                    "candidates": currency_values,
                    "summary_attempts": summary_attempts,
                    "summary_group": summary_group,
                }
            )
            chosen_value_float = None
            if configured_account:
                warnings.append(
                    f"Configured IBKR account '{configured_account}' not found in account summary."
                )

        return chosen_value_float, actions, warnings
    except Exception as exc:  # pragma: no cover - defensive guard
        formatted = _format_exception_message(exc)
        warnings.append(f"Failed to fetch IBKR cash: {formatted}")
        actions.append(
            {
                "type": "ibkr_cash_snapshot_failed",
                "message": formatted,
            }
        )
        return None, actions, warnings
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            actions.append({"type": "ibkr_disconnect",
                           "status": "disconnected", "purpose": "cash_fetch"})


def _fetch_ibkr_positions(*, settings: AppSettings) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    positions: Dict[str, Dict[str, Any]] = {}

    try:
        from ib_insync import IB, util
    except ImportError:  # pragma: no cover - runtime guard
        warnings.append(
            "ib-insync is not available; cannot read IBKR open positions."
        )
        actions.append(
            {
                "type": "ibkr_positions_snapshot",
                "status": "skipped",
                "reason": "missing_dependency",
            }
        )
        return positions, actions, warnings

    util.startLoop()

    ib: Any | None = None

    try:

        connection_attempts = _ibkr_connection_candidates(settings)
        last_error: BaseException | None = None
        for attempt_index, (host, port, client_id) in enumerate(connection_attempts, start=1):
            connect_action = {
                "type": "ibkr_connect",
                "host": host,
                "port": port,
                "client_id": client_id,
                "status": "pending",
                "purpose": "positions_fetch",
                "attempt": attempt_index,
            }
            actions.append(connect_action)

            candidate_ib = IB()
            try:
                candidate_ib.connect(
                    host, port, clientId=client_id, timeout=10)
                if not candidate_ib.isConnected():
                    raise RuntimeError("Failed to establish IBKR connection")
                connect_action["status"] = "connected"
                ib = candidate_ib
                break
            except Exception as exc:  # pragma: no cover - connection failure path
                formatted = _format_exception_message(exc)
                connect_action["status"] = "failed"
                connect_action["error"] = formatted
                last_error = exc
                try:
                    if candidate_ib.isConnected():
                        candidate_ib.disconnect()
                except Exception:
                    pass
                continue

        if ib is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Unable to connect to IBKR API.")

        positions_fn = getattr(ib, "positions", None)
        if callable(positions_fn):
            ib_positions = list(positions_fn())
        else:
            ib_positions = []

        preview: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(ib_positions):
            contract = getattr(item, "contract", None)
            symbol = ""
            if contract is not None:
                symbol = str(
                    getattr(contract, "symbol", "")
                    or getattr(contract, "localSymbol", "")
                ).strip()
            if not symbol:
                continue
            symbol_upper = symbol.upper()

            quantity_raw = getattr(item, "position", 0)
            try:
                quantity = int(quantity_raw)
            except (TypeError, ValueError):
                try:
                    quantity = int(float(quantity_raw))
                except Exception:
                    quantity = 0

            avg_cost_raw = getattr(item, "avgCost", None)
            if avg_cost_raw is None:
                avg_cost_raw = getattr(item, "averageCost", None)
            market_price_raw = getattr(item, "marketPrice", None)
            market_value_raw = getattr(item, "marketValue", None)

            entry: Dict[str, Any] = {"quantity": quantity}
            if avg_cost_raw is not None:
                try:
                    entry["avg_cost"] = float(avg_cost_raw)
                except (TypeError, ValueError):
                    pass
            if market_price_raw is not None:
                try:
                    entry["market_price"] = float(market_price_raw)
                except (TypeError, ValueError):
                    pass
            if market_value_raw is not None:
                try:
                    entry["market_value"] = float(market_value_raw)
                except (TypeError, ValueError):
                    pass

            positions[symbol_upper] = entry
            if idx < 15:
                preview[symbol_upper] = {
                    "quantity": entry["quantity"],
                    "avg_cost": entry.get("avg_cost"),
                }

        actions.append(
            {
                "type": "ibkr_positions_snapshot",
                "status": "ok",
                "count": len(positions),
                "positions": preview,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        formatted = _format_exception_message(exc)
        warnings.append(f"Failed to fetch IBKR positions: {formatted}")
        actions.append(
            {
                "type": "ibkr_positions_snapshot",
                "status": "failed",
                "message": formatted,
            }
        )
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            actions.append(
                {
                    "type": "ibkr_disconnect",
                    "status": "disconnected",
                    "purpose": "positions_fetch",
                }
            )

    return positions, actions, warnings


def _summarize_live_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    paper_window = payload.get("paper_window")
    trade_date = None
    if isinstance(paper_window, dict):
        trade_date = paper_window.get("end") or paper_window.get("start")

    def _normalize_trade_timestamp(value: Any) -> str:
        try:
            return _to_iso(value)
        except Exception:
            return _to_iso(datetime.now(timezone.utc))

    def _normalize_trade_date(value: Any) -> str | None:
        if value in (None, ""):
            return None
        try:
            return pd.Timestamp(value).date().isoformat()
        except Exception:
            text = str(value)
            return text[:10] if text else None

    trades = payload.get("paper_trades")
    normalized_trades: list[Dict[str, Any]] = []
    if isinstance(trades, list):
        for entry in trades:
            if not isinstance(entry, dict):
                continue
            if entry.get("phase") and entry.get("phase") != "paper":
                continue

            normalized = dict(entry)
            quote_ts = normalized.get("quote_timestamp")
            original_ts = normalized.get("timestamp")
            if quote_ts:
                if "paper_timestamp" not in normalized and original_ts:
                    normalized["paper_timestamp"] = original_ts
                normalized["timestamp"] = _normalize_trade_timestamp(quote_ts)
            else:
                if "paper_timestamp" not in normalized and original_ts:
                    normalized["paper_timestamp"] = original_ts
                normalized["timestamp"] = _normalize_trade_timestamp(
                    datetime.now(timezone.utc)
                )
            normalized_trades.append(normalized)

    normalized_trades.sort(key=lambda entry: entry.get("timestamp") or "")

    derived_trade_date = None
    if normalized_trades:
        last_trade = normalized_trades[-1]
        derived_trade_date = _normalize_trade_date(last_trade.get("timestamp"))

    normalized_trade_date = _normalize_trade_date(trade_date)
    trade_date = derived_trade_date or normalized_trade_date

    evaluation = payload.get("evaluation")
    portfolio: Dict[str, Any] | None = None
    is_live_mode = str(payload.get("mode", "")).lower() == "live"
    if not is_live_mode and isinstance(evaluation, dict):
        final_state = evaluation.get("paper_final_state")
        if isinstance(final_state, dict):
            positions = final_state.get("positions", [])
            if isinstance(positions, list):
                positions = [pos for pos in positions if isinstance(pos, dict)]
                positions.sort(key=lambda item: str(item.get("symbol", "")))
            portfolio = {
                "cash": final_state.get("cash"),
                "positions": positions,
            }

    actions: list[Dict[str, Any]] = []
    for action in payload.get("actions", []):
        if not isinstance(action, dict):
            continue
        if action.get("type") == "experiment_run":
            continue
        actions.append(action)

    warning_messages: list[str] = []
    for warning in payload.get("warnings", []):
        if isinstance(warning, str):
            warning_messages.append(warning)
        elif isinstance(warning, dict):
            message = warning.get("reason") or warning.get("message")
            if message:
                warning_messages.append(str(message))

    summary: Dict[str, Any] = {
        "mode": "live",
        "symbols": payload.get("symbols"),
        "initial_cash": payload.get("initial_cash"),
        "execution_requested": bool(payload.get("execution_requested")),
        "paper_trades": normalized_trades,
        "warnings": warning_messages,
        "actions": actions,
        "trade_date": trade_date,
        "ibkr_positions": payload.get("ibkr_positions"),
    }

    payload_portfolio = payload.get("portfolio")
    if isinstance(payload_portfolio, dict):
        portfolio = portfolio or {}
        if payload_portfolio.get("positions") is not None:
            portfolio["positions"] = payload_portfolio.get("positions")
        if payload_portfolio.get("liquidity") is not None:
            portfolio["liquidity"] = payload_portfolio.get("liquidity")
        if payload_portfolio.get("cash") is not None:
            portfolio["cash"] = payload_portfolio.get("cash")

    liquidity_snapshot = payload.get("liquidity")
    if isinstance(liquidity_snapshot, dict):
        summary["liquidity"] = liquidity_snapshot

    if portfolio:
        summary["portfolio"] = portfolio

    quotes_snapshot = payload.get("quotes_snapshot")
    if isinstance(quotes_snapshot, dict):
        summary["quotes_snapshot"] = {
            str(symbol).strip().upper(): dict(entry)
            for symbol, entry in quotes_snapshot.items()
            if isinstance(entry, dict)
        }

    cache_hits = payload.get("quote_cache_hits")
    if isinstance(cache_hits, list) and cache_hits:
        summary["quote_cache_hits"] = [
            str(symbol).strip().upper()
            for symbol in cache_hits
            if isinstance(symbol, str) and symbol.strip()
        ]

    return summary


@dataclass
class LiveTraderRun:
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    status: str
    payload: Dict[str, Any] | None
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        duration = None
        if self.completed_at is not None:
            duration = (self.completed_at - self.started_at).total_seconds()

        payload = self.payload or {}
        actions = payload.get("actions", []) if isinstance(
            payload, dict) else []
        warnings = payload.get("warnings", []) if isinstance(
            payload, dict) else []
        paper_trades = payload.get(
            "paper_trades", []) if isinstance(payload, dict) else []
        order_count = sum(1 for action in actions if action.get(
            "type") == "ibkr_order_submitted")

        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "started_at": _to_iso(self.started_at),
            "completed_at": _to_iso(self.completed_at) if self.completed_at else "",
            "duration_seconds": duration,
            "trade_count": len(paper_trades),
            "order_count": order_count,
            "actions": actions,
            "warnings": warnings,
            "paper_trades": paper_trades,
            "symbols": payload.get("symbols") if isinstance(payload, dict) else None,
            "execution_requested": bool(payload.get("execution_requested")) if isinstance(payload, dict) else False,
            "trade_date": payload.get("trade_date") if isinstance(payload, dict) else None,
            "portfolio": payload.get("portfolio") if isinstance(payload, dict) else None,
            "initial_cash": payload.get("initial_cash") if isinstance(payload, dict) else None,
        }


class MomentumLiveTrader:
    def __init__(self, *, max_history: int = 100) -> None:
        self._lock = threading.RLock()
        self._history: deque[LiveTraderRun] = deque(maxlen=max_history)
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._config: MomentumLiveTradeRequest | None = None
        self._store_path: Path | None = None
        self._iterations = 0
        self._running = False
        self._status_message = "idle"
        self._started_at: datetime | None = None
        self._last_run_completed_at: datetime | None = None
        self._last_initial_cash = None
        self._last_ibkr_positions: Dict[str, Dict[str, Any]] = {}
        self._startup_actions: List[Dict[str, Any]] = []
        self._startup_warnings: List[str] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._equity_curve: List[Dict[str, Any]] = []
        self._state_loaded = False
        self._quote_cache: Dict[str, Dict[str, Any]] = {}
        self._quote_cache_loaded = False

    def start(self, request: MomentumLiveTradeRequest) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                raise RuntimeError("Momentum live trader is already running.")

            self._config = request.model_copy(deep=True)
            self._store_path = Path(
                self._config.store_path or DEFAULT_STORE_PATH
            ).expanduser().resolve()
            self._config.store_path = str(self._store_path)
            self._last_initial_cash = self._config.initial_cash
            self._startup_actions = []
            self._startup_warnings = []
            self._trade_history = []
            self._equity_curve = []
            self._state_loaded = False
            self._quote_cache = {}
            self._quote_cache_loaded = False

            self._iterations = 0
            self._running = True
            self._status_message = "running"
            self._started_at = datetime.now(timezone.utc)
            self._last_run_completed_at = None
            self._history.clear()
            stop_event = threading.Event()
            self._stop_event = stop_event
            thread = threading.Thread(
                target=self._run_loop,
                name="MomentumLiveTrader",
                daemon=True,
            )
            self._thread = thread

        persisted_cash_actions: List[Dict[str, Any]] = []
        persisted_cash_warnings: List[str] = []
        persisted_cash, persisted_cash_warnings = _load_persisted_initial_cash(
            self._store_path
        )
        loaded_from_persisted = False
        if persisted_cash is not None:
            with self._lock:
                config_ref = self._config
                if self._last_initial_cash is None:
                    self._last_initial_cash = float(persisted_cash)
                    loaded_from_persisted = True
                if (
                    config_ref is not None
                    and config_ref.initial_cash is None
                ):
                    config_ref.initial_cash = float(persisted_cash)
                    loaded_from_persisted = True
            if loaded_from_persisted:
                persisted_cash_actions.append(
                    {
                        "type": "initial_cash_loaded",
                        "source": "persisted_state",
                        "value": float(persisted_cash),
                    }
                )

        history_actions, history_warnings = self._load_persisted_history()
        quote_cache_actions, quote_cache_warnings = self._load_persisted_quote_cache()

        settings = AppSettings()
        try:
            startup_actions, startup_warnings = self._synchronize_ibkr_state(
                settings)
        except Exception as exc:  # pragma: no cover - defensive guard
            startup_actions = []
            startup_warnings = [f"Failed to synchronize IBKR state: {exc}"]

        with self._lock:
            combined_actions = (
                list(history_actions)
                + list(quote_cache_actions)
                + persisted_cash_actions
                + list(startup_actions)
            )
            combined_warnings = list(history_warnings)
            combined_warnings.extend(quote_cache_warnings)
            combined_warnings.extend(persisted_cash_warnings)
            combined_warnings.extend(startup_warnings)
            self._startup_actions = combined_actions
            self._startup_warnings = combined_warnings

        thread.start()

        return self.status()

    def stop(self) -> Dict[str, Any]:
        thread: threading.Thread | None
        with self._lock:
            if not self._running:
                return self.status()
            if self._stop_event:
                self._stop_event.set()
            thread = self._thread

        if thread:
            thread.join(timeout=10)

        with self._lock:
            self._running = False
            self._status_message = "stopped"
            self._thread = None
            self._stop_event = None

        return self.status()

    def status(self) -> Dict[str, Any]:
        self._ensure_state_loaded()
        with self._lock:
            config = self._config
            interval = config.interval_seconds if config else None
            next_run_at = None
            if self._running and self._last_run_completed_at and interval:
                next_run_at = self._last_run_completed_at + \
                    timedelta(seconds=interval)

            config_payload: Dict[str, Any] | None = None
            if config:
                initial_cash_value = self._last_initial_cash if self._last_initial_cash is not None else config.initial_cash
                config_payload = {
                    "initial_cash": float(initial_cash_value) if initial_cash_value is not None else None,
                    "interval_seconds": int(config.interval_seconds),
                    "auto_fetch": bool(config.auto_fetch),
                    "execute_orders": bool(config.execute_orders),
                    "symbols": config.symbols,
                    "paper_days": config.paper_days,
                    "training_years": config.training_years,
                    "store_path": str(self._store_path) if self._store_path else None,
                    "max_iterations": config.max_iterations,
                }

            startup_actions = list(self._startup_actions)
            startup_warnings = list(self._startup_warnings)
            last_ibkr_positions = {
                symbol: dict(details)
                for symbol, details in self._last_ibkr_positions.items()
                if isinstance(details, dict)
            }

            return {
                "running": self._running,
                "status": self._status_message,
                "iterations": self._iterations,
                "started_at": _to_iso(self._started_at),
                "last_run_at": _to_iso(self._last_run_completed_at),
                "next_run_at": _to_iso(next_run_at),
                "config": config_payload,
                "startup_actions": startup_actions,
                "startup_warnings": startup_warnings,
                "last_ibkr_positions": last_ibkr_positions,
                "last_ibkr_position_count": len(last_ibkr_positions),
                "trade_history_count": len(self._trade_history),
                "equity_curve_points": len(self._equity_curve),
                "cached_quote_count": len(self._quote_cache),
            }

    def history(self) -> Dict[str, Any]:
        self._ensure_state_loaded()
        with self._lock:
            runs = [run.to_dict() for run in list(self._history)]
            trade_history = [dict(entry) for entry in self._trade_history]
            equity_curve = [dict(point) for point in self._equity_curve]
        return {
            "runs": runs,
            "trade_history": trade_history,
            "equity_curve": equity_curve,
        }

    def reset_history(self) -> Dict[str, Any]:
        store_path = self._resolve_store_path()
        with self._lock:
            if self._running:
                raise RuntimeError(
                    "Momentum live trader is running; stop it before resetting history."
                )
            self._history.clear()
            self._trade_history = []
            self._equity_curve = []
            self._iterations = 0
            self._last_run_completed_at = None
            self._state_loaded = True
            self._quote_cache = {}
            self._quote_cache_loaded = True

        warnings = _reset_persisted_trade_history(store_path)
        return {
            "status": "reset",
            "warnings": warnings,
            "runs": [],
            "trade_history": [],
            "equity_curve": [],
        }

    # Internal helpers -------------------------------------------------

    def _resolve_store_path(self) -> Path:
        store_path = self._store_path
        if store_path is None:
            return Path(DEFAULT_STORE_PATH).expanduser().resolve()
        return Path(store_path).expanduser().resolve()

    def _ensure_state_loaded(self) -> None:
        history_actions, history_warnings = self._load_persisted_history()
        cache_actions, cache_warnings = self._load_persisted_quote_cache()
        if (
            not history_actions
            and not history_warnings
            and not cache_actions
            and not cache_warnings
        ):
            return
        with self._lock:
            if history_actions:
                self._startup_actions.extend(history_actions)
            if cache_actions:
                self._startup_actions.extend(cache_actions)
            if history_warnings:
                self._startup_warnings.extend(history_warnings)
            if cache_warnings:
                self._startup_warnings.extend(cache_warnings)

    def _load_persisted_history(self) -> tuple[List[Dict[str, Any]], List[str]]:
        with self._lock:
            if self._state_loaded:
                return [], []

        trade_history, equity_curve, warnings = _load_persisted_trade_history(
            self._resolve_store_path()
        )

        with self._lock:
            self._trade_history = trade_history
            self._equity_curve = equity_curve
            self._state_loaded = True

        actions: List[Dict[str, Any]] = []
        if trade_history or equity_curve:
            actions.append(
                {
                    "type": "live_history_loaded",
                    "trade_count": len(trade_history),
                    "equity_point_count": len(equity_curve),
                }
            )

        return actions, warnings

    def _load_persisted_quote_cache(self) -> tuple[List[Dict[str, Any]], List[str]]:
        with self._lock:
            if self._quote_cache_loaded:
                return [], []

        cache, warnings = _load_persisted_quote_cache(
            self._resolve_store_path()
        )

        actions: List[Dict[str, Any]] = []
        with self._lock:
            self._quote_cache = cache
            self._quote_cache_loaded = True
            if cache:
                actions.append(
                    {
                        "type": "quote_cache_loaded",
                        "symbol_count": len(cache),
                    }
                )

        return actions, warnings

    def _store_quote_snapshot(
        self, quotes: Mapping[str, Dict[str, Any]]
    ) -> List[str]:
        if not quotes:
            return []

        sanitized: Dict[str, Dict[str, Any]] = {}
        for symbol, entry in quotes.items():
            if not isinstance(entry, dict):
                continue
            price_value = _coerce_float(entry.get("price"))
            if price_value is None or price_value <= 0:
                continue
            normalized = str(symbol).strip().upper()
            snapshot = dict(entry)
            snapshot["price"] = float(price_value)
            sanitized[normalized] = snapshot

        if not sanitized:
            return []

        with self._lock:
            if not self._quote_cache_loaded:
                self._quote_cache = {}
                self._quote_cache_loaded = True
            for symbol, snapshot in sanitized.items():
                self._quote_cache[symbol] = snapshot
            cache_snapshot = {
                symbol: dict(entry)
                for symbol, entry in self._quote_cache.items()
                if isinstance(entry, dict)
            }

        persist_warnings = _persist_quote_cache_to_disk(
            self._resolve_store_path(),
            cache_snapshot,
        )
        return persist_warnings

    def _run_loop(self) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return

        while not stop_event.is_set():
            run = self._execute_iteration()
            should_stop = False
            with self._lock:
                self._history.append(run)
                self._iterations += 1
                self._last_run_completed_at = run.completed_at or run.started_at

                config = self._config
                max_iterations = config.max_iterations if config else None
                should_stop = max_iterations is not None and self._iterations >= max_iterations
                if run.status == "error":
                    self._status_message = "error"
                else:
                    self._status_message = "running"

            if should_stop:
                break

            config_snapshot = self._config
            interval_seconds = config_snapshot.interval_seconds if config_snapshot else 900
            if interval_seconds < 1:
                interval_seconds = 1
            if stop_event.wait(interval_seconds):
                break

        with self._lock:
            self._running = False
            self._status_message = "idle"
            self._thread = None
            self._stop_event = None

    def _synchronize_ibkr_state(
        self, settings: AppSettings
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        actions: List[Dict[str, Any]] = []
        warnings: List[str] = []

        cash_value: float | None
        cash_actions: List[Dict[str, Any]]
        cash_warnings: List[str]
        cash_value, cash_actions, cash_warnings = _fetch_ibkr_account_cash(
            settings=settings
        )
        actions.extend(cash_actions)
        warnings.extend(cash_warnings)

        if cash_value is not None:
            with self._lock:
                self._last_initial_cash = float(cash_value)
                if self._config is not None:
                    self._config.initial_cash = float(cash_value)
            persist_warnings = _persist_initial_cash_to_disk(
                self._resolve_store_path(), float(cash_value)
            )
            warnings.extend(persist_warnings)

        positions_snapshot, position_actions, position_warnings = _fetch_ibkr_positions(
            settings=settings
        )
        actions.extend(position_actions)
        warnings.extend(position_warnings)

        normalized_positions: Dict[str, Dict[str, Any]] = {}
        if isinstance(positions_snapshot, dict):
            for symbol, details in positions_snapshot.items():
                if isinstance(details, dict):
                    normalized_positions[str(symbol)] = dict(details)

        with self._lock:
            self._last_ibkr_positions = normalized_positions

        return actions, warnings

    def _execute_iteration(self) -> LiveTraderRun:
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        payload: Dict[str, Any] | None = None
        error: str | None = None
        status = "completed"

        try:
            settings = AppSettings()
            (
                request,
                extra_actions,
                extra_warnings,
                positions_snapshot,
                liquidity_snapshot,
            ) = self._build_iteration_request(settings)
            with self._lock:
                quote_cache_snapshot = {
                    symbol: dict(entry)
                    for symbol, entry in self._quote_cache.items()
                    if isinstance(entry, dict)
                }
            payload = _run_momentum_realtime_trade(
                request,
                settings=settings,
                positions_snapshot=positions_snapshot,
                liquidity_snapshot=liquidity_snapshot,
                extra_actions=extra_actions,
                extra_warnings=extra_warnings,
                previous_quotes=quote_cache_snapshot,
            )
            if isinstance(payload, dict):
                quote_snapshot = payload.get("quotes_snapshot")
                if isinstance(quote_snapshot, dict) and quote_snapshot:
                    quote_warnings = self._store_quote_snapshot(quote_snapshot)
                    if quote_warnings:
                        warnings_list = payload.setdefault("warnings", [])
                        warnings_list.extend(quote_warnings)
                trades = payload.get("paper_trades")
                actions_list = payload.setdefault("actions", [])
                warnings_list = payload.setdefault("warnings", [])

                should_execute_orders = bool(request.execute_orders)
                if should_execute_orders:
                    has_meaningful_trades = False
                    if isinstance(trades, list):
                        has_meaningful_trades = any(
                            int(trade.get("quantity", 0)) != 0
                            for trade in trades
                            if isinstance(trade, dict)
                        )
                    if has_meaningful_trades:
                        execution_actions, execution_warnings, execution_reports = _execute_ibkr_paper_orders(
                            trades,
                            settings=settings,
                        )
                        actions_list.extend(execution_actions)
                        warnings_list.extend(execution_warnings)
                        if execution_reports:
                            _apply_ibkr_execution_reports(
                                trades,
                                execution_reports,
                                actions=actions_list,
                                warnings=warnings_list,
                            )
                    else:
                        actions_list.append(
                            {
                                "type": "ibkr_execution_skipped",
                                "reason": "no_trades_today",
                            }
                        )

                payload.setdefault("initial_cash", float(request.initial_cash))
                payload.setdefault("execution_requested",
                                   should_execute_orders)
                payload.setdefault("live_run_id", run_id)
                payload = _summarize_live_payload(payload)
        except HTTPException as exc:
            status = "error"
            error = _extract_http_exception_message(exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            error = str(exc)

        completed_at = datetime.now(timezone.utc)
        run = LiveTraderRun(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            payload=payload,
            error=error,
        )

        self._record_live_run_results(run)
        return run

    def _record_live_run_results(self, run: LiveTraderRun) -> None:
        if run.status != "completed":
            return

        payload = run.payload
        if not isinstance(payload, dict):
            return

        trade_date_value: date | None = None
        trade_date_raw = payload.get("trade_date")
        if trade_date_raw:
            try:
                trade_date_value = pd.Timestamp(trade_date_raw).date()
            except Exception:
                trade_date_value = None

        trades_payload = payload.get("paper_trades")
        trades_list = trades_payload if isinstance(
            trades_payload, list) else []

        new_entries: List[Dict[str, Any]] = []
        for idx, trade in enumerate(trades_list):
            if not isinstance(trade, dict):
                continue

            timestamp_source = trade.get(
                "timestamp") or trade.get("paper_timestamp")
            if timestamp_source is None:
                continue

            timestamp_iso = _to_iso(timestamp_source)
            entry_date = _parse_trade_date(timestamp_iso)
            if trade_date_value and entry_date and entry_date != trade_date_value:
                continue

            quantity_raw = trade.get("execution_filled_quantity")
            if quantity_raw is None:
                quantity_raw = trade.get("quantity")
            try:
                quantity_value = float(quantity_raw)
            except (TypeError, ValueError):
                continue
            if abs(quantity_value) < 1e-9:
                continue

            symbol = str(trade.get("symbol", "")).strip().upper()
            if not symbol:
                continue

            direction = "buy" if quantity_value > 0 else "sell"
            quantity_abs = abs(quantity_value)
            rounded_quantity = round(quantity_abs)
            if abs(quantity_abs - rounded_quantity) < 1e-6:
                quantity_display: float | int = int(rounded_quantity)
            else:
                quantity_display = float(round(quantity_abs, 6))

            execution_price_raw = trade.get("execution_price")
            if execution_price_raw is None:
                execution_price_raw = trade.get("price")
            try:
                execution_price_value = float(execution_price_raw)
            except (TypeError, ValueError):
                execution_price_value = None

            paper_price_value = None
            try:
                paper_price_value = float(trade.get("paper_price"))
            except (TypeError, ValueError, AttributeError):
                paper_price_value = None

            cash_after_value = None
            try:
                cash_after_value = float(trade.get("cash_after"))
            except (TypeError, ValueError):
                cash_after_value = None

            signed_quantity_display = (
                int(round(quantity_value))
                if abs(quantity_value - round(quantity_value)) < 1e-6
                else float(round(quantity_value, 6))
            )

            trade_date_iso = None
            if entry_date:
                trade_date_iso = entry_date.isoformat()
            elif trade_date_value:
                trade_date_iso = trade_date_value.isoformat()

            entry: Dict[str, Any] = {
                "trade_id": f"{run.run_id}:{idx}",
                "run_id": run.run_id,
                "timestamp": timestamp_iso,
                "trade_date": trade_date_iso,
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity_display,
                "signed_quantity": signed_quantity_display,
            }

            if execution_price_value is not None:
                entry["price"] = execution_price_value
                entry["execution_price"] = execution_price_value
            if paper_price_value is not None:
                entry["paper_price"] = paper_price_value
            if cash_after_value is not None:
                entry["cash_after"] = cash_after_value

            notional_value = None
            if execution_price_value is not None:
                notional_value = execution_price_value * quantity_abs
            elif paper_price_value is not None:
                notional_value = paper_price_value * quantity_abs
            if notional_value is not None:
                entry["notional"] = float(round(notional_value, 4))

            quote_ts = trade.get("quote_timestamp") or trade.get("quote_time")
            if quote_ts:
                entry["quote_timestamp"] = _to_iso(quote_ts)

            status_text = trade.get("execution_status") or trade.get("status")
            if status_text:
                entry["status"] = str(status_text)

            new_entries.append(entry)

        equity_point: Dict[str, Any] | None = None
        liquidity_snapshot = payload.get("liquidity")
        if isinstance(liquidity_snapshot, dict):
            total_assets = liquidity_snapshot.get("total_assets")
            if total_assets is not None:
                equity_point = {
                    "run_id": run.run_id,
                    "timestamp": _to_iso(run.completed_at or datetime.now(timezone.utc)),
                    "total_assets": float(total_assets),
                }
                cash_available = liquidity_snapshot.get("cash_available")
                if cash_available is not None:
                    try:
                        equity_point["cash_available"] = float(cash_available)
                    except (TypeError, ValueError):
                        pass
                holdings_value = liquidity_snapshot.get("holdings_value")
                if holdings_value is not None:
                    try:
                        equity_point["holdings_value"] = float(holdings_value)
                    except (TypeError, ValueError):
                        pass
                position_count = liquidity_snapshot.get("position_count")
                if position_count is not None:
                    try:
                        equity_point["position_count"] = int(position_count)
                    except (TypeError, ValueError):
                        pass
                for key in (
                    "estimated_buy_value",
                    "estimated_sell_value",
                    "projected_cash_after_buys",
                    "projected_cash_after_sells",
                ):
                    value = liquidity_snapshot.get(key)
                    if value is not None:
                        try:
                            equity_point[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                initial_cash_value = payload.get("initial_cash")
                if initial_cash_value is not None:
                    try:
                        equity_point["initial_cash"] = float(
                            initial_cash_value)
                    except (TypeError, ValueError):
                        pass
                equity_trade_date = None
                if trade_date_value:
                    equity_trade_date = trade_date_value.isoformat()
                elif new_entries:
                    equity_trade_date = new_entries[-1].get("trade_date")
                if equity_trade_date:
                    equity_point["trade_date"] = equity_trade_date

        if not new_entries and equity_point is None:
            return

        should_persist = False
        trade_snapshot: List[Dict[str, Any]] = []
        equity_snapshot: List[Dict[str, Any]] = []

        with self._lock:
            existing_ids = {
                entry.get("trade_id")
                for entry in self._trade_history
                if isinstance(entry, dict) and isinstance(entry.get("trade_id"), str)
            }
            filtered_entries = [
                entry for entry in new_entries if entry.get("trade_id") not in existing_ids
            ]
            if filtered_entries:
                self._trade_history.extend(filtered_entries)
                self._trade_history.sort(
                    key=lambda item: str(item.get("timestamp") or ""))
                should_persist = True

            if equity_point is not None:
                if not any(point.get("run_id") == run.run_id for point in self._equity_curve):
                    self._equity_curve.append(equity_point)
                    self._equity_curve.sort(
                        key=lambda item: str(item.get("timestamp") or ""))
                    should_persist = True

            if not should_persist:
                # Nothing new to persist; still keep state loaded snapshot copies.
                trade_snapshot = [dict(entry) for entry in self._trade_history]
                equity_snapshot = [dict(point) for point in self._equity_curve]
                self._state_loaded = True
            else:
                trade_snapshot = [dict(entry) for entry in self._trade_history]
                equity_snapshot = [dict(point) for point in self._equity_curve]
                self._state_loaded = True

        if not should_persist:
            return

        persist_warnings = _persist_trade_history_to_disk(
            self._resolve_store_path(),
            trade_snapshot,
            equity_snapshot,
        )
        if persist_warnings:
            warnings_list = payload.setdefault("warnings", [])
            warnings_list.extend(persist_warnings)

    def _build_iteration_request(
        self, settings: AppSettings
    ) -> tuple[
        MomentumPaperTradeRequest,
        List[Dict[str, Any]],
        List[str],
        Dict[str, Dict[str, Any]],
        Dict[str, Any],
    ]:
        if self._config is None:
            raise RuntimeError("Momentum live trader is not configured.")

        training_window, paper_window = _default_paper_trade_windows(
            paper_days=self._config.paper_days,
            training_years=self._config.training_years,
        )

        initial_cash_value = self._config.initial_cash
        pre_actions: List[Dict[str, Any]] = []
        pre_warnings: List[str] = []

        cash_value, cash_actions, cash_warnings = _fetch_ibkr_account_cash(
            settings=settings)
        pre_actions.extend(cash_actions)
        pre_warnings.extend(cash_warnings)

        positions_snapshot, position_actions, position_warnings = _fetch_ibkr_positions(
            settings=settings
        )
        pre_actions.extend(position_actions)
        pre_warnings.extend(position_warnings)

        cash_connected = _ibkr_connection_success(
            cash_actions, purpose="cash_fetch")
        if cash_value is None or not cash_connected:
            diagnostics = _summarize_ibkr_cash_failure(
                pre_actions, pre_warnings)
            detail_parts = [diagnostics] if diagnostics else []
            detail_parts.extend(
                warning for warning in cash_warnings if warning)
            detail = " ".join(part for part in detail_parts if part)
            message = "Unable to connect to IBKR cash balance."
            if detail:
                message = f"{message} {detail}"
            raise RuntimeError(message)

        positions_connected = _ibkr_connection_success(
            position_actions, purpose="positions_fetch")
        if not positions_connected:
            detail_parts = [
                warning for warning in position_warnings if warning]
            detail = " ".join(part for part in detail_parts if part)
            message = "Unable to connect to IBKR open positions."
            if detail:
                message = f"{message} {detail}"
            raise RuntimeError(message)

        with self._lock:
            self._last_ibkr_positions = {
                str(symbol): dict(info)
                for symbol, info in positions_snapshot.items()
                if isinstance(info, dict)
            }

        initial_cash_value = float(cash_value)

        if initial_cash_value is None or float(initial_cash_value) <= 0:
            raise RuntimeError(
                "Unable to determine initial cash for momentum live iteration.")

        initial_cash_float = float(initial_cash_value)

        with self._lock:
            self._last_initial_cash = initial_cash_float
            if self._config is not None:
                self._config.initial_cash = initial_cash_float

        liquidity_snapshot = _calculate_portfolio_liquidity(
            initial_cash_float,
            positions_snapshot,
            pre_actions,
        )

        liquidity_action = {
            "type": "portfolio_liquidity_snapshot",
            "cash_available": liquidity_snapshot["cash_available"],
            "holdings_value": liquidity_snapshot["holdings_value"],
            "total_assets": liquidity_snapshot["total_assets"],
            "position_count": liquidity_snapshot.get("position_count", 0),
        }
        if "cash_breakdown" in liquidity_snapshot:
            liquidity_action["cash_breakdown"] = liquidity_snapshot["cash_breakdown"]
        pre_actions.append(liquidity_action)

        persist_warnings = _persist_initial_cash_to_disk(
            self._resolve_store_path(),
            initial_cash_float,
        )
        if persist_warnings:
            pre_warnings.extend(persist_warnings)

        payload = {
            "symbols": self._config.symbols,
            "initial_cash": initial_cash_float,
            "training_window": training_window,
            "paper_window": paper_window,
            "parameters": [self._config.parameters[0].model_dump()],
            "store_path": self._config.store_path,
            "bar_size": self._config.bar_size,
            "auto_fetch": self._config.auto_fetch,
            "execute_orders": self._config.execute_orders,
        }

        request_model = MomentumPaperTradeRequest.model_validate(payload)
        return (
            request_model,
            pre_actions,
            pre_warnings,
            positions_snapshot,
            liquidity_snapshot,
        )


def _extract_http_exception_message(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
        return json.dumps(detail)
    return str(detail)


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


def _load_snp100_paper_trade_symbols(settings: AppSettings) -> tuple[List[str], List[str], str]:
    symbols, warnings = _load_latest_snp100_symbols(settings)
    if symbols:
        source = "snp100_membership"
    else:
        source = str(settings.data_paths.cache / "snp100_universe.json")
    return symbols, warnings, source


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


def _default_paper_trade_windows(
    *,
    paper_days: int = DEFAULT_PAPER_TRADE_PAPER_DAYS,
    training_years: float = DEFAULT_PAPER_TRADE_TRAINING_YEARS,
) -> tuple[tuple[date, date], tuple[date, date]]:
    today = date.today()
    normalized_paper_days = max(int(paper_days), 60)
    paper_end = today
    paper_start = paper_end - timedelta(days=normalized_paper_days)
    if paper_start >= paper_end:
        paper_start = paper_end - timedelta(days=60)

    normalized_training_days = max(int(training_years * 365), 120)
    training_end = paper_start - timedelta(days=1)
    training_start = training_end - timedelta(days=normalized_training_days)
    if training_start >= training_end:
        training_start = training_end - \
            timedelta(days=normalized_training_days or 120)

    return (training_start, training_end), (paper_start, paper_end)


def _normalize_symbol_list(symbols: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        candidate = str(symbol).strip().upper()
        if candidate and candidate not in seen:
            cleaned.append(candidate)
            seen.add(candidate)
    return cleaned


def _apply_ibkr_position_constraints(
    trades: Sequence[Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]] | None,
    *,
    actions: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    if not trades or not positions or not isinstance(positions, dict):
        return

    blocked_symbols: List[str] = []
    for entry in trades:
        if not isinstance(entry, dict):
            continue
        try:
            quantity = int(entry.get("quantity", 0))
        except (TypeError, ValueError):
            quantity = 0
        if quantity <= 0:
            continue
        symbol = str(entry.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        position_info = positions.get(symbol)
        if not isinstance(position_info, dict):
            continue
        try:
            existing_quantity = int(position_info.get("quantity", 0))
        except (TypeError, ValueError):
            existing_quantity = None
        if existing_quantity is None or existing_quantity <= 0:
            continue

        entry["blocked"] = True
        entry["blocked_reason"] = "existing_position"
        entry["original_quantity"] = quantity
        entry["quantity"] = 0
        blocked_symbols.append(symbol)
        actions.append(
            {
                "type": "ibkr_position_blocked",
                "symbol": symbol,
                "existing_quantity": existing_quantity,
                "blocked_quantity": quantity,
                "reason": "existing_position",
            }
        )

    if blocked_symbols:
        unique_symbols = sorted(
            {symbol for symbol in blocked_symbols if symbol})
        warnings.append(
            "Skipped buy orders for existing IBKR holdings: "
            + ", ".join(unique_symbols)
        )


def _apply_ibkr_execution_reports(
    trades: Sequence[Dict[str, Any]],
    reports: Sequence[Dict[str, Any]],
    *,
    actions: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    for report in reports:
        index = report.get("input_index")
        if not isinstance(index, int) or index < 0 or index >= len(trades):
            continue

        entry = trades[index]
        if not isinstance(entry, dict):
            continue

        symbol = entry.get("symbol") or report.get("symbol") or "symbol"
        updated = False
        adjustment: Dict[str, Any] = {
            "type": "ibkr_execution_recorded",
            "input_index": index,
            "symbol": symbol,
        }

        avg_price = report.get("avg_price")
        if avg_price is not None:
            original_price = entry.get("price")
            if original_price is not None and "paper_price" not in entry:
                entry["paper_price"] = float(original_price)
            entry["price"] = float(avg_price)
            entry["execution_price"] = float(avg_price)
            adjustment["execution_price"] = float(avg_price)
            if original_price is not None:
                adjustment["previous_price"] = float(original_price)
                if abs(float(avg_price) - float(original_price)) > 0.01:
                    warnings.append(
                        f"IBKR fill price for {symbol} ({float(avg_price):.2f}) differs from simulated price "
                        f"({float(original_price):.2f})."
                    )
            updated = True

        filled_quantity = report.get("filled_quantity")
        if filled_quantity is not None:
            entry["execution_filled_quantity"] = int(filled_quantity)
            adjustment["execution_filled_quantity"] = int(filled_quantity)
            updated = True

        status = report.get("status")
        if status:
            adjustment["status"] = str(status)

        if updated:
            actions.append(adjustment)


def _execute_ibkr_paper_orders(
    trades: Sequence[Dict[str, Any]],
    *,
    settings: AppSettings,
) -> tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    execution_reports: List[Dict[str, Any]] = []

    meaningful_trades = [
        (idx, trade)
        for idx, trade in enumerate(trades)
        if int(trade.get("quantity", 0)) != 0
    ]
    if not meaningful_trades:
        actions.append({
            "type": "ibkr_execution_skipped",
            "reason": "no_trades",
        })
        return actions, warnings, execution_reports

    try:
        from ib_insync import IB, MarketOrder
        from ib_insync.contract import Contract
    except ImportError:  # pragma: no cover - runtime guard
        warnings.append(
            "ib-insync is not available; cannot submit orders to IBKR."
        )
        actions.append({
            "type": "ibkr_execution_skipped",
            "reason": "missing_dependency",
        })
        return actions, warnings, execution_reports

    ib: Any | None = None
    loop: asyncio.AbstractEventLoop | None = None
    previous_loop: asyncio.AbstractEventLoop | None = None
    loop_created = False
    connect_action = {
        "type": "ibkr_connect",
        "host": settings.ibkr_host,
        "port": settings.ibkr_port,
        "client_id": settings.ibkr_client_id,
        "status": "pending",
    }
    actions.append(connect_action)

    try:
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        if previous_loop is None or previous_loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_created = True
        else:
            loop = previous_loop

        ib = IB()
        ib.connect(settings.ibkr_host, settings.ibkr_port,
                   clientId=settings.ibkr_client_id)
        if not ib.isConnected():
            raise RuntimeError("Failed to establish IBKR connection")
        connect_action["status"] = "connected"
        if loop is not None:
            loop.run_until_complete(asyncio.sleep(0))

        submitted_trades: list[tuple[int, Dict[str, Any], Any]] = []
        for input_index, trade in meaningful_trades:
            quantity = int(trade.get("quantity", 0))
            direction = "BUY" if quantity > 0 else "SELL"
            contract = Contract(
                symbol=str(trade.get("symbol", "")),
                secType="STK",
                currency="USD",
                exchange="SMART",
            )
            ib.qualifyContracts(contract)
            order = MarketOrder(direction, abs(quantity))
            ib_trade = ib.placeOrder(contract, order)
            submitted_trades.append((input_index, trade, ib_trade))
            actions.append(
                {
                    "type": "ibkr_order_submitted",
                    "symbol": contract.symbol,
                    "direction": direction.lower(),
                    "quantity": abs(quantity),
                    "timestamp": trade.get("timestamp"),
                    "price": trade.get("price"),
                    "order_id": getattr(getattr(ib_trade, "order", None), "orderId", None),
                }
            )
            if loop is not None:
                loop.run_until_complete(asyncio.sleep(0.1))

        try:
            ib.waitOnUpdate(timeout=2)
        except Exception:  # pragma: no cover - best effort to flush submissions
            pass

        for input_index, original_trade, ib_trade in submitted_trades:
            order = getattr(ib_trade, "order", None)
            order_status = getattr(ib_trade, "orderStatus", None)
            status_text = str(getattr(order_status, "status", "")).lower()
            fills = list(getattr(ib_trade, "fills", []) or [])
            filled = bool(fills) or status_text == "filled"
            order_id = getattr(order, "orderId", None) if order else None
            symbol = getattr(order, "symbol", "") if order else ""
            requested_quantity = int(original_trade.get("quantity", 0))
            direction_sign = 1 if requested_quantity > 0 else -1

            total_shares = 0.0
            total_cost = 0.0
            for fill in fills:
                execution = getattr(fill, "execution", None)
                if execution is None:
                    continue
                shares = getattr(execution, "shares", 0)
                price = getattr(execution, "price", None)
                if not shares:
                    continue
                fill_shares = abs(float(shares))
                total_shares += fill_shares
                if price is not None:
                    total_cost += fill_shares * float(price)

            avg_price = total_cost / total_shares if total_shares else None
            filled_quantity = int(
                total_shares * direction_sign) if total_shares else None

            actions.append(
                {
                    "type": "ibkr_order_status",
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": status_text or "unknown",
                    "filled": filled,
                    "fill_count": len(fills),
                    "avg_price": float(avg_price) if avg_price is not None else None,
                    "filled_quantity": filled_quantity,
                    "input_index": input_index,
                }
            )

            report: Dict[str, Any] = {
                "input_index": input_index,
                "symbol": symbol,
                "status": status_text or "unknown",
            }
            if avg_price is not None:
                report["avg_price"] = float(avg_price)
            if filled_quantity is not None:
                report["filled_quantity"] = int(filled_quantity)
            execution_reports.append(report)

            if not filled:
                warnings.append(
                    f"IBKR order {order_id or '?'} for {symbol or 'symbol'} not filled (status: {status_text or 'unknown'})."
                )

        return actions, warnings, execution_reports
    except Exception as exc:  # pragma: no cover - defensive guard
        if connect_action.get("status") == "pending":
            connect_action["status"] = "failed"
        warnings.append(f"IBKR execution failed: {exc}")
        actions.append({"type": "ibkr_execution_failed", "message": str(exc)})
        return actions, warnings, execution_reports
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            actions.append({"type": "ibkr_disconnect",
                           "status": "disconnected"})
        if loop_created and loop is not None:
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:  # pragma: no cover - defensive flush
                pass
            finally:
                asyncio.set_event_loop(previous_loop)
                loop.close()


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
