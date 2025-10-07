"""FastAPI application exposing a candlestick UI for strategy backtests."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import AppSettings
from ..data.stores.local import ParquetBarStore
from ..data.ingest import ingest_polygon_daily
from ..experiments import (
    BreakoutParameterSpec,
    FloatRange,
    optimize_breakout_parameters,
    optimize_mean_reversion_parameters,
)
from ..experiments.mean_reversion import HoldRange, OptimizationParameterSpec, ParameterRange
from ..strategies import BreakoutPattern

DEFAULT_STORE_PATH = Path(
    os.getenv("ALGO_TRADE_STORE_PATH", "data/raw/polygon/daily"))
DEFAULT_BAR_SIZE = "1d"
DEFAULT_SYMBOLS = ("AAPL", "MSFT")


class StrategyName(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


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

        mean_reversion_spec = (
            request_body.resolve_mean_reversion_spec()
            if request_body.strategy == StrategyName.MEAN_REVERSION
            else None
        )
        breakout_spec = (
            request_body.resolve_breakout_spec()
            if request_body.strategy == StrategyName.BREAKOUT
            else None
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
                    optimization = optimize_mean_reversion_parameters(
                        store_path=store_root,
                        universe=[symbol],
                        initial_cash=request_body.initial_cash,
                        training_window=training_window,
                        paper_window=paper_window,
                        bar_size=request_body.bar_size,
                        parameter_spec=mean_reversion_spec,
                    )
                else:
                    optimization = optimize_breakout_parameters(
                        store_path=store_root,
                        universe=[symbol],
                        initial_cash=request_body.initial_cash,
                        training_window=training_window,
                        paper_window=paper_window,
                        bar_size=request_body.bar_size,
                        parameter_spec=breakout_spec,
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

            results[symbol] = {
                "candles": candles,
                "buy_signals": buy_signals,
                "equity_curve": equity_curve,
                "metrics": metrics,
                "optimization": _serialize_optimization_summary(optimization),
                "strategy": request_body.strategy.value,
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

    return app


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
