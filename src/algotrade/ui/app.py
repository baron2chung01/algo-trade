"""FastAPI application exposing a candlestick UI for mean reversion backtests."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
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
from ..experiments import OptimizationOutcome, optimize_mean_reversion_parameters
from ..experiments.mean_reversion import HoldRange, OptimizationParameterSpec, ParameterRange

DEFAULT_STORE_PATH = Path(
    os.getenv("ALGO_TRADE_STORE_PATH", "data/raw/polygon/daily"))
DEFAULT_BAR_SIZE = "1d"
DEFAULT_SYMBOLS = ("AAPL", "MSFT")


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


class OptimizationParameterSpecRequest(BaseModel):
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


class OptimizationRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    initial_cash: float = Field(100_000.0, gt=0)
    limit: int = Field(250, ge=0)
    bar_size: str = Field(DEFAULT_BAR_SIZE, description="Historical bar size")
    store_path: str | None = Field(default=None)
    paper_days: int = Field(365, ge=60)
    training_years: float = Field(2.0, gt=0)
    auto_fetch: bool = Field(
        False, description="Download missing Polygon data on demand")
    parameter_spec: OptimizationParameterSpecRequest | None = None
    include_warnings: bool = True

    model_config = ConfigDict(extra="forbid")

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

        return self


def create_app(templates_dir: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    root = Path(__file__).resolve().parent
    templates_path = templates_dir or (root / "templates")
    static_path = static_dir or (root / "static")

    app = FastAPI(title="Mean Reversion Strategy Viewer", version="0.1.0")
    templates = Jinja2Templates(directory=str(templates_path))

    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)),
                  name="static")

    @app.get("/", response_class=HTMLResponse)
    async def render_index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

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

        parameter_spec = (
            request_body.parameter_spec.to_spec()
            if request_body.parameter_spec
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
                optimization = optimize_mean_reversion_parameters(
                    store_path=store_root,
                    universe=[symbol],
                    initial_cash=request_body.initial_cash,
                    training_window=training_window,
                    paper_window=paper_window,
                    bar_size=request_body.bar_size,
                    parameter_spec=parameter_spec,
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
    paper_days: int = 365,
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


def _serialize_optimization_summary(outcome: OptimizationOutcome) -> Dict[str, Any]:
    best = outcome.best_parameters
    rankings = (
        outcome.parameter_frame.sort_values(
            ["cagr", "final_equity"], ascending=[False, False])
        .head(10)
        .to_dict(orient="records")
    )
    for row in rankings:
        for key, value in list(row.items()):
            if key in {"split", "params"}:
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    row[key] = None

    best_parameters = {
        "entry_threshold": best.entry_threshold,
        "exit_threshold": best.exit_threshold,
        "max_hold_days": best.max_hold_days,
        "target_position_pct": best.target_position_pct,
        "stop_loss_pct": best.stop_loss_pct,
        "lot_size": best.lot_size,
    }

    return {
        "best_parameters": best_parameters,
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
