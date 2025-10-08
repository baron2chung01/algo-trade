from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from algotrade.data.stores.local import ParquetBarStore
from algotrade.experiments.vcp import scan_vcp_candidates


def _build_daily_frame(
    *,
    start_price: float,
    daily_slope: float,
    start_date: datetime,
    sessions: int,
    volume_start: float = 600_000.0,
    volume_end: float = 200_000.0,
) -> pd.DataFrame:
    dates = pd.date_range(
        start=start_date, periods=sessions, freq="B", tz=timezone.utc)
    closes = start_price + daily_slope * np.arange(sessions, dtype=float)
    highs = np.empty_like(closes)
    lows = np.empty_like(closes)

    for idx in range(sessions):
        if idx >= sessions - 5:
            high_factor = 1.01
            low_factor = 0.995
        elif idx >= sessions - 25:
            high_factor = 1.05
            low_factor = 0.98
        elif idx >= sessions - 65:
            high_factor = 1.10
            low_factor = 0.93
        else:
            high_factor = 1.12
            low_factor = 0.90
        highs[idx] = closes[idx] * high_factor
        lows[idx] = closes[idx] * low_factor

    opens = closes * 0.995
    volumes = np.linspace(volume_start, volume_end, sessions)
    averages = (highs + lows + closes) / 3
    bar_count = np.full(sessions, 100)

    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "average": averages,
            "bar_count": bar_count,
        }
    )
    return frame


def test_vcp_scan_applies_new_rule_set(tmp_path):
    store = ParquetBarStore(tmp_path)
    sessions = 300
    start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

    aapl_frame = _build_daily_frame(
        start_price=100.0,
        daily_slope=0.22,
        start_date=start_date,
        sessions=sessions,
        volume_start=900_000.0,
        volume_end=300_000.0,
    )
    spy_frame = _build_daily_frame(
        start_price=100.0,
        daily_slope=0.10,
        start_date=start_date,
        sessions=sessions,
    )

    store.save("AAPL", "1d", aapl_frame)
    store.save("SPY", "1d", spy_frame)

    summary = scan_vcp_candidates(
        store_path=tmp_path,
        timeframe="medium",
        bar_size="1d",
        symbols=["AAPL"],
        max_candidates=10,
    )

    assert summary.parameters.rule_set == (
        "Liquidity Filter + Uptrend Nearing Breakout + Higher Lows + Volume Contracting"
    )
    assert summary.parameters.criteria == (
        "liquidity",
        "uptrend_breakout",
        "higher_lows",
        "volume_contraction",
    )
    assert summary.symbols_scanned == 1
    assert summary.analysis_timestamp is not None
    assert not summary.warnings

    assert len(summary.candidates) == 1
    candidate = summary.candidates[0]
    assert candidate.symbol == "AAPL"
    assert candidate.liquidity_pass
    assert candidate.market_cap_pass
    assert candidate.close_above_sma20
    assert candidate.rs_percentile_pass
    assert candidate.uptrend_breakout_pass
    assert candidate.higher_lows_pass
    assert candidate.volume_contraction_pass
    assert candidate.market_cap is not None and candidate.market_cap > 2_000_000_000
    assert candidate.monthly_dollar_volume is not None and candidate.monthly_dollar_volume > 1_500_000
    assert candidate.daily_breakout_distance_pct is not None
    assert candidate.weekly_breakout_distance_pct is not None
    assert candidate.rs_percentile is not None and candidate.rs_percentile > 70.0


def test_vcp_scan_respects_selected_criteria(tmp_path):
    store = ParquetBarStore(tmp_path)
    sessions = 300
    start_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

    strong_frame = _build_daily_frame(
        start_price=120.0,
        daily_slope=0.18,
        start_date=start_date,
        sessions=sessions,
        volume_start=250_000.0,
        volume_end=80_000.0,
    )
    weak_flag_frame = strong_frame.copy()

    spy_frame = _build_daily_frame(
        start_price=100.0,
        daily_slope=0.10,
        start_date=start_date,
        sessions=sessions,
    )

    store.save("BETA", "1d", weak_flag_frame)
    store.save("SPY", "1d", spy_frame)

    summary = scan_vcp_candidates(
        store_path=tmp_path,
        timeframe="medium",
        bar_size="1d",
        symbols=["BETA"],
        max_candidates=5,
        criteria=["uptrend_breakout", "higher_lows", "volume_contraction"],
    )

    assert summary.parameters.criteria == (
        "uptrend_breakout",
        "higher_lows",
        "volume_contraction",
    )
    assert summary.parameters.rule_set == "Uptrend Nearing Breakout + Higher Lows + Volume Contracting"
    assert len(summary.candidates) == 1

    candidate = summary.candidates[0]
    assert candidate.symbol == "BETA"
    assert not candidate.liquidity_pass
    assert not candidate.market_cap_pass
    assert candidate.uptrend_breakout_pass
    assert candidate.higher_lows_pass
    assert candidate.volume_contraction_pass
    assert candidate.rs_percentile is not None and candidate.rs_percentile > 70.0

    strict_summary = scan_vcp_candidates(
        store_path=tmp_path,
        timeframe="medium",
        bar_size="1d",
        symbols=["BETA"],
        max_candidates=5,
        criteria=["liquidity", "uptrend_breakout",
                  "higher_lows", "volume_contraction"],
    )

    assert strict_summary.parameters.criteria == (
        "liquidity",
        "uptrend_breakout",
        "higher_lows",
        "volume_contraction",
    )
    assert len(strict_summary.candidates) == 0
