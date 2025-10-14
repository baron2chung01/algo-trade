from datetime import date, datetime, timedelta, timezone

import pytest
import pandas as pd

from src.algotrade.ui.app import (
    _apply_ibkr_execution_reports,
    _apply_ibkr_position_constraints,
    _calculate_portfolio_liquidity,
    _compute_realtime_momentum_scores,
    _run_momentum_realtime_trade,
    _default_paper_trade_windows,
    _extract_cash_breakdown_from_actions,
    _filter_trades_for_trade_date,
    _project_live_trade_plan,
)
from src.algotrade.ui.app import MomentumPaperTradeRequest, MomentumParameterRequest
from src.algotrade.config import AppSettings
from src.algotrade.data.stores.local import ParquetBarStore


def test_filter_trades_for_trade_date_filters_only_target_day():
    trades = [
        {"symbol": "AAPL", "timestamp": "2024-01-01T15:30:00Z", "quantity": 10},
        {"symbol": "MSFT", "timestamp": "2024-01-02T15:30:00Z", "quantity": 5},
        {"symbol": "TSLA", "timestamp": "2024-01-02T16:00:00Z", "quantity": 7},
    ]

    filtered, skipped = _filter_trades_for_trade_date(trades, date(2024, 1, 2))

    assert [trade["symbol"] for trade in filtered] == ["MSFT", "TSLA"]
    assert [trade["symbol"] for trade in skipped] == ["AAPL"]
    # ensure original trades untouched
    assert trades[0]["quantity"] == 10


def test_filter_trades_for_trade_date_returns_empty_when_no_matches():
    trades = [
        {"symbol": "AAPL", "timestamp": "2024-01-01T15:30:00Z", "quantity": 10}
    ]

    filtered, skipped = _filter_trades_for_trade_date(trades, date(2024, 1, 2))

    assert filtered == []
    assert [trade["symbol"] for trade in skipped] == ["AAPL"]


@pytest.mark.parametrize(
    "quantity",
    [5, 12],
)
def test_apply_ibkr_position_constraints_blocks_existing_positions(quantity: int):
    trades = [
        {"symbol": "AAPL", "quantity": quantity},
        {"symbol": "GOOG", "quantity": -3},
    ]
    positions = {"AAPL": {"quantity": 100}}
    actions: list[dict] = []
    warnings: list[str] = []

    _apply_ibkr_position_constraints(
        trades, positions, actions=actions, warnings=warnings)

    blocked_trade = trades[0]
    assert blocked_trade["blocked"] is True
    assert blocked_trade["quantity"] == 0
    assert blocked_trade["blocked_reason"] == "existing_position"
    assert blocked_trade["original_quantity"] == quantity

    assert trades[1]["quantity"] == -3  # unaffected short sells

    assert any(action["type"] == "ibkr_position_blocked" for action in actions)
    assert any("AAPL" in warning for warning in warnings)


def test_calculate_portfolio_liquidity_computes_values():
    positions = {
        "AAPL": {"quantity": 10, "market_price": 150.0},
        "MSFT": {"quantity": 5, "avg_cost": 200.0},
        "TSLA": {"quantity": 0, "market_price": 250.0},
    }
    actions = [
        {
            "type": "ibkr_cash_snapshot",
            "candidates": {
                "USD": {"value": 12345.0},
                "EUR": {"value": 500.0},
            },
        }
    ]

    result = _calculate_portfolio_liquidity(5000.0, positions, actions)

    assert result["cash_available"] == pytest.approx(5000.0)
    expected_holdings = 10 * 150.0 + 5 * 200.0
    assert result["holdings_value"] == pytest.approx(expected_holdings)
    assert result["total_assets"] == pytest.approx(5000.0 + expected_holdings)
    assert result["position_count"] == 2
    assert result["cash_breakdown"]["USD"] == pytest.approx(12345.0)
    assert result["cash_breakdown"]["EUR"] == pytest.approx(500.0)


def test_project_live_trade_plan_orders_sells_before_buys():
    trades = [
        {"symbol": "AAPL", "quantity": 4, "price": 100.0},
        {"symbol": "MSFT", "quantity": -2, "price": 250.0},
        {"symbol": "GOOG", "quantity": 3, "price": 150.0},
    ]
    actions: list[dict] = []
    liquidity = {"cash_available": 10000.0}

    _project_live_trade_plan(
        trades,
        initial_cash=10000.0,
        actions=actions,
        liquidity=liquidity,
        buy_universe_label="S&P 100",
    )

    assert [trade["symbol"] for trade in trades] == ["MSFT", "AAPL", "GOOG"]
    assert liquidity["estimated_sell_value"] == pytest.approx(2 * 250.0)
    assert liquidity["projected_cash_after_sells"] == pytest.approx(
        10000.0 + 2 * 250.0)
    assert actions[-1]["type"] == "momentum_live_sequence"
    assert actions[-1]["sell_symbols"] == ["MSFT"]
    assert actions[-1]["buy_symbols"] == ["AAPL", "GOOG"]


def test_extract_cash_breakdown_from_actions_returns_latest_snapshot():
    actions = [
        {"type": "ibkr_cash_snapshot", "candidates": {"USD": {"value": 5000}}},
        {"type": "unrelated"},
        {
            "type": "ibkr_cash_snapshot",
            "candidates": {
                "USD": {"value": 12345.0},
                "EUR": {"value": 250.5},
            },
        },
    ]

    breakdown = _extract_cash_breakdown_from_actions(actions)

    assert breakdown == {"USD": pytest.approx(
        12345.0), "EUR": pytest.approx(250.5)}


def test_apply_ibkr_execution_reports_updates_trades_and_actions():
    trades = [{"symbol": "AAPL", "quantity": 5, "price": 100.0}]
    reports = [
        {
            "input_index": 0,
            "symbol": "AAPL",
            "avg_price": 101.0,
            "filled_quantity": 5,
            "status": "filled",
        }
    ]
    actions: list[dict] = []
    warnings: list[str] = []

    _apply_ibkr_execution_reports(
        trades, reports, actions=actions, warnings=warnings)

    updated_trade = trades[0]
    assert updated_trade["price"] == pytest.approx(101.0)
    assert updated_trade["paper_price"] == pytest.approx(100.0)
    assert updated_trade["execution_filled_quantity"] == 5
    assert actions and actions[-1]["type"] == "ibkr_execution_recorded"
    assert any("AAPL" in warning for warning in warnings)


def test_default_paper_trade_windows_returns_chronological_periods():
    training_window, paper_window = _default_paper_trade_windows(
        paper_days=90,
        training_years=1.5,
    )

    training_start, training_end = training_window
    paper_start, paper_end = paper_window

    assert training_start < training_end < paper_start <= paper_end
    assert (paper_end - paper_start).days >= 60
    assert (training_end - training_start).days >= 120


def test_compute_realtime_momentum_scores_uses_current_price(tmp_path):
    store = ParquetBarStore(tmp_path)
    timestamps = [datetime.now(timezone.utc) -
                  timedelta(days=130 - idx) for idx in range(131)]
    rows = []
    for idx, ts in enumerate(timestamps[:-1]):
        price = 50.0 + idx * 0.2
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1_000.0,
                "average": price,
                "bar_count": 1,
            }
        )
    frame = pd.DataFrame(rows)
    store.save("TEST", "1d", frame)

    params = MomentumParameterRequest(
        lookback_days=20,
        skip_days=0,
        rebalance_days=1,
        max_positions=1,
        lot_size=1,
        cash_reserve_pct=0.0,
        min_momentum=-1.0,
        volatility_window=0,
        volatility_exponent=0.0,
    ).to_parameters()

    final_price = 150.0
    quotes = {
        "TEST": {
            "price": final_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        }
    }

    scored, scores_map, warnings = _compute_realtime_momentum_scores(
        ["TEST"],
        store=store,
        bar_size="1d",
        quotes=quotes,
        parameters=params,
    )

    assert warnings == []
    assert scored and scored[0][0] == "TEST"
    expected_momentum = (final_price / rows[-20]["close"]) - 1.0
    assert pytest.approx(scores_map["TEST"], rel=1e-6) == expected_momentum


def test_run_momentum_realtime_trade_generates_actions(monkeypatch, tmp_path):
    quotes = {
        "AAPL": {
            "price": 150.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        },
        "MSFT": {
            "price": 200.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
        },
    }

    scored = [("MSFT", 0.5), ("AAPL", 0.1)]
    scores_map = {"MSFT": 0.5, "AAPL": 0.1}

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_polygon_live_quotes",
        lambda symbols, *, settings: (quotes, [], []),
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._compute_realtime_momentum_scores",
        lambda symbols, **kwargs: (scored, scores_map, []),
    )

    request = MomentumPaperTradeRequest(
        symbols=["AAPL", "MSFT"],
        initial_cash=100_000.0,
        parameters=[
            MomentumParameterRequest(
                lookback_days=20,
                skip_days=0,
                rebalance_days=1,
                max_positions=1,
                lot_size=1,
                cash_reserve_pct=0.0,
            )
        ],
        store_path=str(tmp_path),
        execute_orders=False,
    )

    settings = AppSettings()
    positions_snapshot = {"AAPL": {"quantity": 10, "market_price": 150.0}}
    liquidity_snapshot = {
        "cash_available": 100_000.0,
        "holdings_value": 1_500.0,
        "total_assets": 101_500.0,
    }

    payload = _run_momentum_realtime_trade(
        request,
        settings=settings,
        positions_snapshot=positions_snapshot,
        liquidity_snapshot=liquidity_snapshot,
        extra_actions=[],
        extra_warnings=[],
    )

    trades = payload["paper_trades"]
    assert any(trade["symbol"] == "AAPL" and trade["quantity"]
               < 0 for trade in trades)
    assert any(trade["symbol"] == "MSFT" and trade["quantity"]
               > 0 for trade in trades)
    assert payload["trade_date"] == date.today().isoformat()


def test_run_momentum_realtime_trade_uses_cached_quotes(monkeypatch, tmp_path):
    previous_quotes = {
        "AAPL": {
            "price": 150.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "cached_quote",
        }
    }

    def fake_polygon_fetch(symbols, *, settings):  # noqa: D401
        return {}, [], []

    def fake_ibkr_fetch(symbols, *, settings):  # noqa: D401
        return {}, [], []

    def fake_compute(symbols, *, quotes, **kwargs):  # noqa: D401
        assert "AAPL" in quotes
        assert quotes["AAPL"]["price"] == pytest.approx(150.0)
        assert quotes["AAPL"]["source"] == "cached_quote"
        return [("AAPL", 0.1)], {"AAPL": 0.1}, []

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_polygon_live_quotes",
        fake_polygon_fetch,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_live_quotes",
        fake_ibkr_fetch,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._compute_realtime_momentum_scores",
        fake_compute,
    )

    request = MomentumPaperTradeRequest(
        symbols=["AAPL"],
        initial_cash=100_000.0,
        parameters=[MomentumParameterRequest()],
        store_path=str(tmp_path),
        execute_orders=False,
    )

    positions_snapshot = {"AAPL": {"quantity": 5, "market_price": 150.0}}
    liquidity_snapshot = {
        "cash_available": 100_000.0,
        "holdings_value": 750.0,
        "total_assets": 100_750.0,
    }

    payload = _run_momentum_realtime_trade(
        request,
        settings=AppSettings(),
        positions_snapshot=positions_snapshot,
        liquidity_snapshot=liquidity_snapshot,
        extra_actions=[],
        extra_warnings=[],
        previous_quotes=previous_quotes,
    )

    warnings = payload.get("warnings", [])
    assert any("cached quote" in message.lower() for message in warnings)

    actions = payload.get("actions", [])
    assert any(action.get("type") == "quote_cache_fallback"
               for action in actions)

    quotes_snapshot = payload.get("quotes_snapshot", {})
    assert "AAPL" in quotes_snapshot
    assert quotes_snapshot["AAPL"]["price"] == pytest.approx(150.0)
    assert quotes_snapshot["AAPL"]["source"] == "cached_quote"
