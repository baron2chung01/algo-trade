import json
from datetime import date, datetime, timezone
from typing import Any, Dict

import pytest

from src.algotrade.config import AppSettings
from src.algotrade.ui.app import (
    LiveTraderRun,
    MomentumLiveTradeRequest,
    MomentumLiveTrader,
    MomentumPaperTradeRequest,
    MomentumParameterRequest,
)


def _configure_trader(tmp_path) -> MomentumLiveTrader:
    trader = MomentumLiveTrader()
    trader._store_path = tmp_path  # type: ignore[attr-defined]
    trader._config = MomentumLiveTradeRequest(  # type: ignore[attr-defined]
        symbols=["AAPL"],
        initial_cash=1000.0,
        parameters=[MomentumParameterRequest()],
        store_path=str(tmp_path),
        execute_orders=False,
        auto_fetch=False,
    )
    return trader


def test_start_synchronizes_ibkr_state(monkeypatch, tmp_path):
    cash_actions = [{"type": "ibkr_cash_snapshot",
                     "status": "ok", "value": 7500.0}]
    cash_warnings = ["cash warning"]
    position_actions = [{"type": "ibkr_positions_snapshot", "status": "ok"}]
    position_warnings = ["position warning"]
    positions_snapshot = {"MSFT": {"quantity": 5, "market_price": 320.0}}

    def fake_fetch_cash(*, settings):  # noqa: D401
        return 7500.0, list(cash_actions), list(cash_warnings)

    def fake_fetch_positions(*, settings):  # noqa: D401
        return dict(positions_snapshot), list(position_actions), list(position_warnings)

    persist_calls: list[tuple[object, float]] = []

    def fake_persist(store_path, value):  # noqa: D401
        persist_calls.append((store_path, value))
        return ["persist warning"]

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_account_cash",
        fake_fetch_cash,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_positions",
        fake_fetch_positions,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._persist_initial_cash_to_disk",
        fake_persist,
    )
    monkeypatch.setattr(MomentumLiveTrader, "_run_loop", lambda self: None)

    trader = MomentumLiveTrader()
    request = MomentumLiveTradeRequest(
        symbols=["AAPL"],
        initial_cash=None,
        parameters=[MomentumParameterRequest()],
        store_path=str(tmp_path),
        auto_fetch=False,
        execute_orders=False,
    )

    status = trader.start(request)

    assert trader._last_initial_cash == pytest.approx(
        7500.0)  # type: ignore[attr-defined]
    # type: ignore[attr-defined]
    assert trader._last_ibkr_positions == positions_snapshot
    assert persist_calls and persist_calls[0][1] == pytest.approx(7500.0)

    assert status["config"]["initial_cash"] == pytest.approx(7500.0)
    assert status["last_ibkr_position_count"] == 1
    assert status["last_ibkr_positions"]["MSFT"]["quantity"] == 5

    startup_warnings = status.get("startup_warnings", [])
    assert "cash warning" in startup_warnings
    assert "position warning" in startup_warnings
    assert "persist warning" in startup_warnings

    startup_actions = status.get("startup_actions", [])
    assert any(action.get("type") ==
               "ibkr_cash_snapshot" for action in startup_actions)
    assert any(action.get("type") ==
               "ibkr_positions_snapshot" for action in startup_actions)

    trader.stop()


def test_start_uses_persisted_initial_cash(monkeypatch, tmp_path):
    state_path = tmp_path / "momentum_live_state.json"
    state_path.write_text(
        json.dumps({"last_initial_cash": 4321.0}), encoding="utf-8"
    )

    cash_actions = [
        {"type": "ibkr_cash_snapshot", "status": "ok", "value": 9100.0}
    ]
    position_actions = [
        {"type": "ibkr_positions_snapshot", "status": "ok"}
    ]

    def fake_fetch_cash(*, settings):  # noqa: D401
        return 9100.0, list(cash_actions), []

    def fake_fetch_positions(*, settings):  # noqa: D401
        return {}, list(position_actions), []

    persisted_values: list[tuple[object, float]] = []

    def fake_persist(store_path, value):  # noqa: D401
        persisted_values.append((store_path, value))
        return []

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_account_cash",
        fake_fetch_cash,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_positions",
        fake_fetch_positions,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._persist_initial_cash_to_disk",
        fake_persist,
    )
    monkeypatch.setattr(MomentumLiveTrader, "_run_loop", lambda self: None)

    trader = MomentumLiveTrader()
    request = MomentumLiveTradeRequest(
        symbols=["AAPL"],
        initial_cash=None,
        parameters=[MomentumParameterRequest()],
        store_path=str(tmp_path),
        auto_fetch=False,
        execute_orders=False,
    )

    status = trader.start(request)

    assert status["config"]["initial_cash"] == pytest.approx(9100.0)

    startup_actions = status.get("startup_actions", [])
    loaded_actions = [
        action for action in startup_actions if action.get("type") == "initial_cash_loaded"
    ]
    assert loaded_actions, "expected persisted initial cash action"
    assert loaded_actions[0]["value"] == pytest.approx(4321.0)

    assert persisted_values and persisted_values[0][1] == pytest.approx(9100.0)

    trader.stop()


def test_build_iteration_request_uses_ibkr_cash_when_available(monkeypatch, tmp_path):
    cash_actions = [
        {"type": "ibkr_connect", "status": "connected", "purpose": "cash_fetch"},
        {"type": "ibkr_cash_snapshot", "status": "ok", "value": 5000.0},
    ]

    def fake_fetch_cash(*, settings):
        return 5000.0, list(cash_actions), []

    def fake_fetch_positions(*, settings):
        return {}, [
            {"type": "ibkr_connect", "status": "connected",
                "purpose": "positions_fetch"},
            {"type": "ibkr_positions_snapshot", "status": "ok"},
        ], []

    persist_calls = []

    def fake_persist(store_path, value):
        persist_calls.append((store_path, value))
        return []

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_account_cash", fake_fetch_cash)
    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_positions", fake_fetch_positions)
    monkeypatch.setattr(
        "src.algotrade.ui.app._persist_initial_cash_to_disk", fake_persist)

    trader = _configure_trader(tmp_path)

    (
        request,
        actions,
        warnings,
        positions_snapshot,
        liquidity_snapshot,
        # type: ignore[attr-defined]
    ) = trader._build_iteration_request(AppSettings())

    assert request.initial_cash == pytest.approx(5000.0)
    assert any(action.get("type") ==
               "ibkr_cash_snapshot" for action in actions)
    assert liquidity_snapshot["cash_available"] == pytest.approx(5000.0)
    assert positions_snapshot == {}
    assert warnings == []
    assert persist_calls and persist_calls[0][1] == pytest.approx(5000.0)


def test_build_iteration_request_raises_without_ibkr_cash(monkeypatch, tmp_path):
    def fake_fetch_cash(*, settings):
        return None, [
            {"type": "ibkr_connect", "status": "failed", "purpose": "cash_fetch"},
            {"type": "ibkr_cash_snapshot", "status": "error"},
        ], [
            "cash fetch failed"
        ]

    def fake_fetch_positions(*, settings):
        return {}, [
            {"type": "ibkr_connect", "status": "connected",
                "purpose": "positions_fetch"},
            {"type": "ibkr_positions_snapshot", "status": "ok"},
        ], []

    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_account_cash", fake_fetch_cash)
    monkeypatch.setattr(
        "src.algotrade.ui.app._fetch_ibkr_positions", fake_fetch_positions)

    trader = _configure_trader(tmp_path)
    trader._last_initial_cash = 2000.0  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError) as excinfo:
        trader._build_iteration_request(
            AppSettings())  # type: ignore[attr-defined]

    message = str(excinfo.value).lower()
    assert "unable to connect" in message
    assert "cash" in message


def test_execute_iteration_merges_payload_and_actions(monkeypatch, tmp_path):
    def fake_build_iteration_request(self, settings):  # noqa: D401
        request = MomentumPaperTradeRequest(
            symbols=["AAPL", "MSFT"],
            initial_cash=2500.0,
            training_window=(date(2024, 1, 1), date(2024, 6, 30)),
            paper_window=(date(2024, 7, 1), date(2024, 9, 30)),
            parameters=[MomentumParameterRequest()],
            store_path=str(tmp_path),
            bar_size="1d",
            auto_fetch=False,
            execute_orders=False,
        )
        extra_actions = [{"type": "pre_action", "detail": "cash_snapshot"}]
        extra_warnings = ["cash fallback applied"]
        positions_snapshot = {"AAPL": {"quantity": 5, "market_price": 100.0}}
        liquidity_snapshot = {
            "cash_available": 2500.0,
            "holdings_value": 500.0,
            "total_assets": 3000.0,
            "position_count": 1,
        }
        return (
            request,
            extra_actions,
            extra_warnings,
            positions_snapshot,
            liquidity_snapshot,
        )

    def fake_project_trade_plan(trades, *, initial_cash, actions, liquidity, buy_universe_label):
        actions.append({
            "type": "projected_plan",
            "buy_universe": buy_universe_label,
            "initial_cash": initial_cash,
        })

    def fake_run_realtime_trade(
        request,
        *,
        settings,
        positions_snapshot,
        liquidity_snapshot,
        extra_actions=None,
        extra_warnings=None,
        previous_quotes=None,
    ):  # noqa: D401
        assert isinstance(settings, AppSettings)
        actions = list(extra_actions or [])
        warnings = list(extra_warnings or [])
        actions.append({"type": "paper_action", "detail": "momentum"})
        warnings.append("paper trade warning")
        fake_project_trade_plan(
            [],
            initial_cash=float(request.initial_cash),
            actions=actions,
            liquidity=liquidity_snapshot,
            buy_universe_label="Configured universe",
        )
        return {
            "mode": "live",
            "symbols": request.symbols,
            "initial_cash": request.initial_cash,
            "paper_trades": [],
            "actions": actions,
            "warnings": warnings,
            "portfolio": {
                "positions": positions_snapshot,
                "liquidity": liquidity_snapshot,
                "cash": liquidity_snapshot["cash_available"],
            },
            "liquidity": liquidity_snapshot,
        }

    monkeypatch.setattr(
        MomentumLiveTrader,
        "_build_iteration_request",
        fake_build_iteration_request,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._run_momentum_realtime_trade",
        fake_run_realtime_trade,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._project_live_trade_plan",
        fake_project_trade_plan,
    )

    trader = _configure_trader(tmp_path)

    run = trader._execute_iteration()  # type: ignore[attr-defined]

    assert run.status == "completed"
    assert run.error is None
    assert run.payload is not None
    payload = run.payload

    actions = payload.get("actions", [])
    warnings = payload.get("warnings", [])
    assert {action.get("type") for action in actions} >= {
        "paper_action",
        "pre_action",
        "projected_plan",
    }
    assert "cash fallback" in " ".join(warnings)
    assert payload.get("initial_cash") == pytest.approx(2500.0)
    liquidity = payload.get("liquidity", {})
    assert liquidity.get("cash_available") == pytest.approx(2500.0)
    assert liquidity.get("holdings_value") == pytest.approx(500.0)
    portfolio = payload.get("portfolio", {})
    assert "positions" in portfolio
    assert portfolio.get("liquidity", {}).get(
        "total_assets") == pytest.approx(3000.0)


def test_execute_iteration_with_trades_sequences_and_executes_orders(monkeypatch, tmp_path):
    execution_calls: dict[str, Any] = {}

    def fake_build_iteration_request(self, settings):  # noqa: D401
        request = MomentumPaperTradeRequest(
            symbols=["MSFT", "AAPL", "QQQ"],
            initial_cash=5000.0,
            training_window=(date(2024, 1, 1), date(2024, 6, 30)),
            paper_window=(date(2024, 7, 1), date(2024, 7, 31)),
            parameters=[MomentumParameterRequest()],
            store_path=str(tmp_path),
            bar_size="1d",
            auto_fetch=False,
            execute_orders=True,
        )
        extra_actions = [{"type": "pre_action", "detail": "cash_snapshot"}]
        extra_warnings = ["cash fallback applied"]
        positions_snapshot: Dict[str, Dict[str, Any]] = {
            "MSFT": {"quantity": 4, "market_price": 320.0}
        }
        liquidity_snapshot = {
            "cash_available": 5000.0,
            "holdings_value": 1280.0,
            "total_assets": 6280.0,
            "position_count": 1,
        }
        return (
            request,
            extra_actions,
            extra_warnings,
            positions_snapshot,
            liquidity_snapshot,
        )

    trade_entries = [
        {"symbol": "MSFT", "quantity": -5, "price": 310.0},
        {"symbol": "AAPL", "quantity": 10, "price": 150.0},
        {"symbol": "QQQ", "quantity": 0, "price": 370.0},
    ]

    def fake_project_trade_plan(trades, *, initial_cash, actions, liquidity, buy_universe_label):
        sells = [
            str(entry.get("symbol", "")).strip()
            for entry in trades
            if entry.get("quantity", 0) < 0
        ]
        buys = [
            str(entry.get("symbol", "")).strip()
            for entry in trades
            if entry.get("quantity", 0) > 0
        ]
        sell_value = sum(
            abs(float(entry.get("quantity", 0) or 0)) *
            float(entry.get("price", 0) or 0)
            for entry in trades
            if entry.get("quantity", 0) < 0
        )
        buy_value = sum(
            abs(float(entry.get("quantity", 0) or 0)) *
            float(entry.get("price", 0) or 0)
            for entry in trades
            if entry.get("quantity", 0) > 0
        )
        liquidity["estimated_sell_value"] = sell_value
        liquidity["estimated_buy_value"] = buy_value
        liquidity["projected_cash_after_sells"] = float(
            initial_cash) + sell_value
        actions.append(
            {
                "type": "momentum_live_sequence",
                "sell_symbols": sells,
                "buy_symbols": buys,
                "sell_count": len(sells),
                "buy_count": len(buys),
                "available_cash_before_buys": float(initial_cash),
                "projected_cash_after_sells": float(initial_cash) + sell_value,
                "buy_universe": buy_universe_label,
                "sequence": ["sell", "buy"] if sells or buys else [],
            }
        )

    def fake_run_realtime_trade(
        request,
        *,
        settings,
        positions_snapshot,
        liquidity_snapshot,
        extra_actions=None,
        extra_warnings=None,
        previous_quotes=None,
    ):  # noqa: D401
        actions = list(extra_actions or [])
        warnings = list(extra_warnings or [])
        actions.append({"type": "paper_action", "detail": "momentum"})
        warnings.append("paper trade warning")
        portfolio_positions = {"MSFT": {"quantity": 2}}
        fake_project_trade_plan(
            [dict(entry) for entry in trade_entries],
            initial_cash=float(request.initial_cash),
            actions=actions,
            liquidity=liquidity_snapshot,
            buy_universe_label="Configured universe",
        )
        return {
            "mode": "live",
            "symbols": request.symbols,
            "initial_cash": request.initial_cash,
            "paper_trades": [dict(entry) for entry in trade_entries],
            "actions": actions,
            "warnings": warnings,
            "portfolio": {
                "positions": portfolio_positions,
                "liquidity": liquidity_snapshot,
                "cash": liquidity_snapshot["cash_available"],
            },
            "liquidity": liquidity_snapshot,
            "ibkr_positions": positions_snapshot,
        }

    def fake_execute_orders(trades, *, settings):  # noqa: D401
        execution_calls["called"] = True
        execution_calls["quantities"] = [
            trade.get("quantity") for trade in trades]
        return (
            [{"type": "ibkr_order_submitted", "symbol": "AAPL", "status": "submitted"}],
            ["execution warning stub"],
            [
                {
                    "input_index": 1,
                    "symbol": "AAPL",
                    "avg_price": 151.0,
                    "filled_quantity": 10,
                    "status": "Filled",
                }
            ],
        )

    monkeypatch.setattr(
        MomentumLiveTrader,
        "_build_iteration_request",
        fake_build_iteration_request,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._run_momentum_realtime_trade",
        fake_run_realtime_trade,
    )
    monkeypatch.setattr(
        "src.algotrade.ui.app._execute_ibkr_paper_orders",
        fake_execute_orders,
    )

    trader = _configure_trader(tmp_path)

    run = trader._execute_iteration()  # type: ignore[attr-defined]

    assert run.status == "completed"
    payload = run.payload
    assert payload is not None

    actions = payload.get("actions", [])
    warnings = payload.get("warnings", [])
    assert execution_calls.get("called") is True
    assert {action.get("type") for action in actions} >= {
        "pre_action",
        "paper_action",
        "momentum_live_sequence",
        "ibkr_order_submitted",
        "ibkr_execution_recorded",
    }
    assert "execution warning stub" in warnings

    reordered_trades = payload.get("paper_trades")
    assert isinstance(reordered_trades, list)
    assert [trade.get("quantity") for trade in reordered_trades] == [-5, 10, 0]
    assert reordered_trades[1].get("execution_price") == pytest.approx(151.0)
    assert reordered_trades[1].get("execution_filled_quantity") == 10

    liquidity = payload.get("liquidity", {})
    assert liquidity.get("estimated_sell_value") == pytest.approx(1550.0)
    assert liquidity.get("estimated_buy_value") == pytest.approx(1500.0)
    assert liquidity.get("projected_cash_after_sells") == pytest.approx(6550.0)

    sequence_actions = [action for action in actions if action.get(
        "type") == "momentum_live_sequence"]
    assert sequence_actions, "Expected momentum_live_sequence action"
    sequence_payload = sequence_actions[0]
    assert sequence_payload.get("sequence") == ["sell", "buy"]
    assert sequence_payload.get("sell_symbols") == ["MSFT"]
    assert sequence_payload.get("buy_symbols") == ["AAPL"]
    assert sequence_payload.get(
        "projected_cash_after_sells") == pytest.approx(6550.0)

    portfolio = payload.get("portfolio", {})
    assert portfolio.get("positions") == {"MSFT": {"quantity": 2}}
    assert portfolio.get("liquidity", {}).get(
        "estimated_buy_value") == pytest.approx(1500.0)
    assert payload.get("ibkr_positions") == {
        "MSFT": {"quantity": 4, "market_price": 320.0}}


def test_record_live_run_results_filters_historical_trades(tmp_path):
    trader = MomentumLiveTrader()
    trader._store_path = tmp_path  # type: ignore[attr-defined]

    payload = {
        "trade_date": "2025-10-09",
        "paper_trades": [
            {
                "symbol": "AAPL",
                "timestamp": "2025-10-09T14:45:00Z",
                "quantity": 5,
                "price": 175.0,
                "cash_after": 96500.0,
            },
            {
                "symbol": "MSFT",
                "timestamp": "2025-10-08T14:45:00Z",
                "quantity": 3,
                "price": 330.0,
                "cash_after": 95510.0,
            },
        ],
        "liquidity": {
            "total_assets": 120_000.0,
            "cash_available": 96_500.0,
            "holdings_value": 23_500.0,
            "position_count": 2,
        },
        "initial_cash": 100_000.0,
    }

    run = LiveTraderRun(
        run_id="run-test",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status="completed",
        payload=payload,
        error=None,
    )

    trader._record_live_run_results(run)  # type: ignore[attr-defined]

    history = trader.history()
    trades = history.get("trade_history", [])
    assert len(trades) == 1
    recorded = trades[0]
    assert recorded["symbol"] == "AAPL"
    assert recorded["trade_date"] == "2025-10-09"
    assert recorded["run_id"] == "run-test"

    equity_curve = history.get("equity_curve", [])
    assert len(equity_curve) == 1
    assert equity_curve[0]["total_assets"] == pytest.approx(120_000.0)

    state_path = tmp_path / "momentum_live_state.json"
    assert state_path.exists(), "expected state file to be persisted"
    persisted = json.loads(state_path.read_text())
    assert len(persisted.get("trade_history", [])) == 1
    assert persisted["trade_history"][0]["symbol"] == "AAPL"


def test_reset_history_requires_stopped(tmp_path):
    trader = MomentumLiveTrader()
    trader._store_path = tmp_path  # type: ignore[attr-defined]

    payload = {
        "trade_date": "2025-10-09",
        "paper_trades": [
            {
                "symbol": "AAPL",
                "timestamp": "2025-10-09T14:45:00Z",
                "quantity": 5,
                "price": 175.0,
                "cash_after": 96500.0,
            },
        ],
        "liquidity": {
            "total_assets": 120_000.0,
            "cash_available": 96_500.0,
            "holdings_value": 23_500.0,
        },
        "initial_cash": 100_000.0,
    }

    run = LiveTraderRun(
        run_id="run-reset",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status="completed",
        payload=payload,
        error=None,
    )

    trader._record_live_run_results(run)  # type: ignore[attr-defined]

    result = trader.reset_history()
    assert result["status"] == "reset"
    assert result["trade_history"] == []
    assert result["equity_curve"] == []

    history = trader.history()
    assert history["trade_history"] == []
    assert history["equity_curve"] == []

    trader._running = True  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        trader.reset_history()
