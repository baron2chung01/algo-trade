from datetime import date, datetime, timedelta, timezone
from math import sin

from fastapi.testclient import TestClient

from algotrade.data.schemas import IBKRBar, IBKRBarDataFrame
from algotrade.data.stores.local import ParquetBarStore
from algotrade.ui import create_app


def _build_bars(closes: list[float], start: date) -> IBKRBarDataFrame:
    rows: list[IBKRBar] = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += timedelta(days=1)
        timestamp = datetime(current.year, current.month,
                             current.day, tzinfo=timezone.utc)
        rows.append(
            IBKRBar(
                timestamp=timestamp,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1_000,
                average=close,
                bar_count=100,
            )
        )
        current += timedelta(days=1)
    return IBKRBarDataFrame.from_bars(rows)


def _generate_prices(length: int, base: float) -> list[float]:
    values: list[float] = []
    for idx in range(length):
        # Small oscillations plus upward drift for variety.
        values.append(base + 0.5 * idx + 5.0 * sin(idx / 20.0))
    return values


def _default_parameter_payload() -> dict:
    return {
        "entry_threshold": {"minimum": 5, "maximum": 10, "step": 5},
        "exit_threshold": {"minimum": 70, "maximum": 80, "step": 10},
        "max_hold_days": {"minimum": 3, "maximum": 5, "step": 2, "include_infinite": True},
        "target_position_pct": {"minimum": 10, "maximum": 20, "step": 10},
        "include_no_stop_loss": True,
        "lot_size": 10,
    }


def test_optimize_endpoint_returns_payload(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    start_date = date.today() - timedelta(days=4 * 365)
    length = 900
    store.save("AAPL", "1d", _build_bars(
        _generate_prices(length, 120.0), start_date))
    store.save("MSFT", "1d", _build_bars(
        _generate_prices(length, 220.0), start_date))

    client = TestClient(create_app())
    response = client.post(
        "/api/optimize",
        json={
            "symbols": ["AAPL", "MSFT"],
            "store_path": str(store.root),
            "limit": 120,
            "auto_fetch": False,
            "parameter_spec": _default_parameter_payload(),
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert "AAPL" in payload["results"]
    aapl = payload["results"]["AAPL"]
    assert aapl["candles"]
    assert aapl["metrics"]["final_equity"] >= 0
    assert "sharpe_ratio" in aapl["metrics"]
    assert aapl["equity_curve"]
    optimization = aapl["optimization"]
    assert optimization["best_parameters"]["entry_threshold"] >= 0
    assert optimization["training"]["metrics"]["final_equity"] >= 0
    assert optimization["paper"]["metrics"]["final_equity"] >= 0
    assert "sharpe_ratio" in optimization["paper"]["metrics"]
    assert len(optimization["rankings"]) <= 10
    assert "warnings" not in payload


def test_optimize_endpoint_reports_missing_files(tmp_path):
    client = TestClient(create_app())
    response = client.post(
        "/api/optimize",
        json={
            "symbols": ["AAPL"],
            "store_path": str(tmp_path / "bars"),
            "auto_fetch": False,
            "parameter_spec": _default_parameter_payload(),
        },
    )
    assert response.status_code == 404
    payload = response.json()
    assert "detail" in payload
    assert payload["detail"]["message"] == "No historical bars found for the requested symbols."
    missing = payload["detail"]["missing"]
    assert isinstance(missing, list) and missing
    assert missing[0]["symbol"] == "AAPL"


def test_symbols_endpoint_lists_cached_symbols(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    start_date = date.today() - timedelta(days=30)
    store.save("AAPL", "1d", _build_bars([150.0, 151.0, 152.0], start_date))
    store.save("MSFT", "1d", _build_bars([300.0, 302.0, 301.0], start_date))

    client = TestClient(create_app())
    response = client.get(
        "/api/symbols",
        params={"store_path": str(store.root)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload["symbols"]) == ["AAPL", "MSFT"]


def test_optimize_endpoint_supports_only_infinite_hold(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    start_date = date.today() - timedelta(days=4 * 365)
    series = _generate_prices(800, 150.0)
    store.save("AAPL", "1d", _build_bars(series, start_date))

    client = TestClient(create_app())
    payload = {
        "symbols": ["AAPL"],
        "store_path": str(store.root),
        "auto_fetch": False,
        "parameter_spec": {
            "entry_threshold": {"minimum": 5, "maximum": 5, "step": 1},
            "exit_threshold": {"minimum": 70, "maximum": 70, "step": 1},
            "max_hold_days": {
                "minimum": 1,
                "maximum": 5,
                "step": 2,
                "include_infinite": True,
                "only_infinite": True,
            },
            "target_position_pct": {"minimum": 10, "maximum": 10, "step": 1},
            "include_no_stop_loss": True,
            "lot_size": 10,
        },
    }

    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    aapl = body["results"]["AAPL"]
    best_hold = aapl["optimization"]["best_parameters"]["max_hold_days"]
    assert best_hold == 0


def test_optimize_endpoint_supports_vcp(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    start_date = date.today() - timedelta(days=4 * 365)
    series = _generate_prices(900, 120.0)
    store.save("AAPL", "1d", _build_bars(series, start_date))

    client = TestClient(create_app())
    payload = {
        "symbols": ["AAPL"],
        "store_path": str(store.root),
        "strategy": "vcp",
        "auto_fetch": False,
        "limit": 120,
        "vcp_spec": {
            "base_lookback_days": {"minimum": 50, "maximum": 50, "step": 1},
            "pivot_lookback_days": {"minimum": 3, "maximum": 3, "step": 1},
            "min_contractions": {"minimum": 3, "maximum": 3, "step": 1},
            "max_contraction_pct": {"minimum": 0.12, "maximum": 0.12, "step": 0.01},
            "contraction_decay": {"minimum": 0.7, "maximum": 0.7, "step": 0.05},
            "breakout_buffer_pct": {"minimum": 0.002, "maximum": 0.002, "step": 0.001},
            "volume_squeeze_ratio": {"minimum": 0.7, "maximum": 0.7, "step": 0.05},
            "breakout_volume_ratio": {"minimum": 1.5, "maximum": 1.5, "step": 0.1},
            "volume_lookback_days": {"minimum": 20, "maximum": 20, "step": 1},
            "trend_ma_period": {"minimum": 50, "maximum": 50, "step": 1},
            "stop_loss_r_multiple": {"minimum": 1.0, "maximum": 1.0, "step": 0.1},
            "profit_target_r_multiple": {"minimum": 2.5, "maximum": 2.5, "step": 0.1},
            "trailing_stop_r_multiple": {"minimum": 1.5, "maximum": 1.5, "step": 0.1},
            "include_no_trailing_stop": True,
            "max_hold_days": {
                "minimum": 45,
                "maximum": 45,
                "step": 1,
                "include_infinite": False,
                "only_infinite": False,
            },
            "target_position_pct": {"minimum": 10, "maximum": 10, "step": 1},
            "lot_size": 5,
            "cash_reserve_pct": 0.1,
        },
    }

    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "vcp"
    assert body["symbols"] == ["AAPL"]
    aapl = body["results"]["AAPL"]
    assert aapl["strategy"] == "vcp"
    assert "annotations" in aapl
    assert isinstance(aapl["annotations"], list)
    assert "optimization" in aapl
    params = aapl["optimization"]["best_parameters"]
    assert params["base_lookback_days"] == 50
    assert params["pivot_lookback_days"] == 3
