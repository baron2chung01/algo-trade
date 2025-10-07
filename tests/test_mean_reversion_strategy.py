from datetime import date, datetime, timedelta, timezone

import pytest

from algotrade.backtest import BacktestConfig, BacktestEngine
from algotrade.data.schemas import IBKRBar, IBKRBarDataFrame
from algotrade.data.stores.local import ParquetBarStore
from algotrade.strategies import MeanReversionConfig, MeanReversionStrategy


def _build_bars(closes: list[float], start: date) -> IBKRBarDataFrame:
    rows: list[IBKRBar] = []
    current = start
    for close in closes:
        while current.weekday() >= 5:  # skip weekends
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


def _make_store(tmp_path, series: dict[str, list[float]], start: date) -> ParquetBarStore:
    store = ParquetBarStore(tmp_path / "bars")
    for symbol, closes in series.items():
        df = _build_bars(closes, start)
        store.save(symbol, "1d", df)
    return store


def _strategy_config(symbols: list[str], initial_cash: float = 100_000.0, **overrides):
    params = dict(
        symbols=symbols,
        initial_cash=initial_cash,
        entry_threshold=20.0,
        exit_threshold=70.0,
        target_position_pct=0.5,
        max_positions=2,
        max_hold_days=5,
        stop_loss_pct=None,
        lot_size=10,
        cash_reserve_pct=0.0,
    )
    params.update(overrides)
    return MeanReversionConfig(**params)


def test_mean_reversion_enters_on_oversold_and_exits_on_rebound(tmp_path):
    store = _make_store(
        tmp_path, {"AAPL": [100.0, 99.0, 95.0, 110.0]}, start=date(2024, 1, 2))
    config = BacktestConfig(symbols=["AAPL"], initial_cash=100_000.0, start=date(
        2024, 1, 2), end=date(2024, 1, 5))
    strategy = MeanReversionStrategy(
        _strategy_config(["AAPL"], max_positions=1))

    result = BacktestEngine(config=config, store=store).run(strategy)

    assert len(result.trades) == 2
    buy, sell = result.trades
    assert buy.symbol == "AAPL" and sell.symbol == "AAPL"
    assert buy.quantity == 520
    assert buy.price == pytest.approx(95.0)
    assert sell.quantity == -520
    assert sell.price == pytest.approx(110.0)
    assert result.final_state.positions["AAPL"].quantity == 0
    assert result.final_state.cash > config.initial_cash


def test_mean_reversion_respects_max_positions(tmp_path):
    series = {
        "AAPL": [100.0, 99.0, 95.0, 104.0],
        "MSFT": [200.0, 198.0, 190.0, 205.0],
    }
    store = _make_store(tmp_path, series, start=date(2024, 1, 2))
    config = BacktestConfig(symbols=["AAPL", "MSFT"], initial_cash=100_000.0, start=date(
        2024, 1, 2), end=date(2024, 1, 5))
    strategy = MeanReversionStrategy(
        _strategy_config(["AAPL", "MSFT"], max_positions=1,
                         target_position_pct=0.4)
    )

    result = BacktestEngine(config=config, store=store).run(strategy)

    traded_symbols = {trade.symbol for trade in result.trades}
    assert traded_symbols == {"AAPL"}
    assert result.final_state.positions.get(
        "MSFT") is None or result.final_state.positions["MSFT"].quantity == 0


def test_mean_reversion_exits_on_max_hold(tmp_path):
    store = _make_store(
        tmp_path, {"AAPL": [100.0, 99.0, 95.0, 94.0, 93.0]}, start=date(2024, 1, 2))
    config = BacktestConfig(symbols=["AAPL"], initial_cash=50_000.0, start=date(
        2024, 1, 2), end=date(2024, 1, 8))
    strategy = MeanReversionStrategy(
        _strategy_config(["AAPL"], initial_cash=50_000.0,
                         max_positions=1, max_hold_days=2, exit_threshold=99.0)
    )

    result = BacktestEngine(config=config, store=store).run(strategy)

    assert len(result.trades) == 2
    _, sell = result.trades
    assert sell.price == pytest.approx(93.0)
    assert result.final_state.positions["AAPL"].quantity == 0


def test_mean_reversion_triggers_stop_loss(tmp_path):
    store = _make_store(
        tmp_path, {"AAPL": [100.0, 99.0, 95.0, 89.0]}, start=date(2024, 1, 2))
    config = BacktestConfig(symbols=["AAPL"], initial_cash=60_000.0, start=date(
        2024, 1, 2), end=date(2024, 1, 6))
    strategy = MeanReversionStrategy(
        _strategy_config(
            ["AAPL"],
            initial_cash=60_000.0,
            max_positions=1,
            stop_loss_pct=0.05,
            max_hold_days=5,
            exit_threshold=99.0,
        )
    )

    result = BacktestEngine(config=config, store=store).run(strategy)

    assert len(result.trades) == 2
    sell = result.trades[1]
    assert sell.price == pytest.approx(89.0)
    assert result.final_state.positions["AAPL"].quantity == 0
