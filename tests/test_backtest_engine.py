from datetime import date, datetime, timezone

import pytest

from algotrade.backtest import (
    BacktestConfig,
    BacktestEngine,
    BpsSlippage,
    InteractiveBrokersCommission,
    LocalDailyBarFeed,
    Order,
    StrategyContext,
)
from algotrade.data.schemas import IBKRBar, IBKRBarDataFrame
from algotrade.data.stores.local import ParquetBarStore


class BuySellStrategy:
    def __init__(self) -> None:
        self._entered = False

    def on_bar(self, context: StrategyContext, data):
        orders: list[Order] = []
        if not self._entered:
            cash = context.portfolio.cash
            price = data.bars["AAPL"].close
            quantity = int(cash // (price * 10)) * 10 or 10
            orders.append(Order(symbol="AAPL", quantity=quantity))
            self._entered = True
        else:
            quantity = context.portfolio.position(
                "AAPL").quantity  # type: ignore[union-attr]
            if quantity > 0:
                orders.append(Order(symbol="AAPL", quantity=-quantity))
        return orders


def make_store(tmp_path):
    store = ParquetBarStore(tmp_path / "bars")
    bars = IBKRBarDataFrame.from_bars(
        [
            IBKRBar(
                timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.5,
                close=101.0,
                volume=1_000,
                average=100.5,
                bar_count=200,
            ),
            IBKRBar(
                timestamp=datetime(2024, 1, 4, tzinfo=timezone.utc),
                open=101.0,
                high=105.0,
                low=100.5,
                close=104.0,
                volume=1_200,
                average=103.0,
                bar_count=220,
            ),
            IBKRBar(
                timestamp=datetime(
                    2024, 1, 6, tzinfo=timezone.utc),  # Saturday
                open=104.0,
                high=104.5,
                low=103.5,
                close=104.2,
                volume=500,
                average=104.0,
                bar_count=80,
            ),
        ]
    )
    store.save("AAPL", "1d", bars)
    return store


def test_backtest_engine_executes_strategy(tmp_path):
    store = make_store(tmp_path)
    config = BacktestConfig(symbols=["AAPL"], initial_cash=100_000.0, start=date(
        2024, 1, 3), end=date(2024, 1, 4))
    engine = BacktestEngine(config=config, store=store)
    strategy = BuySellStrategy()

    result = engine.run(strategy)

    assert len(result.equity_curve) == 2
    assert result.trades  # at least one trade executed
    assert result.final_state.cash > config.initial_cash  # profit after sell
    assert result.final_state.positions["AAPL"].quantity == 0


def test_backtest_engine_applies_commission_and_slippage(tmp_path):
    store = make_store(tmp_path)
    config = BacktestConfig(
        symbols=["AAPL"],
        initial_cash=10_000.0,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
        commission_model=InteractiveBrokersCommission(
            per_share=0.01, minimum=1.0),
        slippage_model=BpsSlippage(bps=100),
    )
    engine = BacktestEngine(config=config, store=store)

    class OneShotStrategy:
        def on_bar(self, context: StrategyContext, data):
            if context.portfolio.position("AAPL"):
                return []
            return [Order(symbol="AAPL", quantity=100)]

    result = engine.run(OneShotStrategy())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == pytest.approx(102.01)  # 1% slippage on $101 close
    assert trade.slippage == pytest.approx(1.01)
    assert trade.commission == pytest.approx(1.0)
    expected_cash = 10_000.0 - (102.01 * 100) - 1.0
    assert result.final_state.cash == pytest.approx(expected_cash)


def test_local_feed_skips_non_trading_sessions(tmp_path):
    store = make_store(tmp_path)
    feed = LocalDailyBarFeed(store, symbols=["AAPL"], start=date(
        2024, 1, 1), end=date(2024, 1, 10))
    timestamps = [slice.timestamp.date() for slice in feed]
    assert date(2024, 1, 6) not in timestamps  # weekend should be excluded
    assert timestamps == [date(2024, 1, 3), date(2024, 1, 4)]
