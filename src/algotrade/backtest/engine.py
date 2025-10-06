"""Backtest engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Sequence, Tuple

from ..data.stores.local import ParquetBarStore
from .data import BarSlice, LocalDailyBarFeed
from .fees import CommissionModel, SlippageModel, ZeroCommission, ZeroSlippage
from .portfolio import Portfolio, PortfolioState, Trade
from .strategy import Order, Strategy, StrategyContext


@dataclass(slots=True)
class BacktestConfig:
    symbols: Sequence[str]
    start: date | None = None
    end: date | None = None
    bar_size: str = "1d"
    initial_cash: float = 100_000.0
    commission_model: CommissionModel | None = None
    slippage_model: SlippageModel | None = None


@dataclass(slots=True)
class BacktestResult:
    equity_curve: List[Tuple[datetime, float]]
    trades: List[Trade]
    final_state: PortfolioState


class BacktestEngine:
    """Coordinate data feed, strategy decisions, and portfolio accounting."""

    def __init__(
        self,
        config: BacktestConfig,
        store: ParquetBarStore,
        *,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.commission_model = commission_model or config.commission_model or ZeroCommission()
        self.slippage_model = slippage_model or config.slippage_model or ZeroSlippage()

    def _build_feed(self) -> LocalDailyBarFeed:
        return LocalDailyBarFeed(
            self.store,
            self.config.symbols,
            bar_size=self.config.bar_size,
            start=self.config.start,
            end=self.config.end,
        )

    def run(self, strategy: Strategy) -> BacktestResult:
        feed = self._build_feed()
        portfolio = Portfolio(self.config.initial_cash)
        equity_curve: List[tuple] = []
        trades: List[Trade] = []

        for bar_slice in feed:
            context = StrategyContext(portfolio=portfolio.snapshot())
            orders = strategy.on_bar(context, bar_slice) or []
            for order in orders:
                if order.symbol not in bar_slice.bars:
                    continue
                base_price = bar_slice.bars[order.symbol].close
                fill_price = self.slippage_model.adjust(order.symbol, order.quantity, base_price)
                commission = self.commission_model.calculate(order.symbol, order.quantity, fill_price)
                trade = portfolio.execute(
                    order.symbol,
                    order.quantity,
                    fill_price,
                    bar_slice.timestamp,
                    commission=commission,
                    slippage=fill_price - base_price,
                )
                trades.append(trade)
            equity = portfolio.mark_to_market(bar_slice.bars)
            equity_curve.append((bar_slice.timestamp, equity))

        return BacktestResult(equity_curve=equity_curve, trades=trades, final_state=portfolio.snapshot())
