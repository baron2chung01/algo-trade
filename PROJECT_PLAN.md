# Algo-Trade Project Plan

## Mission

Build a Python-based trading research and execution stack that can research, backtest (minimum 3-year lookback), and live-trade multiple US equity strategies through IBKR TWS, starting with a mean reversion system optimized for a ~$10k CAD (≈$7.5k USD) portfolio and 3–4 hours of daily oversight.

## Guardrails & Constraints

- **Capital**: Treat capital base as $C=7{,}500\,\text{USD}$ after currency conversion and buffer 5% for fees.
- **Time**: Max 4 hours/day hands-on; prefer end-of-day batch jobs and scheduled intraday monitors.
- **Compliance**: Pattern Day Trader rule (<$25k USD) permits at most 3 day trades per rolling 5 business days.
- **Broker**: IBKR TWS + API (ib-insync). Paper trading first (port 7497) with clear toggle for live (port 7496).
- **Data Access**: Acquire at least 3 years of daily bars for all symbols and 1 minute intraday bars for intraday strategies; cache locally (Parquet) for reproducibility.

## Strategy Portfolio Roadmap

| Priority | Strategy | Horizon | Core Idea | Status |
| --- | --- | --- | --- | --- |
| 1 | Mean Reversion (focus) | Multi-day | Buy oversold large-cap US stocks (e.g., S&P 100) based on RSI(2), close gap fill, or z-score of returns; exit on mean reversion to moving average. | **Start here** |
| 2 | Swing Momentum ETF Rotation | Multi-day | Rotate among SPY/QQQ/IWM (and potentially sector ETFs) on breakouts or MA crosses. | Todo |
| 3 | Opening Range Breakout (ORB) | Intraday | Break of first 15-minute range on liquid tickers with bracket orders. | Todo |
| 4 | Gap-and-Go | Intraday | Trade premarket gappers with volume filters and tight risk. | Todo |
| 5 | Earnings Drift | 1–3 days | Follow-through trades post-earnings with fundamental surprise filters. | Todo |

## Phased Delivery Plan

1. **Foundation & Environment**
   - Setup `pyproject.toml`, virtual environment, linting (ruff), formatting (black), type hints (mypy), and `.env` secret handling.
   - Integrate logging and configuration management (pydantic settings / Hydra / YAML).
2. **Data Layer**
  - Provider adapters (e.g., Polygon, Tiingo, AlphaVantage for daily; Polygon or Intrinio for intraday) with rate limit handling.
  - Date-aligned, split- and dividend-adjusted historical data stored as Parquet per symbol/timeframe.
  - Normalize all bar payloads to the IBKR TWS schema (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `average`, `bar_count`) to guarantee drop-in compatibility with the broker.
  - Universe management: S&P 100 constituents with point-in-time membership snapshot.
  - CLI ingestion utilities (starting with Polygon daily fetcher) writing into deterministic folder structure.

3. **Backtest Engine**
   - Event-driven loop with NYSE calendar and RTH session filtering.
   - Portfolio accounting incorporating commissions, slippage, borrow, and cash management.
   - Metrics module: CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor, exposure, turnover, trade count, and PDT guardrail.
   - Reporting: HTML dashboard + CSV artifacts per run, including equity curve and drawdown charts.
4. **Strategy Implementation**
   - Implement strategy interface (`signal`, `size`, `risk_controls`).
   - Build adapters for each candidate strategy starting with mean reversion.
5. **Experimentation & Validation**
   - 3-year backtests (train/validation/test splits) with parameter grid search, walk-forward analysis, and Monte Carlo randomizations.
   - Compare strategies on risk-adjusted metrics and capital efficiency.
6. **Paper Trading Readiness**
   - IBKR execution layer with bracket/OCO orders, kill switches, throttling, and order/position reconciliation.
   - Runbooks and dashboards for daily monitoring.
7. **Live Cutover (Optional)**
   - Currency conversion workflow, compliance checks, and incremental rollout plan.

## Mean Reversion Strategy Blueprint (Phase 1 Focus)

- **Universe**: S&P 100 constituents filtered by minimum average daily dollar volume ($\geq 20\text{M USD}$) and price ($\geq 20$ USD).
- **Signal Inputs**:
  - RSI(2) < threshold (e.g., 5).
  - Close < 5-day EMA or z-score of close-to-close returns below -1.5.
  - Optional filter: prior-day gap down magnitude > 1%.
- **Entry Logic**:
  - Generate signal when criteria met on daily close; place next-day limit entry at min(close, close - 0.25 \* ATR(10)).
  - Cap open positions (max 5) and cluster exposure (max 2 per sector).
- **Exit Logic**:
  - Target: close crosses above 5-day EMA or RSI(2) > 70.
  - Protective stop at recent swing low - 1 \* ATR(10) (cancel if not filled next day).
  - Hard time stop at 5 trading days if mean reversion fails.
- **Position Sizing**:
  - Risk per trade $R = \min(0.005, 0.01)$ of capital; shares $= \left\lfloor \frac{R \cdot C}{\text{EntryPrice} - \text{StopPrice}} \right\rfloor$.
  - Max capital per position 25% of portfolio.
- **Risk Controls**:
  - Daily loss limit 2% of equity; halt new entries if triggered.
  - Portfolio heatmap for sector concentration.
- **Backtest Tasks**:
  - Acquire 3 years daily data with corporate action adjustments.
  - Implement slippage model (e.g., 5 bps per trade) and IBKR commissions (USD 0.005/share, $1 min).
  - Evaluate parameter grid (RSI thresholds, ATR multipliers, holding periods).
  - Produce performance report and trade log.

## Data Management Plan

- **Sources**: Prioritize Polygon/Tiingo for historical accuracy; fallback: AlphaVantage for daily (with cleaning) and IBKR for live quotes.
- **Storage**: `/data/raw/<provider>/<symbol>_<freq>.parquet` with metadata (download timestamp, provider, adjustments).
- **Quality Checks**: Validate against survivorship bias, missing bars, splits/dividends. Automated unit tests on sample symbols.
- **Refresh Cadence**: Daily cron to update EOD bars; intraday sync on demand pre-market.

## Backtest & Analytics Requirements

- Deterministic random seed handling for Monte Carlo resampling.
- Parameter sweeps recorded with config hash; results stored in `/reports/<strategy>/<timestamp>/`.
- Visualization: Equity curve, drawdown, rolling Sharpe, distribution of trade returns.
- Compliance: Flag scenarios exceeding PDT day-trade limits.

## Execution Layer Roadmap

- Abstraction for broker operations (submit orders, modify, cancel, fetch positions, PnL).
- Bracket/OCO order factory with stop-loss and take-profit.
- Safety infrastructure:
  - Max open orders/trades per session.
  - Heartbeat watchdog (auto cancel if no ACK within threshold).
  - Kill switch command.
- Live vs paper toggle via configuration; environment variables for credentials.

## Tooling & Infrastructure

- **Core Libraries**: `ib-insync`, `pandas`, `numpy`, `scipy`, `pyarrow`, `backtrader` or custom event engine, `plotly`/`matplotlib`, `pydantic` or `hydra` for configs.
- **Dev Utilities**: `pytest`, `coverage`, `ruff`, `black`, `mypy`, `pre-commit` hooks.
- **Scheduling**: `APScheduler` or cron jobs for daily tasks; consider lightweight dashboard (Streamlit/FastAPI) later.
- **CI/CD**: GitHub Actions for lint + tests (future).

## Reporting & Documentation

- Strategy specification sheets with hypothesis, indicators, entry/exit, risk, and validation evidence.
- Backtest summary per strategy with narrative interpretation and limitations.
- Live trading runbook: setup steps, monitoring checklist, incident response.
- Changelog tracking parameter changes and deployment dates.

## Risk Register

| Risk                                 | Impact                | Mitigation                                                                    |
| ------------------------------------ | --------------------- | ----------------------------------------------------------------------------- |
| Data gaps or corporate action errors | Misleading backtests  | Multiple data sources, validation scripts, manual spot checks                 |
| Overfitting during optimization      | Poor live performance | Walk-forward split, cross-validation, out-of-sample test, parameter shrinkage |
| PDT violations                       | Trading restrictions  | Track day trades in engine, throttle entries                                  |
| API disconnections                   | Lost execution        | Heartbeat monitor, auto-reconnect, fallback alerts                            |
| Strategy crowding/decay              | Returns degrade       | Continuous monitoring, deploy multiple decorrelated strategies                |

## Immediate Next Actions (Mean Reversion Sprint)

1. **Environment setup**: initialize repo scaffold (pyproject, lint, venv, .env).
2. **Universe snapshot**: ingest point-in-time S&P 100 membership (CSV/API), persist versioned lists in `data/universe/`; new CLI supports auto-fetching from Wikipedia.
3. **Daily bars**: bulk-download 3 years of Polygon daily OHLCV for universe constituents into Parquet cache (automated via enhanced `fetch_polygon_daily` CLI).
4. **Backtest skeleton**: expand the new engine (event loop + portfolio skeleton now in place) with calendar handling and commission/slippage models.
5. **Mean reversion strategy module**: code signal, position sizing, exits, risk limits with unit tests.
6. **Experiment #1**: baseline parameter sweep over 2012–2024 (train 2012–2022, validate 2023, test 2024 YTD).
7. **Reporting**: generate initial performance report and document findings/background assumptions.

## Backlog (Post Mean Reversion)

- Extend engine for intraday 1-min data (ORB, Gap-and-Go).
- Implement ETF momentum strategy and earnings drift module.
- Live execution harness with monitoring UI.
- Performance comparison dashboard across strategies.
- Automated risk alerting (email/Slack) for paper/live runs.

## Definition of Done

- 3-year backtest suite with reproducible results and documented methodology.
- Deployed paper trading scripts for chosen strategies with protective controls.
- Comprehensive documentation enabling transition to live trading once metrics validated.
