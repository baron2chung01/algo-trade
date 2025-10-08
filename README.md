# Algo-Trade

Python research and execution stack for US equities strategies routed through Interactive Brokers (IBKR) Trader Workstation.

## Quickstart

1. **Create environment**

   ```bash
   uv venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. **Configure credentials**

   - Copy `.env.example` to `.env` and fill in IBKR client / port settings.

3. **Run tests**
   ```bash
   pytest
   ```

## Data providers

- **QuantConnect Data Library**: set `QUANTCONNECT_USER_ID` and `QUANTCONNECT_API_TOKEN` in `.env` to enable daily bar downloads via the `QuantConnectDailyEquityProvider`. Bars are normalized to the IBKR schema (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `average`, `bar_count`).

### Download daily bars (QuantConnect)

```bash
# Bulk download 3-year lookback for the latest S&P 100 universe
python -m scripts.fetch_quantconnect_daily --universe snp100 --years 3

# Ad-hoc tickers and custom range
python -m scripts.fetch_quantconnect_daily AAPL MSFT --start 2022-01-01 --end 2024-01-01
```

Results are written under `data/raw/quantconnect/daily/` (directories are created automatically). Use `--dry-run` to preview bar counts without touching the filesystem, or `--effective-date` to target an older universe snapshot. The legacy `fetch_polygon_daily` module now proxies to the QuantConnect downloader for backward compatibility.

### Ingest universe membership (S&P 100)

Provide a CSV with columns `effective_date` and `symbol`, then run:

```bash
python -m scripts.ingest_universe_snapshot data/source/snp100.csv --name snp100
```

Or fetch the latest list directly from Wikipedia:

```bash
python -m scripts.ingest_universe_snapshot --fetch-snp100 --effective-date 2025-10-06
```

The script normalizes symbols, sorts by date, and stores a Parquet file at `data/raw/universe/snp100/membership.parquet`. Load the latest membership in code via:

```python
from algotrade.data import latest_symbols, load_universe

snapshots = load_universe(Path("data/raw/universe/snp100/membership.parquet"))
current_symbols = latest_symbols(snapshots)
```

## Backtesting skeleton

Use the new backtest framework to iterate over cached bars and execute strategies with instant fills:

```python
from pathlib import Path
from algotrade.backtest import BacktestConfig, BacktestEngine, Order, Strategy, StrategyContext
from algotrade.data import ParquetBarStore


class BuyAndHold(Strategy):
   def __init__(self):
      self.entered = False

   def on_bar(self, context: StrategyContext, data):
      if self.entered:
         return []
      self.entered = True
      price = data.bars["AAPL"].close
      quantity = int(context.portfolio.cash // price)
      return [Order(symbol="AAPL", quantity=quantity)]


store = ParquetBarStore(Path("data/raw/quantconnect/daily"))
engine = BacktestEngine(BacktestConfig(symbols=["AAPL"]), store)
result = engine.run(BuyAndHold())
print(f"Final equity: {result.equity_curve[-1][1]:.2f}")
```

The engine stitches daily bars from the local Parquet cache, filters iterations to the XNYS trading calendar, calls `Strategy.on_bar` for each timestamp, executes returned market orders at the bar close, and tracks equity/position history for later analysis.

## Strategies

- **Mean Reversion RSI(2)** – Instantiate `MeanReversionStrategy` from `algotrade.strategies` with a `MeanReversionConfig` to trade oversold symbols using configurable capital allocation, max active positions, stop-loss, and holding-period rules. Drop it directly into the backtest engine:

  ```python
  from algotrade.strategies import MeanReversionConfig, MeanReversionStrategy

  strategy = MeanReversionStrategy(
        MeanReversionConfig(
              symbols=["AAPL", "MSFT"],
        initial_cash=10_000.0,
              target_position_pct=0.2,
              max_positions=3,
              max_hold_days=5,
        )
  )
  result = engine.run(strategy)
  ```

## Experiments

Use the experiment runner to sweep RSI(2) parameters across walk-forward splits and persist the results:

1. **Prepare a JSON config** (see `scripts/configs/mean_reversion_baseline.json` as a template):

   ```json
   {
     "store_path": "data/raw/quantconnect/daily",
     "symbols": ["AAPL", "MSFT", "GOOGL"],
     "initial_cash": 10000,
     "splits": [
       { "name": "train", "start": "2019-01-01", "end": "2022-12-31" },
       { "name": "validate", "start": "2023-01-01", "end": "2023-12-31" },
       { "name": "test", "start": "2024-01-01", "end": "2024-09-30" }
     ],
     "parameters": [
       { "entry_threshold": 15, "exit_threshold": 70, "max_hold_days": 5, "target_position_pct": 0.2 },
       { "entry_threshold": 5, "exit_threshold": 80, "max_hold_days": 3, "target_position_pct": 0.15, "stop_loss_pct": 0.05 }
     ]
   }
   ```

2. **Run the sweep**

   ```bash
   python -m scripts.run_mean_reversion_experiment path/to/config.json --output reports/mean_reversion/latest.csv
   ```

   The command prints a summary table to stdout and writes a CSV if `--output` is supplied. Each row contains the split name, parameter label, and metrics such as final equity, total return, CAGR, max drawdown, cash balance, and trade count.

## Web UI

Spin up a lightweight FastAPI + Plotly dashboard to visualize candlesticks, buy signals, and equity curves in the browser:

```bash
python -m scripts.run_ui --store-path data/raw/quantconnect/daily --port 8000
```

- Navigate to <http://127.0.0.1:8000> and enter the symbols, RSI thresholds, and allocation settings you want to inspect.
- The page overlays buy markers on top of candlesticks, streams the cumulative equity curve, and highlights metrics such as net profit, total return, and drawdown.
- Adjust the `--store-path` flag (or `ALGO_TRADE_STORE_PATH` environment variable) to point at the Parquet cache containing historical bars.
- On the VCP page, pick between an exhaustive grid search or the new simulated annealing optimizer. Grid runs every combination in your parameter ranges, while annealing samples the search space using the iteration, temperature, and cooling settings exposed in the form.

## Repository layout (work in progress)

- `src/algotrade/` – core package (data adapters, strategies, execution).
- `tests/` – automated test suite.
- `PROJECT_PLAN.md` – project roadmap and scope.

Refer to `PROJECT_PLAN.md` for the detailed roadmap and upcoming milestones.
