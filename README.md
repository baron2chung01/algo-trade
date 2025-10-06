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
- **Polygon.io**: set `POLYGON_API_KEY` in `.env` to enable daily bar downloads via the `PolygonDailyBarsProvider`. Bars are normalized to the IBKR schema (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `average`, `bar_count`).

### Download daily bars (Polygon)
````bash
# Bulk download 3-year lookback for the latest S&P 100 universe
python -m scripts.fetch_polygon_daily --universe snp100 --years 3

# Ad-hoc tickers and custom range
python -m scripts.fetch_polygon_daily AAPL MSFT --start 2022-01-01 --end 2024-01-01
````

Results are written under `data/raw/polygon/daily/` (directories are created automatically). Use `--dry-run` to preview bar counts without touching the filesystem, or `--effective-date` to target an older universe snapshot.

### Ingest universe membership (S&P 100)

Provide a CSV with columns `effective_date` and `symbol`, then run:

````bash
python -m scripts.ingest_universe_snapshot data/source/snp100.csv --name snp100
````

Or fetch the latest list directly from Wikipedia:

````bash
python -m scripts.ingest_universe_snapshot --fetch-snp100 --effective-date 2025-10-06
````

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


store = ParquetBarStore(Path("data/raw/polygon/daily"))
engine = BacktestEngine(BacktestConfig(symbols=["AAPL"]), store)
result = engine.run(BuyAndHold())
print(f"Final equity: {result.equity_curve[-1][1]:.2f}")
```

The engine stitches daily bars from the local Parquet cache, calls `Strategy.on_bar` for each timestamp, executes returned market orders at the bar close, and tracks equity/position history for later analysis.

## Repository layout (work in progress)

- `src/algotrade/` – core package (data adapters, strategies, execution).
- `tests/` – automated test suite.
- `PROJECT_PLAN.md` – project roadmap and scope.

Refer to `PROJECT_PLAN.md` for the detailed roadmap and upcoming milestones.
