from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from algotrade.experiments import (
    MomentumExperimentConfig,
    MomentumParameters,
    run_momentum_experiment,
)


def sample_parameters(rng: random.Random) -> MomentumParameters:
    lookback = rng.choice([42, 63, 84, 105, 126, 168, 189, 210, 252])
    skip = rng.choice([0, 5, 10, 21])
    if skip >= lookback:
        skip = max(0, lookback // 4)
    rebalance = rng.choice([5, 10, 15, 21, 42])
    max_positions = rng.choice([3, 4, 5, 6, 8, 10])
    cash_reserve = rng.choice([0.05, 0.1, 0.15, 0.2, 0.25])
    min_momentum = rng.choice([0.0, 0.02, 0.04, 0.06, 0.08])
    vol_window = rng.choice([10, 15, 20, 30])
    vol_exponent = rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])
    lot_size = 1

    return MomentumParameters(
        lookback_days=lookback,
        skip_days=skip,
        rebalance_days=rebalance,
        max_positions=max_positions,
        lot_size=lot_size,
        cash_reserve_pct=cash_reserve,
        min_momentum=min_momentum,
        volatility_window=vol_window,
        volatility_exponent=vol_exponent,
    )


def run_search(iterations: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    config = MomentumExperimentConfig(
        store_path=Path("data/raw/polygon/daily"),
        universe=[
            "AAPL", "ADBE", "ADI", "ADSK", "AMAT", "AMD", "APP", "ARM", "ASML", "AVGO",
            "CDNS", "CDW", "CRWD", "CSCO", "CTSH", "DDOG", "FTNT", "GFS", "INTC", "INTU",
            "KLAC", "LRCX", "MCHP", "MRVL", "MSFT", "MSTR", "MU", "NVDA", "NXPI", "ON",
            "PANW", "PLTR", "QCOM", "ROP", "SHOP", "SNPS", "TEAM", "TXN", "WDAY", "ZS"
        ],
        initial_cash=100_000.0,
        training_window=(date(2023, 10, 10), date(2025, 10, 7)),
        paper_window=(date(2024, 10, 8), date(2025, 10, 7)),
    )

    best: list[tuple[float, dict[str, Any]]] = []

    for idx in range(iterations):
        params = sample_parameters(rng)
        outcome = run_momentum_experiment(config, params)
        sharpe = outcome.paper_metrics.sharpe_ratio
        payload = {
            "iteration": idx,
            "paper_sharpe": sharpe,
            "paper_total_return": outcome.paper_metrics.total_return,
            "paper_cagr": outcome.paper_metrics.cagr,
            "paper_max_drawdown": outcome.paper_metrics.max_drawdown,
            "train_sharpe": outcome.training_metrics.sharpe_ratio,
            "train_cagr": outcome.training_metrics.cagr,
            "train_max_drawdown": outcome.training_metrics.max_drawdown,
            "parameters": asdict(params),
        }
        if len(best) < 15:
            best.append((sharpe, payload))
        else:
            worst_idx = min(range(len(best)), key=lambda i: best[i][0])
            if sharpe > best[worst_idx][0]:
                best[worst_idx] = (sharpe, payload)

    best.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in best]


def main() -> None:
    results = run_search(iterations=200, seed=2025)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
