#!/usr/bin/env python3
"""CLI utility to evaluate cross-sectional momentum strategies."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from algotrade.experiments import (
    MomentumExperimentConfig,
    MomentumParameters,
    run_momentum_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON file describing the experiment configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregated metrics as CSV.",
    )
    return parser.parse_args()


def load_parameters(payload: Dict[str, Any]) -> List[MomentumParameters]:
    return [
        MomentumParameters(
            lookback_days=int(item.get("lookback_days", 126)),
            skip_days=int(item.get("skip_days", 21)),
            rebalance_days=int(item.get("rebalance_days", 21)),
            max_positions=int(item.get("max_positions", 5)),
            lot_size=int(item.get("lot_size", 1)),
            cash_reserve_pct=float(item.get("cash_reserve_pct", 0.05)),
            min_momentum=float(item.get("min_momentum", 0.0)),
            volatility_window=int(item.get("volatility_window", 20)),
            volatility_exponent=float(item.get("volatility_exponent", 1.0)),
        )
        for item in payload.get("parameters", [])
    ]


def load_config(path: Path) -> tuple[MomentumExperimentConfig, List[MomentumParameters]]:
    data = json.loads(path.read_text())
    store_path = Path(data["store_path"]).expanduser()
    universe = [symbol.upper() for symbol in data["symbols"]]
    initial_cash = float(data.get("initial_cash", 100_000.0))
    training_start, training_end = _parse_date_pair(data["training_window"])
    paper_start, paper_end = _parse_date_pair(data["paper_window"])
    bar_size = data.get("bar_size", "1d")
    calendar_name = data.get("calendar_name", "XNYS")

    config = MomentumExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        training_window=(training_start, training_end),
        paper_window=(paper_start, paper_end),
        bar_size=bar_size,
        calendar_name=calendar_name,
    )
    parameters = load_parameters(data)
    if not parameters:
        parameters = [MomentumParameters()]
    return config, parameters


def _parse_date_pair(raw: Any) -> tuple[date, date]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("Date pair must be a list with [start, end].")
    return date.fromisoformat(raw[0]), date.fromisoformat(raw[1])


def main() -> None:
    args = parse_args()
    config, parameters = load_config(args.config)

    records: List[Dict[str, Any]] = []
    for params in parameters:
        outcome = run_momentum_experiment(config, params)
        payload: Dict[str, Any] = {
            "parameter_label": params.label(),
        }
        payload.update(outcome.as_dict())
        records.append(payload)

    frame = pd.DataFrame.from_records(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    else:
        if frame.empty:
            print("No experiment results produced.")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 160):
                print(frame.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
