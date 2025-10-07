"""CLI entry point for running the mean reversion experiment grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from algotrade.experiments import (
    ExperimentSplit,
    MeanReversionExperimentConfig,
    MeanReversionParameters,
    run_mean_reversion_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to JSON file describing the experiment configuration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write experiment results as CSV.",
    )
    return parser.parse_args()


def load_config(path: Path) -> MeanReversionExperimentConfig:
    data = json.loads(path.read_text())
    store_path = Path(data["store_path"]).expanduser()
    universe = data["symbols"]
    initial_cash = float(data["initial_cash"])
    splits = [
        ExperimentSplit(
            name=split["name"],
            start=_parse_date(split["start"]),
            end=_parse_date(split["end"]),
        )
        for split in data["splits"]
    ]
    parameters = [
        MeanReversionParameters(
            entry_threshold=float(params["entry_threshold"]),
            exit_threshold=float(params["exit_threshold"]),
            max_hold_days=int(params["max_hold_days"]),
            target_position_pct=float(params["target_position_pct"]),
            stop_loss_pct=float(params["stop_loss_pct"]) if params.get(
                "stop_loss_pct") is not None else None,
            lot_size=int(params.get("lot_size", 10)),
        )
        for params in data["parameters"]
    ]
    bar_size = data.get("bar_size", "1d")
    calendar_name = data.get("calendar_name", "XNYS")
    return MeanReversionExperimentConfig(
        store_path=store_path,
        universe=universe,
        initial_cash=initial_cash,
        splits=splits,
        parameters=parameters,
        bar_size=bar_size,
        calendar_name=calendar_name,
    )


def _parse_date(value: str):
    from datetime import date

    return date.fromisoformat(value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_mean_reversion_experiment(config)
    frame = result.to_frame()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    else:
        # Print to stdout in tabular form
        if frame.empty:
            print("No experiment rows produced.")
        else:
            print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
