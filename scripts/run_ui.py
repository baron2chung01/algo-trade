"""Launch the interactive UI for visualizing mean reversion strategy signals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from algotrade.ui import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000).",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=None,
        help="Override the default Parquet store path for bar data.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.store_path is not None:
        os.environ["ALGO_TRADE_STORE_PATH"] = str(args.store_path.expanduser())

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
