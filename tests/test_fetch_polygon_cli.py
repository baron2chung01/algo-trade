from datetime import date, timedelta
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from algotrade.config import AppSettings, DataPaths
from algotrade.data.universe import ingest_universe_frame

SCRIPT_PATH = Path(__file__).resolve(
).parents[1] / "scripts" / "fetch_quantconnect_daily.py"
_module_spec = util.spec_from_file_location(
    "fetch_quantconnect_daily", SCRIPT_PATH)
assert _module_spec and _module_spec.loader
cli = util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(cli)


class DummySettings(AppSettings):
    def __init__(self, tmp_path: Path):
        super().__init__(
            QUANTCONNECT_USER_ID="1",
            QUANTCONNECT_API_TOKEN="token",
            data_paths=DataPaths(raw=tmp_path / "raw",
                                 cache=tmp_path / "cache"),
        )


def test_compute_date_range_defaults_to_years_window():
    args = SimpleNamespace(start=None, end="2024-01-10", years=3)
    start, end = cli.compute_date_range(args)
    assert end == date(2024, 1, 10)
    assert start == date(2024, 1, 10) - timedelta(days=3 * 365)


def test_resolve_symbols_prefers_universe(tmp_path):
    settings = DummySettings(tmp_path)
    df = pd.DataFrame(
        {
            "effective_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "symbol": ["AAPL", "MSFT"],
        }
    )
    universe_path = ingest_universe_frame(
        df, settings=settings, universe_name="snp100")

    base_args = {
        "symbols": [],
        "effective_date": None,
        "universe": "snp100",
        "universe_file": None,
    }

    resolved = cli.resolve_symbols(SimpleNamespace(**base_args), settings)
    assert resolved == ["AAPL", "MSFT"]

    base_args["universe"] = None
    base_args["universe_file"] = universe_path
    base_args["effective_date"] = "2024-01-01"
    resolved_explicit = cli.resolve_symbols(
        SimpleNamespace(**base_args), settings)
    assert resolved_explicit == ["AAPL", "MSFT"]
