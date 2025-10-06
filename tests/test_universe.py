from datetime import date
from pathlib import Path

import pandas as pd

from algotrade.config import AppSettings, DataPaths
from algotrade.data.universe import (
    fetch_snp100_members,
    ingest_universe_csv,
    ingest_universe_frame,
    latest_symbols,
    load_universe,
)


def make_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_paths=DataPaths(raw=tmp_path / "raw", cache=tmp_path / "cache"))


def test_ingest_universe_csv_normalizes_symbols(tmp_path):
    csv_path = tmp_path / "snp100.csv"
    csv_path.write_text(
        """effective_date,symbol\n2024-01-01,aapl\n2024-01-01, msft \n2024-02-01,msft\n""",
        encoding="utf-8",
    )

    settings = make_settings(tmp_path)
    output_path = ingest_universe_csv(csv_path, settings=settings)

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert sorted(df.columns) == ["effective_date", "symbol"]
    assert df.iloc[0].symbol == "AAPL"
    assert df.iloc[0].effective_date == date(2024, 1, 1)
    assert df.iloc[1].symbol == "MSFT"
    assert df.iloc[1].effective_date == date(2024, 1, 1)


def test_load_universe_returns_snapshots(tmp_path):
    csv_path = tmp_path / "snp100.csv"
    csv_path.write_text(
        """effective_date,symbol\n2024-01-01,AAPL\n2024-01-01,MSFT\n2024-02-01,MSFT\n2024-02-01,GOOG\n""",
        encoding="utf-8",
    )

    settings = make_settings(tmp_path)
    output_path = ingest_universe_csv(csv_path, settings=settings)

    snapshots = load_universe(output_path)
    assert len(snapshots) == 2
    assert snapshots[0].effective_date == date(2024, 1, 1)
    assert snapshots[0].symbols == frozenset({"AAPL", "MSFT"})
    assert latest_symbols(snapshots) == frozenset({"MSFT", "GOOG"})


def test_ingest_universe_frame_writes_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "effective_date": [date(2024, 3, 1), date(2024, 3, 1)],
            "symbol": ["aapl", " msft"],
        }
    )
    settings = make_settings(tmp_path)
    target = ingest_universe_frame(df, settings=settings, universe_name="custom")
    assert target.exists()
    loaded = pd.read_parquet(target)
    assert loaded.shape == (2, 2)
    assert set(loaded.symbol) == {"AAPL", "MSFT"}


def test_fetch_snp100_members_parses_table(monkeypatch):
    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self) -> None:  # noqa: D401
            """Pretend the response was successful."""

    html = """
    <table>
        <tr><th>Ticker symbol</th><th>Security</th></tr>
        <tr><td>MSFT</td><td>Microsoft</td></tr>
        <tr><td>AAPL</td><td>Apple Inc.</td></tr>
    </table>
    """

    def fake_get(url: str, timeout: int):  # noqa: ARG001
        return FakeResponse(html)

    monkeypatch.setattr("requests.get", fake_get)

    df = fetch_snp100_members(effective_date=date(2024, 3, 1))
    assert list(df.symbol) == ["AAPL", "MSFT"]
    assert df.effective_date.unique().tolist() == [date(2024, 3, 1)]
