from datetime import date, datetime, timezone
from pathlib import Path

from algotrade.config import AppSettings, DataPaths
from algotrade.data.ingest import ingest_quantconnect_daily
from algotrade.data.providers.quantconnect import QuantConnectDailyEquityProvider


class StubQuantConnectProvider:
    def __init__(self, frames_by_symbol):
        self.frames_by_symbol = frames_by_symbol
        self.closed = False

    def fetch_historical_bars(self, request):
        symbol = request.contract.symbol
        return self.frames_by_symbol.get(symbol, self.frames_by_symbol.get(symbol.upper()))

    def close(self) -> None:
        self.closed = True


class DummySettings(AppSettings):
    def __init__(self, tmp_path: Path):
        super().__init__(
            QUANTCONNECT_USER_ID="user",
            QUANTCONNECT_API_TOKEN="token",
            data_paths=DataPaths(raw=tmp_path / "raw", cache=tmp_path / "cache"),
        )


def test_ingest_quantconnect_daily_writes_parquet(tmp_path, monkeypatch):
    import pandas as pd

    frames = {
        "AAPL": pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 5, tzinfo=timezone.utc)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.5],
                "close": [100.5],
                "volume": [1_000],
                "average": [100.2],
                "bar_count": [42],
            }
        )
    }

    settings = DummySettings(tmp_path)
    provider = StubQuantConnectProvider(frames)

    result = ingest_quantconnect_daily(
        ["AAPL"],
        date(2024, 1, 5),
        date(2024, 1, 5),
        settings=settings,
        provider=provider,
    )

    assert len(result.paths) == 1
    assert result.paths[0].exists()
    df = pd.read_parquet(result.paths[0])
    assert df.iloc[0]["close"] == 100.5
    assert result.frames["AAPL"].equals(frames["AAPL"])
    assert provider.closed is False  # Provided provider should not be closed by ingest


def test_ingest_quantconnect_daily_respects_empty_frames(tmp_path):
    import pandas as pd

    frames = {
        "AAPL": pd.DataFrame(),
    }
    settings = DummySettings(tmp_path)
    provider = StubQuantConnectProvider(frames)

    result = ingest_quantconnect_daily(
        ["AAPL"],
        date(2024, 1, 5),
        date(2024, 1, 5),
        settings=settings,
        provider=provider,
    )

    assert result.paths == []
    assert result.frames == {}


def test_ingest_quantconnect_daily_dry_run(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 5, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1_000],
            "average": [100.2],
            "bar_count": [42],
        }
    )

    settings = DummySettings(tmp_path)
    provider = StubQuantConnectProvider({"AAPL": frame})

    result = ingest_quantconnect_daily(
        ["AAPL"],
        date(2024, 1, 5),
        date(2024, 1, 5),
        settings=settings,
        provider=provider,
        write=False,
    )

    assert result.paths == []
    assert "AAPL" in result.frames
    expected_path = settings.data_paths.raw / "quantconnect" / "daily" / "AAPL_1d.parquet"
    assert not expected_path.exists()