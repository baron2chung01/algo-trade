from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from algotrade.config import AppSettings, DataPaths
from algotrade.data.ingest import (
    ingest_polygon_daily,
    ingest_quantconnect_daily,
    update_polygon_daily_incremental,
    _RateLimiter,
)


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
            POLYGON_API_KEY="polygon",
            data_paths=DataPaths(raw=tmp_path / "raw",
                                 cache=tmp_path / "cache"),
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
    expected_path = settings.data_paths.raw / \
        "quantconnect" / "daily" / "AAPL_1d.parquet"
    assert not expected_path.exists()


def test_ingest_polygon_daily_writes_parquet(tmp_path):
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

    result = ingest_polygon_daily(
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


def test_ingest_polygon_daily_dry_run(tmp_path):
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

    result = ingest_polygon_daily(
        ["AAPL"],
        date(2024, 1, 5),
        date(2024, 1, 5),
        settings=settings,
        provider=provider,
        write=False,
    )

    assert result.paths == []
    assert "AAPL" in result.frames
    expected_path = settings.data_paths.raw / \
        "polygon" / "daily" / "AAPL_1d.parquet"
    assert not expected_path.exists()


def test_update_polygon_daily_incremental_handles_rate_limit(tmp_path):
    import pandas as pd

    class RateLimitedProvider:
        def __init__(self):
            self.calls = 0

        def fetch_historical_bars(self, request):  # noqa: ARG002
            self.calls += 1
            response = requests.Response()
            response.status_code = 429
            response.reason = "Too Many Requests"
            raise requests.HTTPError(response=response)

    class PreloadedStore:
        def __init__(self, existing_frame):
            self.existing_frame = existing_frame

        def load(self, symbol, bar_size):  # noqa: ARG002
            return self.existing_frame

        def save(self, symbol, bar_size, df):  # noqa: ARG002
            raise AssertionError("save should not be called when rate limited")

    settings = DummySettings(tmp_path)
    existing = pd.DataFrame(
        {
            "timestamp": [
                datetime.now(timezone.utc) - timedelta(days=2)
            ],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1_000],
            "average": [100.2],
            "bar_count": [42],
        }
    )
    provider = RateLimitedProvider()
    store = PreloadedStore(existing)

    report = update_polygon_daily_incremental(
        ["AAPL", "MSFT"],
        settings=settings,
        provider=provider,
        store=store,
    )

    assert provider.calls == 1
    assert report.updated == {}
    assert report.written_paths == []
    assert report.skipped == ["AAPL"]
    assert report.warnings == [
        "Polygon rate limit reached while downloading daily history. Wait about a minute and try again."
    ]


def test_rate_limiter_enforces_window():
    class FakeClock:
        def __init__(self):
            self.current = 0.0

        def now(self):
            return self.current

        def advance(self, seconds):
            self.current += seconds

    clock = FakeClock()
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock.advance(seconds)

    limiter = _RateLimiter(2, now=clock.now, sleeper=fake_sleep)

    limiter.before_request()
    limiter.record_request()
    clock.advance(10)
    limiter.before_request()
    limiter.record_request()
    clock.advance(10)
    limiter.before_request()
    limiter.record_request()

    assert sleep_calls == [40.0]

    clock.advance(61)
    limiter.before_request()
    limiter.record_request()

    assert sleep_calls == [40.0]


def test_update_polygon_incremental_uses_rate_limiter(monkeypatch, tmp_path):
    import pandas as pd

    limiter_calls: dict[str, int] = {"before": 0, "record": 0}

    class SpyRateLimiter:
        def __init__(self, limit):  # noqa: ARG002
            pass

        def before_request(self):
            limiter_calls["before"] += 1

        def record_request(self):
            limiter_calls["record"] += 1

    monkeypatch.setattr("algotrade.data.ingest._RateLimiter", SpyRateLimiter)

    class Provider:
        def __init__(self):
            self.calls = 0

        def fetch_historical_bars(self, request):  # noqa: ARG002
            self.calls += 1
            return pd.DataFrame(
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

    class Store:
        def __init__(self):
            self.saved = []

        def load(self, symbol, bar_size):  # noqa: ARG002
            import pandas as pd

            return pd.DataFrame(columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "average",
                "bar_count",
            ])

        def save(self, symbol, bar_size, df):  # noqa: ARG002
            path = tmp_path / f"{symbol}_{bar_size}.parquet"
            self.saved.append((symbol, bar_size, df))
            return path

    settings = DummySettings(tmp_path)
    provider = Provider()
    store = Store()

    report = update_polygon_daily_incremental(
        ["AAPL", "MSFT"],
        settings=settings,
        provider=provider,
        store=store,
        rate_limit_per_minute=5,
    )

    assert limiter_calls == {"before": 2, "record": 2}
    assert provider.calls == 2
    assert len(report.updated) == 2
