from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from algotrade.data.contracts import ContractSpec
from algotrade.data.providers.base import HistoricalDataRequest
from algotrade.data.providers.polygon import PolygonDailyBarsProvider


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no failure path in tests
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[SimpleNamespace] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, params=params, timeout=timeout))
        return DummyResponse(self.payload)

    def close(self):  # pragma: no cover
        return None


def test_polygon_provider_returns_ibkr_formatted_dataframe():
    payload = {
        "results": [
            {
                "t": 1704412800000,  # 2024-01-05 UTC in ms
                "o": 100.0,
                "h": 105.0,
                "l": 99.0,
                "c": 104.5,
                "v": 1_234_567,
                "vw": 102.0,
                "n": 123,
            },
            {
                "t": 1704499200000,  # 2024-01-06 UTC
                "o": 104.5,
                "h": 106.0,
                "l": 103.5,
                "c": 105.0,
                "v": 890_000,
                "vw": 104.5,
                "n": 98,
            },
        ]
    }
    session = DummySession(payload)
    provider = PolygonDailyBarsProvider(api_key="test", session=session)
    request = HistoricalDataRequest(
        contract=ContractSpec(symbol="AAPL"),
        end=datetime(2024, 1, 6, tzinfo=timezone.utc),
        duration="2 D",
        bar_size="1 day",
    )

    df = provider.fetch_historical_bars(request)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "average",
        "bar_count",
    ]
    assert len(df) == 2
    assert df.iloc[0]["close"] == pytest.approx(104.5)
    assert df.iloc[1]["bar_count"] == 98
    assert session.calls[0].params["adjusted"] == "true"


def test_polygon_provider_rejects_unsupported_duration():
    provider = PolygonDailyBarsProvider(api_key="test", session=DummySession({}))
    request = HistoricalDataRequest(
        contract=ContractSpec(symbol="AAPL"),
        end=datetime(2024, 1, 6, tzinfo=timezone.utc),
        duration="1 F",
        bar_size="1 day",
    )

    with pytest.raises(ValueError):
        provider.fetch_historical_bars(request)
