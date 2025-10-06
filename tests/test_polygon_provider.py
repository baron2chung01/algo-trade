from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from algotrade.data.contracts import ContractSpec
from algotrade.data.providers.base import HistoricalDataRequest
from algotrade.data.providers.quantconnect import QuantConnectDailyEquityProvider


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


def test_quantconnect_provider_returns_ibkr_formatted_dataframe():
    payload = {
        "success": True,
        "data": [
            {
                "time": "2024-01-05T00:00:00Z",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.5,
                "volume": 1_234_567,
                "vwap": 102.0,
                "trades": 123,
            },
            {
                "time": "2024-01-06T00:00:00Z",
                "open": 104.5,
                "high": 106.0,
                "low": 103.5,
                "close": 105.0,
                "volume": 890_000,
                "vwap": 104.5,
                "trades": 98,
            },
        ],
    }
    session = DummySession(payload)
    provider = QuantConnectDailyEquityProvider(user_id="1", api_token="token", session=session)
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
    params = session.calls[0].params
    assert params["userId"] == "1"
    assert params["ticker"] == "AAPL"
    assert params["resolution"] == "Daily"


def test_quantconnect_provider_rejects_unsupported_duration():
    provider = QuantConnectDailyEquityProvider(user_id="1", api_token="token", session=DummySession({}))
    request = HistoricalDataRequest(
        contract=ContractSpec(symbol="AAPL"),
        end=datetime(2024, 1, 6, tzinfo=timezone.utc),
        duration="1 F",
        bar_size="1 day",
    )

    with pytest.raises(ValueError):
        provider.fetch_historical_bars(request)
