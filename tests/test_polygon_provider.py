from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from algotrade.data.contracts import ContractSpec
from algotrade.data.providers.base import HistoricalDataRequest
from algotrade.data.providers.polygon import PolygonDailyEquityProvider


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no failure path in tests
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls: list[SimpleNamespace] = []

    def get(self, url, params=None, timeout=None):
        key = url
        payload = self.payloads.get(key)
        if payload is None:  # pragma: no cover - defensive fallback for unexpected endpoints
            raise AssertionError(
                f"Unexpected endpoint in dummy session: {url}")
        self.calls.append(SimpleNamespace(
            url=url, params=params, timeout=timeout))
        return DummyResponse(payload)

    def close(self):  # pragma: no cover
        return None


def _build_agg_url(ticker: str, start: str, end: str) -> str:
    return f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"


def test_polygon_provider_returns_ibkr_formatted_dataframe(monkeypatch):
    start = "2024-01-04"
    end = "2024-01-06"
    url = _build_agg_url("AAPL", start, end)
    payload = {
        "status": "OK",
        "results": [
            {"t": 1704412800000, "o": 100.0, "h": 105.0, "l": 99.0,
                "c": 104.5, "v": 1_234_567, "vw": 102.0, "n": 123},
            {"t": 1704499200000, "o": 104.5, "h": 106.0, "l": 103.5,
                "c": 105.0, "v": 890_000, "vw": 104.5, "n": 98},
        ],
    }
    session = DummySession({url: payload})
    provider = PolygonDailyEquityProvider(
        api_key="key", session=session, rate_limit_per_minute=0)
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
    call = session.calls[0]
    assert call.params["apiKey"] == "key"
    provider.close()


def test_polygon_provider_handles_empty_results():
    url = _build_agg_url("AAPL", "2024-01-04", "2024-01-05")
    payload = {"status": "OK", "results": []}
    session = DummySession({url: payload})
    provider = PolygonDailyEquityProvider(
        api_key="key", session=session, rate_limit_per_minute=0)
    request = HistoricalDataRequest(
        contract=ContractSpec(symbol="AAPL"),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        duration="1 D",
        bar_size="1 day",
    )

    df = provider.fetch_historical_bars(request)

    assert df.empty
    provider.close()


def test_polygon_provider_raises_on_error():
    url = _build_agg_url("AAPL", "2024-01-04", "2024-01-05")
    payload = {"status": "ERROR", "error": "No access"}
    session = DummySession({url: payload})
    provider = PolygonDailyEquityProvider(
        api_key="key", session=session, rate_limit_per_minute=0)
    request = HistoricalDataRequest(
        contract=ContractSpec(symbol="AAPL"),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        duration="1 D",
        bar_size="1 day",
    )

    with pytest.raises(RuntimeError):
        provider.fetch_historical_bars(request)

    provider.close()
