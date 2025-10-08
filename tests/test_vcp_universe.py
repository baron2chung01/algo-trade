import json
from pathlib import Path

import pytest

from algotrade.config import AppSettings, DataPaths
from algotrade.experiments.vcp import (
    _LIQUID_UNIVERSE_CACHE,
    _fetch_polygon_liquid_symbols,
    _load_liquid_universe,
)


def _make_settings(tmp_path: Path) -> AppSettings:
    data_paths = DataPaths(cache=tmp_path / "cache", raw=tmp_path / "raw")
    data_paths.ensure()
    return AppSettings(data_paths=data_paths)


def test_fetch_polygon_liquid_symbols_filters_by_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: dict, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    responses = iter(
        [
            FakeResponse(
                {
                    "results": [
                        {"T": "AAPL", "v": 6000, "c": 150},
                        {"T": "LOWV", "v": 1000, "c": 12},
                    ]
                }
            ),
            FakeResponse(
                {
                    "results": [
                        {"T": "AAPL", "v": 5000, "c": 149},
                        {"T": "LOWV", "v": 800, "c": 11},
                    ]
                }
            ),
        ]
    )

    def fake_get(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("algotrade.experiments.vcp.requests.get", fake_get)

    class DummySettings:
        @staticmethod
        def require_polygon_api_key() -> str:
            return "mock"

    symbols, warnings = _fetch_polygon_liquid_symbols(
        DummySettings(),
        min_dollar_volume=1_500_000.0,
        lookback_days=2,
    )

    assert symbols == ["AAPL"]
    assert any("Identified" in message for message in warnings)


def test_fetch_polygon_liquid_symbols_handles_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySettings:
        @staticmethod
        def require_polygon_api_key() -> str:
            return "mock"

    class UnauthorizedResponse:
        status_code = 401

        @staticmethod
        def json() -> dict:
            return {"error": "Invalid API Key"}

    monkeypatch.setattr(
        "algotrade.experiments.vcp.requests.get",
        lambda *args, **kwargs: UnauthorizedResponse(),
    )

    symbols, warnings = _fetch_polygon_liquid_symbols(
        DummySettings(),
        min_dollar_volume=1_500_000.0,
        lookback_days=1,
    )

    assert symbols == []
    assert any("unauthorized" in message.lower() for message in warnings)
    assert any("POLYGON_API_KEY" in message for message in warnings)


def test_load_liquid_universe_prefers_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    cache_path = settings.data_paths.cache / _LIQUID_UNIVERSE_CACHE
    cache_payload = {
        "symbols": ["AAPL", "MSFT"],
        "source": "test",
        "cached_at": "2025-02-01T00:00:00Z",
    }
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    def _should_not_fetch(*args, **kwargs):  # pragma: no cover - safety
        raise AssertionError("Fetch should not be called when cache is warm")

    monkeypatch.setattr(
        "algotrade.experiments.vcp._fetch_polygon_liquid_symbols",
        _should_not_fetch,
    )

    symbols, warnings = _load_liquid_universe(settings)

    assert symbols == ["AAPL", "MSFT"]
    assert warnings == []


def test_load_liquid_universe_force_refresh_falls_back_to_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)

    cache_path = settings.data_paths.cache / _LIQUID_UNIVERSE_CACHE
    cache_payload = {
        "symbols": ["NVDA", "TSLA"],
        "source": "test",
        "cached_at": "2025-02-01T00:00:00Z",
    }
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    monkeypatch.setattr(
        "algotrade.experiments.vcp._fetch_polygon_liquid_symbols",
        lambda *args, **kwargs: ([], ["API unavailable"]),
    )

    symbols, warnings = _load_liquid_universe(settings, force_refresh=True)

    assert symbols == ["NVDA", "TSLA"]
    assert "API unavailable" in warnings
