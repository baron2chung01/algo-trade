import json
from pathlib import Path

import pytest
import requests

from algotrade.config import AppSettings, DataPaths
from algotrade.experiments.vcp import (
    _TECH_UNIVERSE_CACHE,
    _load_technology_universe,
)


def _make_settings(tmp_path: Path) -> AppSettings:
    data_paths = DataPaths(cache=tmp_path / "cache", raw=tmp_path / "raw")
    data_paths.ensure()
    return AppSettings(data_paths=data_paths)


def test_load_technology_universe_scrape_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    monkeypatch.setattr(
        "algotrade.experiments.vcp._fetch_polygon_technology_symbols",
        lambda _settings: ([], ["Polygon API returned no symbols."]),
    )

    monkeypatch.setattr(
        "algotrade.experiments.vcp._scrape_nasdaq_technology_symbols",
        lambda: (["AAPL", "MSFT"], [
                 "Scraped technology symbols from test-source."]),
    )

    symbols, warnings = _load_technology_universe(settings, force_refresh=True)

    assert symbols == ["AAPL", "MSFT"]
    assert any("test-source" in message for message in warnings)

    cache_path = settings.data_paths.cache / _TECH_UNIVERSE_CACHE
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["source"] == "web"
    assert payload["symbols"] == symbols


def test_scrape_nasdaq_from_wikipedia(monkeypatch: pytest.MonkeyPatch) -> None:
    from algotrade.experiments.vcp import _scrape_nasdaq_technology_symbols

    html = """
    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>GICS Sector</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>AAPL</td><td>Information Technology</td></tr>
            <tr><td>MSFT</td><td>Information Technology</td></tr>
            <tr><td>GOOG</td><td>Communication Services</td></tr>
        </tbody>
    </table>
    """

    class FakeResponse:
        def __init__(self, text: str, status_code: int = 200) -> None:
            self.text = text
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

    urls: list[str] = []

    def fake_get(url: str, *args, **kwargs):
        urls.append(url)
        if "wikipedia" in url:
            return FakeResponse(html)
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr("algotrade.experiments.vcp.requests.get", fake_get)

    symbols, warnings = _scrape_nasdaq_technology_symbols()

    assert urls == ["https://en.wikipedia.org/wiki/NASDAQ-100"]
    assert symbols == ["AAPL", "MSFT"]
    assert warnings == [
        "Scraped technology symbols from Wikipedia NASDAQ-100."]


def test_load_technology_universe_prefers_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    cache_path = settings.data_paths.cache / _TECH_UNIVERSE_CACHE
    cache_payload = {
        "symbols": ["NVDA", "TSLA"],
        "source": "web",
        "cached_at": "2024-02-20T00:00:00Z",
    }
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    def _should_not_fetch(_settings):  # pragma: no cover - safety
        raise AssertionError(
            "Polygon fetch should not be invoked when cache is warm")

    monkeypatch.setattr(
        "algotrade.experiments.vcp._fetch_polygon_technology_symbols",
        _should_not_fetch,
    )
    monkeypatch.setattr(
        "algotrade.experiments.vcp._scrape_nasdaq_technology_symbols",
        _should_not_fetch,
    )

    symbols, warnings = _load_technology_universe(settings)

    assert symbols == ["NVDA", "TSLA"]
    assert warnings == []
