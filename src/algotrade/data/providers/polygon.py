"""Compatibility module mapping Polygon provider to QuantConnect implementation."""

from __future__ import annotations

from .quantconnect import QuantConnectDailyEquityProvider as PolygonDailyBarsProvider

__all__ = ["PolygonDailyBarsProvider"]
