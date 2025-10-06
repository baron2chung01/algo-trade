"""Instrument and contract specification helpers aligned with IBKR TWS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SecurityType(str, Enum):
    """Enumerate supported IBKR security types."""

    STOCK = "STK"
    ETF = "ETF"
    FUTURE = "FUT"
    OPTION = "OPT"
    FOREX = "CASH"


@dataclass(slots=True)
class ContractSpec:
    """Lightweight contract descriptor mirroring IBKR contract fields.

    Attributes
    ----------
    symbol:
        Underlying symbol or ticker, e.g., ``"AAPL"``.
    security_type:
        IBKR security type code. Defaults to ``SecurityType.STOCK``.
    currency:
        Trading currency, e.g., ``"USD"``.
    exchange:
        Routing exchange (e.g., ``"SMART"`` for SMART routing).
    primary_exchange:
        Primary listing exchange (e.g., ``"NASDAQ"``).
    """

    symbol: str
    security_type: SecurityType = SecurityType.STOCK
    currency: str = "USD"
    exchange: str = "SMART"
    primary_exchange: str | None = None

    def as_ibkr_kwargs(self) -> dict[str, str]:
        """Return keyword arguments compatible with ``ib_insync.Stock`` etc."""

        kwargs: dict[str, str] = {
            "symbol": self.symbol,
            "secType": self.security_type.value,
            "currency": self.currency,
            "exchange": self.exchange,
        }
        if self.primary_exchange:
            kwargs["primaryExchange"] = self.primary_exchange
        return kwargs
