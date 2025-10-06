"""Commission and slippage models for the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CommissionModel(Protocol):
    """Return the commission (positive cash cost) for a proposed fill."""

    def calculate(self, symbol: str, quantity: int, price: float) -> float:
        """Compute commission for the trade at ``price`` per share."""


class SlippageModel(Protocol):
    """Return an adjusted fill price based on the requested trade."""

    def adjust(self, symbol: str, quantity: int, price: float) -> float:
        """Return the adjusted execution price."""


@dataclass(slots=True)
class ZeroCommission(CommissionModel):
    """Commission model that charges nothing."""

    def calculate(self, symbol: str, quantity: int, price: float) -> float:  # noqa: D401
        return 0.0


@dataclass(slots=True)
class InteractiveBrokersCommission(CommissionModel):
    """Replicate IBKR US stock commission schedule (simplified)."""

    per_share: float = 0.005
    minimum: float = 1.0

    def calculate(self, symbol: str, quantity: int, price: float) -> float:
        shares = abs(quantity)
        if shares == 0:
            return 0.0
        commission = shares * self.per_share
        return max(self.minimum, commission)


@dataclass(slots=True)
class ZeroSlippage(SlippageModel):
    """No slippage adjustment."""

    def adjust(self, symbol: str, quantity: int, price: float) -> float:  # noqa: D401
        return price


@dataclass(slots=True)
class BpsSlippage(SlippageModel):
    """Apply symmetric basis-point slippage to the fill price."""

    bps: float = 5.0  # 5 bps default

    def adjust(self, symbol: str, quantity: int, price: float) -> float:
        if quantity == 0:
            return price
        direction = 1 if quantity > 0 else -1
        return price * (1 + direction * (self.bps / 10_000))
