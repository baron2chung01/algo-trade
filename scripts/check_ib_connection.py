"""Quick connectivity check to IBKR TWS using ib_insync."""

from __future__ import annotations

from datetime import datetime

from ib_insync import IB

from algotrade.config import AppSettings
from algotrade.data.contracts import ContractSpec, SecurityType
from algotrade.data.providers.base import HistoricalDataRequest
from algotrade.data.providers.ibkr import IBKRHistoricalDataProvider


def main() -> None:
    settings = AppSettings()
    ib = IB()
    try:
        ib.connect(settings.ibkr_host, settings.ibkr_port,
                   clientId=settings.ibkr_client_id)
        if not ib.isConnected():
            raise RuntimeError("Failed to establish IBKR connection")
        spec = ContractSpec(symbol="AAPL", security_type=SecurityType.STOCK)
        request = HistoricalDataRequest(
            contract=spec,
            end=datetime.now(),
            duration="1 D",
            bar_size="1 hour",
            what_to_show="TRADES",
            use_rth=True,
        )

        provider = IBKRHistoricalDataProvider(ib)
        df = provider.fetch_historical_bars(request)
        print(df.head())
        print(f"Retrieved {len(df)} bars from IBKR for {spec.symbol}.")
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
