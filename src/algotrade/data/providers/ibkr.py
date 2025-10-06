"""Interactive Brokers TWS historical data provider implementation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from ib_insync import IB
from ib_insync.contract import Contract

from ..contracts import ContractSpec
from ..schemas import IBKRBar, IBKRBarDataFrame
from .base import BaseDataProvider, HistoricalDataRequest

IB_DATETIME_FMT = "%Y%m%d %H:%M:%S"


def _to_ibkr_contract(spec: ContractSpec) -> Contract:
    """Convert a :class:`ContractSpec` to an `ib_insync` :class:`Contract`."""

    return Contract(**spec.as_ibkr_kwargs())


def _format_end_datetime(end: datetime) -> str:
    """Format ``end`` per IBKR historical API requirements."""

    return end.strftime(IB_DATETIME_FMT)


class IBKRHistoricalDataProvider(BaseDataProvider):
    """Fetch historical bars through an ``ib_insync.IB`` connection."""

    def __init__(self, ib: IB):
        self._ib = ib

    def fetch_historical_bars(self, request: HistoricalDataRequest) -> pd.DataFrame:
        contract = _to_ibkr_contract(request.contract)
        self._ib.qualifyContracts(contract)
        bars = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime=_format_end_datetime(request.end),
            durationStr=request.duration,
            barSizeSetting=request.bar_size,
            whatToShow=request.what_to_show,
            useRTH=request.use_rth,
            formatDate=1,
        )
        parsed = [IBKRBar.from_ibkr_bar(bar) for bar in bars]
        return IBKRBarDataFrame.from_bars(parsed)
