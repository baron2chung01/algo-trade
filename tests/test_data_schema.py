from datetime import datetime, timezone

import pandas as pd

from algotrade.data.schemas import IBKRBar, IBKRBarDataFrame


def test_bar_to_dataframe_roundtrip() -> None:
    bar = IBKRBar(
        timestamp=datetime(2024, 1, 2, 15, 30, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1_000,
        average=100.3,
        bar_count=42,
    )

    df = IBKRBarDataFrame.from_bars([bar])
    assert list(df.columns) == list(IBKRBarDataFrame.columns)
    assert len(df) == 1
    loaded = IBKRBarDataFrame.ensure_schema(df)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.iloc[0]["close"] == 100.5
