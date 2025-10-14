from math import nan

import pandas as pd

from algotrade.ui.app import _extract_first_valid_price, _normalize_symbol_for_ibkr


def test_normalize_symbol_for_ibkr_replaces_dot_and_uppercases():
    assert _normalize_symbol_for_ibkr("brk.b") == "BRK B"
    assert _normalize_symbol_for_ibkr(" aapl ") == "AAPL"
    assert _normalize_symbol_for_ibkr("MSFT") == "MSFT"


def test_extract_first_valid_price_skips_invalid_values():
    assert _extract_first_valid_price(None, 0.0, -5.0, nan, pd.NA) is None
    assert _extract_first_valid_price(None, nan, 0.0, 150.25) == 150.25
    assert _extract_first_valid_price(pd.NA, 120.0, 130.0) == 120.0
