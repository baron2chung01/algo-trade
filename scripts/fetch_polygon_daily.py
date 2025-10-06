#!/usr/bin/env python3
"""Backward-compatible shim that proxies to the QuantConnect daily downloader."""

from __future__ import annotations

import warnings

from .fetch_quantconnect_daily import (
    compute_date_range,
    main as _quantconnect_main,
    parse_args,
    resolve_symbols,
)


__all__ = ["parse_args", "resolve_symbols", "compute_date_range", "main"]


def main() -> None:
    warnings.warn(
        "scripts.fetch_polygon_daily is deprecated; use scripts.fetch_quantconnect_daily instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _quantconnect_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
