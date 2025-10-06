"""Core package for the algo-trade research and execution stack."""

from importlib.metadata import version

__all__ = ["__version__"]

try:
    __version__ = version("algo-trade")
except Exception:  # pragma: no cover - package not installed in dev mode yet.
    __version__ = "0.0.0"
