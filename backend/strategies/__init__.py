"""Strategy registry."""

from __future__ import annotations

from .base import Series, Strategy
from .hybrid import HybridStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .scalping import ScalpingStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "scalping": ScalpingStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "hybrid": HybridStrategy,
}


def create_strategy(name: str, params: dict[str, float] | None = None) -> Strategy:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return cls(params)


__all__ = [
    "Series",
    "Strategy",
    "REGISTRY",
    "create_strategy",
    "ScalpingStrategy",
    "VolatilityBreakoutStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "HybridStrategy",
]
