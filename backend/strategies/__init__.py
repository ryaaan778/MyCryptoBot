"""Strategy registry."""

from __future__ import annotations

from .base import Series, Strategy
from .hybrid import HybridStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .scalping import ScalpingStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

# Imported lazily inside the factory: the LLM path pulls in the decision
# stack, and a desk running only hand-written strategies should not pay for
# it — nor break if the provider SDK is absent.

REGISTRY: dict[str, type[Strategy]] = {
    "scalping": ScalpingStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "hybrid": HybridStrategy,
}

#: Strategies that need more than parameters to construct. ``BotAgent``
#: builds these itself; ``create_strategy`` returns one that holds until
#: it is bound.
DEFERRED: frozenset[str] = frozenset({"llm"})


def create_strategy(name: str, params: dict[str, float] | None = None) -> Strategy:
    if name == "llm":
        from ..llm.strategy import LLMStrategy
        return LLMStrategy(params)
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; available: "
            f"{', '.join(sorted(set(REGISTRY) | DEFERRED))}"
        ) from None
    return cls(params)


__all__ = [
    "Series",
    "Strategy",
    "REGISTRY",
    "DEFERRED",
    "create_strategy",
    "ScalpingStrategy",
    "VolatilityBreakoutStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "HybridStrategy",
]
