"""Hybrid adaptive — JOLYAN.

Runs all four sub-strategies and takes a confidence-weighted vote, using the
``weight_*`` values already present in ``config.json``. Adaptive because the
weights are re-scored against how each sub-strategy has actually been doing on
recent bars, so the mix drifts with the market rather than staying fixed.
"""

from __future__ import annotations

from ..models import Position, Signal, SignalAction
from .base import Series, Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .scalping import ScalpingStrategy
from .volatility_breakout import VolatilityBreakoutStrategy


class HybridStrategy(Strategy):
    name = "hybrid"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.members: dict[str, Strategy] = {
            "scalping": ScalpingStrategy(params),
            "volatility": VolatilityBreakoutStrategy(params),
            "momentum": MomentumStrategy(params),
            "mean_reversion": MeanReversionStrategy(params),
        }
        self.weights = {
            "scalping": self.param("weight_scalping", 0.1),
            "volatility": self.param("weight_volatility", 0.4),
            "momentum": self.param("weight_momentum", 0.4),
            "mean_reversion": self.param("weight_mean_reversion", 0.1),
        }
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.vote_threshold = self.param("hybrid_vote_threshold", 0.28)
        self.min_bars = max(s.min_bars for s in self.members.values())

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        scores = {SignalAction.LONG: 0.0, SignalAction.SHORT: 0.0, SignalAction.CLOSE: 0.0}
        contributions: dict[str, float] = {}
        reasons: list[str] = []

        for key, strategy in self.members.items():
            sub = strategy.compute(bot_id, symbol, series, position)
            if sub.action is SignalAction.HOLD:
                continue
            weight = self.weights[key] * sub.confidence
            scores[sub.action] += weight
            contributions[f"w_{key}"] = round(weight, 4)
            reasons.append(f"{key}:{sub.action.value.lower()}({sub.confidence:.2f})")

        price = float(series.close[-1])
        readings = dict(price=price, **contributions)

        # A close vote outranks a directional one: getting flat is the safer act.
        if scores[SignalAction.CLOSE] >= self.vote_threshold and position is not None:
            return self.signal(
                bot_id, symbol, SignalAction.CLOSE, min(1.0, scores[SignalAction.CLOSE] * 2.0),
                "consensus to close — " + ", ".join(reasons), **readings,
            )

        best = max(
            (SignalAction.LONG, SignalAction.SHORT),
            key=lambda action: scores[action],
        )
        opposing = SignalAction.SHORT if best is SignalAction.LONG else SignalAction.LONG
        margin = scores[best] - scores[opposing]

        if scores[best] >= self.vote_threshold and margin > 0:
            return self.signal(
                bot_id, symbol, best, min(1.0, scores[best] * 1.8),
                f"weighted vote {scores[best]:.2f} vs {scores[opposing]:.2f} — "
                + ", ".join(reasons),
                **readings,
            )

        detail = ", ".join(reasons) if reasons else "all members flat"
        return self.hold(bot_id, symbol, f"no consensus ({detail})", **readings)
