"""What the model is allowed to say.

The decision comes back as a parsed :class:`TradeStance` and nothing else — not
prose, not an order, not a tool call the trading loop would honour. That is the
containment boundary, and it is worth being explicit about *why* it is shaped
this way:

**A stance carries no size.** There is no quantity field, no leverage field, no
notional. The model expresses direction and conviction; the risk engine turns
that into a position size, and it is the only thing that can. This is not a
policy we enforce with a check that could be forgotten — the model has no
vocabulary for size, so there is no sentence it could emit that would move more
capital than the envelope allows.

**Every stance must name its own invalidation.** A thesis that cannot be wrong
is not a thesis, and requiring the model to state what would falsify it up front
gives the memory layer something to score later. It also tends to improve the
reasoning: a model asked "what would prove you wrong" hedges less.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import SignalAction


class TradeStance(BaseModel):
    """One decision from the model. Direction and conviction only."""

    action: SignalAction = Field(
        description=(
            "LONG to open or hold a long, SHORT to open or hold a short, "
            "CLOSE to exit the current position, HOLD to do nothing."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How strongly the evidence supports this action. Below 0.5 the desk "
            "treats it as not worth acting on. Be honest: an unearned 0.9 costs "
            "real money, and your past calibration is shown back to you."
        ),
    )
    reason: str = Field(
        max_length=240,
        description="One line, for the trade log and the world's event feed.",
    )
    thesis: str = Field(
        max_length=2000,
        description="The full argument: what you saw, and why it implies this action.",
    )
    invalidation: str = Field(
        max_length=600,
        description=(
            "What would prove this wrong — a price level, a data release, a "
            "change in conditions. Required. 'Nothing' is not an answer."
        ),
    )
    horizon_minutes: int = Field(
        ge=1, le=60 * 24 * 14,
        description="How long this stance is expected to remain valid.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs or dataset names this rests on. Empty if price action only.",
    )
    stop_distance_pct: float | None = Field(
        default=None, gt=0.0,
        description=(
            "Where your stop belongs, as a percentage move against you. Set it "
            "where the idea is actually wrong, not where it feels comfortable. "
            "This does NOT change how much you risk — a wider stop simply buys "
            "a smaller position for the same risk budget. Leave null to use the "
            "desk default."
        ),
    )
    take_profit_pct: float | None = Field(
        default=None, gt=0.0,
        description=(
            "Where you would take the trade off, as a percentage move in your "
            "favour. Be realistic about what the timeframe can deliver: a "
            "target many ATR away never fills. Leave null for the desk default."
        ),
    )

    @property
    def actionable(self) -> bool:
        return self.action is not SignalAction.HOLD
