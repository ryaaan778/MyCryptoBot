"""The model as a strategy, so the rest of the desk needs no special case.

:class:`LLMStrategy` implements the same ``Strategy`` interface the five
hand-written strategies do. Everything downstream — the orchestrator, the risk
engine, the execution engines, the portfolio, the voxel world — treats it as
just another strategy, because from where they sit it is one.

The one asymmetry is timing, and it is handled by separation rather than by
making the interface async: ``compute`` is pure and instant, reading whatever
stance the brain currently holds, while the brain refreshes that stance on its
own schedule in the background. If there is no live stance, this holds. Holding
is a correct and cheap answer, which is what makes the degradation safe: a dead
API key, an expired card, a network partition — all of them mean "this bot stops
trading", never "this bot trades badly".
"""

from __future__ import annotations

import logging

from ..models import Position, PositionSide, Signal, SignalAction
from ..strategies.base import Series, Strategy
from .brain import LLMBrain
from .schema import TradeStance

logger = logging.getLogger("jojo.llm.strategy")


class LLMStrategy(Strategy):
    """Trades the stance held by an :class:`LLMBrain`."""

    name = "llm"
    #: Enough history for the 96-bar range and vol figures in the prompt.
    min_bars = 100

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.brain: LLMBrain | None = None
        #: Below this, a stance is treated as an opinion rather than a trade.
        self.min_confidence = self.param("llm_min_confidence", 0.5)

    def bind_brain(self, brain: LLMBrain) -> None:
        self.brain = brain
        self.min_confidence = brain.min_confidence

    # ---- decision ----------------------------------------------------------

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        if self.brain is None:
            return self.hold(bot_id, symbol, "no model attached")

        stance = self.brain.current()
        if stance is None:
            reason = self.brain.last_error or "waiting on the model"
            return self.hold(bot_id, symbol, reason)

        if stance.confidence < self.min_confidence:
            return self.hold(
                bot_id, symbol,
                f"{stance.action.value} at {stance.confidence:.2f} is below the "
                f"{self.min_confidence:.2f} bar",
            )

        action = self._resolve(stance, position)
        if action is SignalAction.HOLD:
            return self.hold(bot_id, symbol, f"already positioned: {stance.reason}")

        return Signal(
            bot_id=bot_id, symbol=symbol, action=action,
            confidence=float(stance.confidence),
            reason=stance.reason, strategy=self.name,
            indicators={
                "llm_confidence": round(float(stance.confidence), 4),
                "llm_horizon_min": float(stance.horizon_minutes),
                "llm_sources": float(len(stance.sources)),
            },
        )

    @staticmethod
    def _resolve(stance: TradeStance, position: Position | None) -> SignalAction:
        """Turn a standing stance into the action that gets us there from here.

        A stance is a description of where the bot wants to be, not an event. So
        LONG while already long is not a second entry — it is "stay". Without
        this the bot would re-fire an entry every tick for as long as it agreed
        with itself, which is both an execution bug and an expensive one.
        """
        flat = position is None or position.quantity <= 0
        if stance.action is SignalAction.CLOSE:
            return SignalAction.HOLD if flat else SignalAction.CLOSE
        if stance.action is SignalAction.HOLD:
            return SignalAction.HOLD
        if flat:
            return stance.action

        assert position is not None
        holding_long = position.side is PositionSide.LONG
        wants_long = stance.action is SignalAction.LONG
        if holding_long == wants_long:
            return SignalAction.HOLD        # already where it wants to be
        return SignalAction.CLOSE           # reverse: exit first, re-enter next tick
