"""The decision loop that actually runs the model.

The brain is deliberately *not* driven by the bot's tick. A tick is a few
seconds; a reasoning call with web search is tens of seconds and costs money.
Binding them together would give you a desk that either stutters or bankrupts
you on API spend, depending which way you resolved the conflict.

So the brain keeps its own cadence. It holds a current stance with an expiry;
the trading loop reads that stance every tick and acts on it. When the stance
ages out and the cadence allows, the brain refreshes it in the background. The
practical consequences are worth stating plainly:

- **The desk never blocks on the model.** A slow or dead provider degrades to
  "no current stance", which reads as HOLD. It cannot wedge the trading loop.
- **Spend is bounded before the call, not regretted after it.** A daily call
  budget and a minimum interval are both checked up front.
- **A stale opinion is not acted on.** A stance from six hours ago about a
  market that has since moved 8% is worse than no opinion, so stances expire on
  the horizon the model itself gave them, capped by ``stance_ttl_sec``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..models import Position, SignalAction
from ..strategies.base import Series
from .client import DecisionClient, LLMUnavailable, Usage
from .memory import LLMMemory, memory_path
from .prompts import SYSTEM_PROMPT, account_block, build_prompt
from .schema import TradeStance

logger = logging.getLogger("jojo.llm.brain")


@dataclass
class AccountContext:
    """The account facts a decision should not be made without."""

    equity: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: int = 0
    max_positions: int = 0
    realized_pnl: float = 0.0
    win_rate: float = 0.0
    trades: int = 0

    def render(self) -> str:
        return account_block(
            equity=self.equity, drawdown_pct=self.drawdown_pct,
            open_positions=self.open_positions, max_positions=self.max_positions,
            realized_pnl=self.realized_pnl, win_rate=self.win_rate,
            trades=self.trades,
        )


@dataclass
class HeldStance:
    """A stance plus when it stops being worth believing."""

    stance: TradeStance
    taken_at: float
    expires_at: float

    def fresh(self, now: float | None = None) -> bool:
        return (now if now is not None else time.monotonic()) < self.expires_at

    def age_sec(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.taken_at


class LLMBrain:
    """One model-backed decision maker, for one bot on one market."""

    def __init__(
        self,
        *,
        bot_id: str,
        bot_name: str,
        client: DecisionClient,
        data_dir: str | Path = "data",
        persona: str = "",
        decide_every_sec: float = 300.0,
        stance_ttl_sec: float = 1800.0,
        min_confidence: float = 0.5,
        daily_call_budget: int = 500,
        memory_window: int = 20,
        web_search: bool = True,
    ) -> None:
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.client = client
        self.persona = persona
        self.decide_every_sec = max(1.0, float(decide_every_sec))
        self.stance_ttl_sec = max(1.0, float(stance_ttl_sec))
        self.min_confidence = min_confidence
        self.daily_call_budget = max(0, int(daily_call_budget))
        self.web_search = web_search

        self.memory = LLMMemory(memory_path(data_dir, bot_id), window=memory_window)
        self.held: HeldStance | None = None
        self.last_error: str | None = None
        self.last_attempt_at: float = float("-inf")

        self._budget_day = time.gmtime().tm_yday
        self._calls_today = 0
        self._lock = asyncio.Lock()
        self._inflight: asyncio.Task | None = None

    # ---- budget ------------------------------------------------------------

    @property
    def usage(self) -> Usage:
        return self.client.usage

    def _roll_budget(self) -> None:
        today = time.gmtime().tm_yday
        if today != self._budget_day:
            self._budget_day = today
            self._calls_today = 0

    @property
    def budget_remaining(self) -> int:
        self._roll_budget()
        return max(0, self.daily_call_budget - self._calls_today)

    # ---- cadence -----------------------------------------------------------

    def due(self, now: float | None = None) -> bool:
        """Is it time to ask the model again?"""
        now = now if now is not None else time.monotonic()
        if self.budget_remaining <= 0:
            return False
        if now - self.last_attempt_at < self.decide_every_sec:
            return False
        return self.held is None or not self.held.fresh(now)

    def current(self, now: float | None = None) -> TradeStance | None:
        """The stance to trade on right now, or None if there isn't a live one."""
        if self.held is None or not self.held.fresh(now):
            return None
        return self.held.stance

    # ---- the call ----------------------------------------------------------

    async def refresh(
        self,
        *,
        symbol: str,
        timeframe: str,
        series: Series,
        position: Position | None,
        account: AccountContext,
    ) -> TradeStance | None:
        """Ask the model for a new stance. Returns None on any failure."""
        async with self._lock:
            if self.budget_remaining <= 0:
                self.last_error = "daily call budget exhausted"
                return None
            self.last_attempt_at = time.monotonic()
            self._calls_today += 1

            prompt = build_prompt(
                bot_name=self.bot_name, symbol=symbol, timeframe=timeframe,
                series=series, position=position, account=account.render(),
                memory=self.memory.render(), persona=self.persona,
                web_search=self.web_search,
            )

            try:
                stance = await self.client.decide(system=SYSTEM_PROMPT, prompt=prompt)
            except LLMUnavailable as exc:
                self.last_error = str(exc)
                logger.warning("%s: %s", self.bot_name, exc)
                return None
            except Exception as exc:            # pragma: no cover - defensive
                self.last_error = f"unexpected failure: {exc}"
                logger.exception("%s: brain refresh failed", self.bot_name)
                return None

            if stance is None:
                self.last_error = "the model returned no stance"
                return None

            self.last_error = None
            now = time.monotonic()
            ttl = min(self.stance_ttl_sec, max(60.0, stance.horizon_minutes * 60.0))
            self.held = HeldStance(stance=stance, taken_at=now, expires_at=now + ttl)

            last_price = float(series.close[-1]) if len(series) else 0.0
            if stance.actionable:
                self.memory.record(self.bot_id, symbol, stance, price=last_price)
            return stance

    def refresh_in_background(self, **kwargs) -> None:
        """Kick off a refresh without making the caller wait for it."""
        if self._inflight is not None and not self._inflight.done():
            return
        self._inflight = asyncio.create_task(
            self._guarded_refresh(**kwargs), name=f"llm-brain-{self.bot_id}"
        )

    async def _guarded_refresh(self, **kwargs) -> None:
        try:
            await self.refresh(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:                       # pragma: no cover - defensive
            logger.exception("%s: background refresh failed", self.bot_name)

    async def aclose(self) -> None:
        task, self._inflight = self._inflight, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ---- outcomes ----------------------------------------------------------

    def record_outcome(self, pnl: float, note: str = "") -> None:
        """Close the loop: attach a realised P&L to the decision that caused it."""
        self.memory.resolve_latest(pnl, note=note)

    # ---- introspection -----------------------------------------------------

    def snapshot(self) -> dict:
        stance = self.current()
        return {
            "bot_id": self.bot_id,
            "model_stance": stance.action.value if stance else SignalAction.HOLD.value,
            "confidence": round(stance.confidence, 3) if stance else 0.0,
            "reason": stance.reason if stance else "",
            "thesis": stance.thesis if stance else "",
            "invalidation": stance.invalidation if stance else "",
            "sources": list(stance.sources) if stance else [],
            "age_sec": round(self.held.age_sec(), 1) if self.held else None,
            "budget_remaining": self.budget_remaining,
            "last_error": self.last_error,
            "usage": self.usage.as_dict(),
        }
