"""One autonomous trading agent — one voxel character in the world.

The loop is deliberately small: read candles, ask the strategy, hand any
actionable signal to the orchestrator. Sizing, risk and execution all live
elsewhere, so a bot cannot bypass a limit by being clever.

Status transitions are part of the product, not debug state: the world animates
straight off them (IDLE stands, ANALYZING studies holograms, TRADING strikes).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .models import (
    BotConfig,
    BotState,
    BotStatus,
    RiskLevel,
    Signal,
    SignalAction,
    now_ms,
)
from .llm.brain import AccountContext
from .strategies import create_strategy
from .strategies.base import Series

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import JojoOrchestrator

logger = logging.getLogger("jojo.bot")


class BotAgent:
    def __init__(self, config: BotConfig, engine: "JojoOrchestrator") -> None:
        self.config = config
        self.engine = engine
        self.strategy = create_strategy(config.strategy, engine.settings.strategy_parameters)
        self.brain = self._build_brain()
        self.status: BotStatus = BotStatus.OFFLINE
        self.last_signal: Signal | None = None
        self.last_heartbeat: int = 0
        self.error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._paused = False
        # Guards against the event log filling with the same signal re-fired
        # every tick while a bot sits blocked or already positioned.
        self._last_emitted: tuple[str, str] | None = None
        self._last_emitted_ts = 0
        # Realised P&L at the last tick. A change means a trade closed, which
        # is how a model-driven bot finds out whether it was right.
        self._last_realized: float | None = None

    def _build_brain(self):
        """Give the bot a model, if it is configured to have one.

        Failure here is never fatal: a missing SDK or a missing key leaves the
        strategy unbound, which holds. A bot that cannot reach its model does
        not trade — it does not fall back to guessing.
        """
        from .llm.strategy import LLMStrategy

        if not isinstance(self.strategy, LLMStrategy):
            return None
        settings = self.engine.settings
        if not getattr(settings, "llm", None) or not settings.llm.enabled:
            logger.info("%s is on the llm strategy but llm.enabled is false", self.name)
            return None
        try:
            from .llm import build_brain
            brain = build_brain(self.config, settings)
        except Exception as exc:
            logger.error("%s could not build its model brain: %s", self.name, exc)
            return None
        self.strategy.bind_brain(brain)
        return brain

    # ---- identity ----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    # ---- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._paused = False
        self._set_status(BotStatus.IDLE)
        self._task = asyncio.create_task(self._run(), name=f"bot-{self.id}")
        self.engine.bus.emit(
            "bot.started", f"{self.name} has entered the world.",
            severity="success", bot_id=self.id, symbol=self.config.symbol,
        )

    async def stop(self) -> None:
        self._running = False
        if self.brain is not None:
            await self.brain.aclose()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._set_status(BotStatus.OFFLINE)

    def pause(self) -> None:
        self._paused = True
        self._set_status(BotStatus.PAUSED)

    def resume(self) -> None:
        self._paused = False
        if self._running:
            self._set_status(BotStatus.IDLE)

    def halt(self, reason: str) -> None:
        self.error = reason
        self._set_status(BotStatus.HALTED)

    # ---- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        # Stagger the roster so five bots don't all hit the strategy at once.
        await asyncio.sleep(0.4 * (hash(self.id) % 7))
        while self._running:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = str(exc)
                logger.exception("%s tick failed", self.name)
                self.engine.bus.emit(
                    "bot.error", f"{self.name} hit an error: {exc}",
                    severity="danger", bot_id=self.id,
                )
            await asyncio.sleep(self.config.interval_sec)

    async def tick(self) -> Signal | None:
        """One evaluation cycle. Separated from the loop so tests can drive it."""
        self.last_heartbeat = now_ms()
        self.engine.bus.publish(
            "bot.heartbeat", {"bot_id": self.id, "ts": self.last_heartbeat}
        )

        if self._paused or self.status is BotStatus.HALTED:
            return None
        if self.engine.risk.emergency_stop:
            self._set_status(BotStatus.HALTED)
            return None

        candles = self.engine.provider.candles(
            self.config.symbol, self.config.timeframe, limit=self.strategy.min_bars + 60
        )
        if len(candles) < self.strategy.min_bars:
            self._set_status(BotStatus.IDLE)
            return None

        self._set_status(BotStatus.ANALYZING)
        position = self.engine.portfolio.position_for(self.id, self.config.symbol)
        self._score_closed_trade()
        self._maybe_think(candles, position)
        signal = self.strategy.evaluate(self.id, self.config.symbol, candles, position)
        self.last_signal = signal

        if signal.action is not SignalAction.HOLD:
            self.engine.bus.publish("bot.signal", signal.model_dump(mode="json"))
            if self._should_announce(signal):
                self.engine.bus.emit(
                    "signal.detected",
                    f"{self.name} detected {signal.action.value} on "
                    f"{self.config.symbol}: {signal.reason}",
                    severity="info", bot_id=self.id, symbol=self.config.symbol,
                    action=signal.action.value, confidence=round(signal.confidence, 3),
                )
            self._set_status(BotStatus.TRADING)
            await self.engine.handle_signal(self, signal)

        if self.status is BotStatus.TRADING:
            self._set_status(BotStatus.IDLE)
        elif self.status is BotStatus.ANALYZING:
            self._set_status(BotStatus.IDLE)
        return signal

    # ---- the model ---------------------------------------------------------

    def _score_closed_trade(self) -> None:
        """Feed a realised outcome back to the model that asked for the trade.

        Detected from the bot's own realised P&L rather than by listening for a
        close event, so it works no matter *how* the position ended — the
        model's call, a stop, a take-profit, or the risk engine flattening it.
        Being wrong because your stop hit is still being wrong, and the record
        should say so.
        """
        if self.brain is None:
            return
        realized = self.engine.portfolio.bot_realized(self.id)
        previous, self._last_realized = self._last_realized, realized
        if previous is None:
            return
        delta = realized - previous
        if abs(delta) < 1e-9:
            return
        self.brain.record_outcome(delta, note=f"realised on {self.config.symbol}")

    def _maybe_think(self, candles, position) -> None:
        """Kick off a reasoning call if one is due.

        Fire-and-forget on purpose. The tick loop must stay responsive — the
        world is animating off it — so this never awaits the model. The stance
        it produces is picked up by a later tick.
        """
        if self.brain is None or not self.brain.due():
            return

        portfolio = self.engine.portfolio
        total, _won, win_rate = portfolio.bot_win_rate(self.id)
        account = AccountContext(
            equity=portfolio.equity,
            drawdown_pct=portfolio.drawdown_pct,
            open_positions=len(portfolio.positions),
            max_positions=self.engine.risk.limits.max_portfolio_positions,
            realized_pnl=portfolio.bot_realized(self.id),
            win_rate=win_rate,
            trades=total,
        )
        self._set_status(BotStatus.ANALYZING)
        self.brain.refresh_in_background(
            symbol=self.config.symbol, timeframe=self.config.timeframe,
            series=Series.from_candles(candles), position=position,
            account=account,
        )

    # A repeat of the same call is only worth re-announcing this often.
    REANNOUNCE_SEC = 60

    def _should_announce(self, signal: Signal) -> bool:
        key = (signal.action.value, signal.reason)
        elapsed_ms = now_ms() - self._last_emitted_ts
        if key == self._last_emitted and elapsed_ms < self.REANNOUNCE_SEC * 1000:
            return False
        self._last_emitted = key
        self._last_emitted_ts = now_ms()
        return True

    # ---- state -------------------------------------------------------------

    def _set_status(self, status: BotStatus) -> None:
        if status is self.status:
            return
        self.status = status
        self.engine.bus.publish(
            "bot.status",
            {"bot_id": self.id, "status": status.value, "ts": now_ms()},
        )

    def state(self) -> BotState:
        portfolio = self.engine.portfolio
        realized = portfolio.bot_realized(self.id)
        unrealized = portfolio.bot_unrealized(self.id)
        total, won, win_rate = portfolio.bot_win_rate(self.id)
        level, utilization = self.engine.risk.bot_risk(self.config)
        allocation_equity = portfolio.equity * self.config.allocation

        return BotState(
            id=self.id,
            name=self.name,
            strategy=self.config.strategy,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            status=self.status,
            allocation=round(self.config.allocation, 4),
            equity=round(allocation_equity + realized + unrealized, 2),
            realized_pnl=round(realized, 2),
            unrealized_pnl=round(unrealized, 2),
            total_pnl=round(realized + unrealized, 2),
            exposure=round(portfolio.bot_exposure(self.id), 2),
            leverage=self.config.leverage,
            trades_total=total,
            trades_won=won,
            win_rate=round(win_rate, 2),
            open_positions=len(portfolio.positions_for(self.id)),
            risk_level=level if self.status is not BotStatus.OFFLINE else RiskLevel.SAFE,
            risk_utilization=round(utilization, 4),
            last_signal=self.last_signal,
            last_heartbeat=self.last_heartbeat,
            error=self.error,
            persona=self.config.persona,
        )
