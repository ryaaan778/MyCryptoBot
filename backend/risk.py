"""Risk engine.

Two jobs: decide how large a trade may be, and decide when trading must stop.
Every rejection carries a human-readable reason, because those strings are what
the world shows above a bot's head when it refuses to trade.

The utilization figure (0..1) is the single number that drives the giant voxel
risk meter beside JOJO.
"""

from __future__ import annotations

import logging

from .config import Settings
from .models import BotConfig, PositionSide, RiskLevel, RiskState
from .portfolio import Portfolio

logger = logging.getLogger("jojo.risk")


class RiskDecision:
    __slots__ = ("allowed", "reason", "quantity")

    def __init__(self, allowed: bool, reason: str = "", quantity: float = 0.0) -> None:
        self.allowed = allowed
        self.reason = reason
        self.quantity = quantity

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RiskDecision(allowed={self.allowed}, qty={self.quantity:.6f}, {self.reason!r})"


class RiskEngine:
    def __init__(self, settings: Settings, portfolio: Portfolio) -> None:
        self.settings = settings
        self.portfolio = portfolio
        self.limits = settings.risk
        self.emergency_stop = False
        self.halted_bots: set[str] = set()
        self._breaches: list[str] = []

    # ---- sizing ------------------------------------------------------------

    def size_position(
        self, bot: BotConfig, price: float, stop_distance_pct: float
    ) -> RiskDecision:
        """Risk-based sizing: never stake more than the bot's per-trade budget.

        Quantity is chosen so that being stopped out costs exactly
        ``risk_per_trade`` of the bot's allocated equity.
        """
        if price <= 0:
            return RiskDecision(False, "invalid price")
        if stop_distance_pct <= 0:
            return RiskDecision(False, "stop distance must be positive")

        equity = self.portfolio.equity
        if equity <= 0:
            return RiskDecision(False, "no equity remaining")

        allocated = equity * bot.allocation
        risk_budget = allocated * min(bot.risk_per_trade, self.limits.max_risk_per_trade)
        loss_per_unit = price * (stop_distance_pct / 100.0)
        quantity = risk_budget / loss_per_unit

        # Never reserve more margin than the bot's allocation can back.
        notional = quantity * price
        max_margin = allocated
        margin = notional / max(bot.leverage, 1e-9)
        if margin > max_margin:
            quantity *= max_margin / margin
            notional = quantity * price
            margin = notional / max(bot.leverage, 1e-9)

        if margin > self.portfolio.cash:
            if self.portfolio.cash <= 0:
                return RiskDecision(False, "no free cash to post as margin")
            quantity *= self.portfolio.cash / margin
            notional = quantity * price

        if quantity <= 0:
            return RiskDecision(False, "computed size rounded to zero")

        return RiskDecision(True, "sized against per-trade risk budget", quantity)

    # ---- gating ------------------------------------------------------------

    def can_open(self, bot: BotConfig, notional: float) -> RiskDecision:
        if self.emergency_stop:
            return RiskDecision(False, "EMERGENCY STOP is active")
        if bot.id in self.halted_bots:
            return RiskDecision(False, "bot is halted by the risk engine")

        open_count = len(self.portfolio.positions)
        if open_count >= self.limits.max_portfolio_positions:
            return RiskDecision(
                False,
                f"desk already holds {open_count}/{self.limits.max_portfolio_positions} positions",
            )

        bot_positions = len(self.portfolio.positions_for(bot.id))
        if bot_positions >= bot.max_positions:
            return RiskDecision(
                False, f"{bot.name} already holds {bot_positions}/{bot.max_positions} positions"
            )

        equity = self.portfolio.equity
        if equity <= 0:
            return RiskDecision(False, "no equity remaining")

        projected = (self.portfolio.total_exposure + notional) / equity * 100.0
        if projected > self.limits.max_exposure_pct:
            return RiskDecision(
                False,
                f"exposure would reach {projected:.0f}% of equity "
                f"(limit {self.limits.max_exposure_pct:.0f}%)",
            )

        drawdown = self.portfolio.drawdown_pct
        if drawdown >= self.limits.drawdown_limit_pct:
            return RiskDecision(
                False,
                f"drawdown {drawdown:.1f}% has reached the "
                f"{self.limits.drawdown_limit_pct:.0f}% limit",
            )

        return RiskDecision(True, "within risk limits", 0.0)

    # ---- stops -------------------------------------------------------------

    @staticmethod
    def stop_levels(
        side: PositionSide, entry: float, stop_loss_pct: float, take_profit_pct: float
    ) -> tuple[float, float]:
        if side is PositionSide.LONG:
            return (
                entry * (1.0 - stop_loss_pct / 100.0),
                entry * (1.0 + take_profit_pct / 100.0),
            )
        return (
            entry * (1.0 + stop_loss_pct / 100.0),
            entry * (1.0 - take_profit_pct / 100.0),
        )

    # ---- state -------------------------------------------------------------

    # A full book is "fully deployed", not "critical" — so slot pressure is
    # capped below the danger bands and only exposure or drawdown can escalate
    # the meter to HIGH/CRITICAL on its own.
    SLOT_PRESSURE_CEILING = 0.6

    def utilization(self) -> float:
        """Worst of the three pressures, as a 0..1 fraction of its limit."""
        equity = self.portfolio.equity
        if equity <= 0:
            return 1.0
        exposure_ratio = (
            self.portfolio.total_exposure / equity * 100.0 / self.limits.max_exposure_pct
        )
        drawdown_ratio = self.portfolio.drawdown_pct / max(self.limits.drawdown_limit_pct, 1e-9)
        slots_ratio = (
            len(self.portfolio.positions)
            / max(self.limits.max_portfolio_positions, 1)
            * self.SLOT_PRESSURE_CEILING
        )
        return max(0.0, min(1.0, max(exposure_ratio, drawdown_ratio, slots_ratio)))

    def level(self, utilization: float | None = None) -> RiskLevel:
        u = self.utilization() if utilization is None else utilization
        if self.emergency_stop or u >= self.limits.critical_utilization:
            return RiskLevel.CRITICAL
        if u >= self.limits.high_utilization:
            return RiskLevel.HIGH
        if u >= self.limits.elevated_utilization:
            return RiskLevel.ELEVATED
        return RiskLevel.SAFE

    def evaluate(self) -> RiskState:
        """Recompute the portfolio risk picture and record any fresh breaches."""
        equity = self.portfolio.equity
        exposure = self.portfolio.total_exposure
        exposure_pct = (exposure / equity * 100.0) if equity > 0 else 0.0
        drawdown = self.portfolio.drawdown_pct
        utilization = self.utilization()

        breaches: list[str] = []
        if exposure_pct > self.limits.max_exposure_pct:
            breaches.append(
                f"exposure {exposure_pct:.0f}% over {self.limits.max_exposure_pct:.0f}% limit"
            )
        if drawdown >= self.limits.drawdown_limit_pct:
            breaches.append(
                f"drawdown {drawdown:.1f}% at or past {self.limits.drawdown_limit_pct:.0f}% limit"
            )
        if len(self.portfolio.positions) > self.limits.max_portfolio_positions:
            breaches.append("open positions exceed the configured maximum")
        self._breaches = breaches

        return RiskState(
            level=self.level(utilization),
            total_exposure=round(exposure, 2),
            exposure_pct=round(exposure_pct, 2),
            max_exposure_pct=self.limits.max_exposure_pct,
            current_drawdown_pct=round(drawdown, 3),
            max_drawdown_pct=round(
                max(drawdown, getattr(self, "_max_drawdown", 0.0)), 3
            ),
            drawdown_limit_pct=self.limits.drawdown_limit_pct,
            open_positions=len(self.portfolio.positions),
            max_open_positions=self.limits.max_portfolio_positions,
            utilization=round(utilization, 4),
            emergency_stop=self.emergency_stop,
            halted_bots=sorted(self.halted_bots),
            breaches=breaches,
        )

    def bot_risk(self, bot: BotConfig) -> tuple[RiskLevel, float]:
        """Per-bot risk, used for the aura around each character.

        Two pressures, whichever is worse:

        * **deployment** — how much of the bot's allocated capital is posted as
          margin, scaled by leverage. Leverage must *raise* this reading: KIRA
          at 8x running the same notional as JONATHAN at 3x is carrying far more
          risk, and the world should show that.
        * **loss pressure** — how deep the open loss is relative to what the bot
          was supposed to be risking. Reaches 1.0 at three times its per-trade
          budget, which is the point the aura should be screaming.
        """
        if bot.id in self.halted_bots or self.emergency_stop:
            return RiskLevel.CRITICAL, 1.0

        equity = self.portfolio.equity
        allocated = equity * bot.allocation
        if allocated <= 0:
            return RiskLevel.CRITICAL, 1.0

        margin_used = sum(p.margin for p in self.portfolio.positions.values() if p.bot_id == bot.id)
        reference_leverage = max(self.settings.trading.leverage, 1.0)
        leverage_factor = min(1.0, max(bot.leverage, 1.0) / reference_leverage)
        deployment = (margin_used / allocated) * (0.45 + 0.55 * leverage_factor)

        unrealized = self.portfolio.bot_unrealized(bot.id)
        budget = allocated * max(bot.risk_per_trade, 1e-6) * 3.0
        loss_pressure = max(0.0, -unrealized) / budget if budget > 0 else 0.0

        utilization = min(1.0, max(deployment, loss_pressure))

        if utilization >= self.limits.critical_utilization:
            return RiskLevel.CRITICAL, utilization
        if utilization >= self.limits.high_utilization:
            return RiskLevel.HIGH, utilization
        if utilization >= self.limits.elevated_utilization:
            return RiskLevel.ELEVATED, utilization
        return RiskLevel.SAFE, utilization

    # ---- controls ----------------------------------------------------------

    def halt(self, bot_id: str) -> None:
        self.halted_bots.add(bot_id)

    def resume(self, bot_id: str) -> None:
        self.halted_bots.discard(bot_id)

    def trigger_emergency_stop(self) -> None:
        self.emergency_stop = True

    def clear_emergency_stop(self) -> None:
        self.emergency_stop = False
        self.halted_bots.clear()
