"""Reward function v1.

    r_t = Δlog_equity_t
        − λ_dd   · max(0, DD_t − DD_{t−1})     # NEW drawdown only
        − λ_to   · 1[position changed]          # turnover, beyond fees
        − λ_tail · max(0, −Δlog_equity_t)²      # convex tail penalty

Four decisions in that formula are worth defending, because each is a place
where the obvious choice produces a policy that games the metric.

**Costs are not a term.** The brief lists transaction costs and slippage as
separate subtractions. They are already inside ``E``: the environment fills
through ``PaperExecution``, which crosses the spread, adds slippage and charges
fees before the equity curve ever sees the trade. Subtracting them again would
charge every trade twice and bias the agent toward inaction — it would learn
that trading is roughly twice as expensive as it really is, and the policy that
came out would be wrong in live markets in a way no backtest would reveal.

**New drawdown, not absolute drawdown.** Penalising the *level* of drawdown on
every step charges the agent again and again for a mistake it can no longer
undo, which produces paralysis: once underwater, every action looks equally
bad, so the gradient stops distinguishing between recovering and doing nothing.
Charging only the increment attributes the cost to the action that caused it.

**No inactivity penalty.** Doing nothing scores exactly zero, and zero beats
negative, so a lazy policy is a genuine local optimum — that is not an
oversight. The alternative, paying the agent to act, teaches churn, and churn is
how the shipped strategies lost their edge to fees. The minimum-trade
requirement lives in the *validation gate* instead: the reward stays clean and
the gate filters. A policy that learns to sit out a bad market is behaving
correctly; a policy that sits out everything simply fails the gate.

**The tail penalty is convex.** A squared term is nearly free at ordinary bar
returns (a −1% bar costs 5e-5) and severe at catastrophic ones (a −50% bar
costs 0.125, dwarfing the linear term). That asymmetry is the point: two −1%
bars should not feel the same as one −2% bar to something optimising an
average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REWARD_VERSION = "v1"

#: Guards against a reward exploding a training run. Reached only by a bar that
#: nearly wipes the account, at which point the episode is over anyway.
REWARD_CLIP = 10.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the reward terms.

    Recorded in policy metadata alongside ``version`` so that a policy trained
    under one reward is never silently compared against one trained under
    another — a change here changes what "good" means, and comparing across the
    change is comparing two different objectives.
    """

    lambda_drawdown: float = 1.0
    lambda_turnover: float = 2e-4
    lambda_tail: float = 0.5
    clip: float = REWARD_CLIP
    version: str = REWARD_VERSION

    def validate(self) -> None:
        for name in ("lambda_drawdown", "lambda_turnover", "lambda_tail"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.clip <= 0:
            raise ValueError("clip must be positive")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "lambda_drawdown": self.lambda_drawdown,
            "lambda_turnover": self.lambda_turnover,
            "lambda_tail": self.lambda_tail,
            "clip": self.clip,
            "version": self.version,
        }


@dataclass(frozen=True)
class RewardBreakdown:
    """Every term separately, so a training curve can be diagnosed rather than guessed at."""

    total: float
    log_return: float
    drawdown_penalty: float
    turnover_penalty: float
    tail_penalty: float
    clipped: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "reward": self.total,
            "reward_log_return": self.log_return,
            "reward_drawdown_penalty": self.drawdown_penalty,
            "reward_turnover_penalty": self.turnover_penalty,
            "reward_tail_penalty": self.tail_penalty,
        }


def compute_reward(
    *,
    log_return: float,
    drawdown_pct: float,
    drawdown_before_pct: float,
    position_changed: bool,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """One step's reward.

    ``drawdown_pct`` arrives in percent because that is what ``Portfolio``
    reports, and it is converted to a fraction here. Getting that wrong is not a
    rounding error: a percent-scaled drawdown term with ``lambda_drawdown=1.0``
    is a hundred times heavier than intended, and it would swamp the return term
    so completely that the agent would learn to never open a position — while
    the training curve looked perfectly healthy.
    """
    cfg = config or RewardConfig()

    if not np.isfinite(log_return):
        log_return = 0.0

    new_drawdown = max(0.0, (drawdown_pct - drawdown_before_pct) / 100.0)
    drawdown_penalty = cfg.lambda_drawdown * new_drawdown
    turnover_penalty = cfg.lambda_turnover if position_changed else 0.0
    tail_penalty = cfg.lambda_tail * max(0.0, -log_return) ** 2

    total = float(log_return - drawdown_penalty - turnover_penalty - tail_penalty)
    clipped = abs(total) > cfg.clip
    if clipped:
        total = float(np.clip(total, -cfg.clip, cfg.clip))

    return RewardBreakdown(
        total=total,
        log_return=float(log_return),
        drawdown_penalty=float(-drawdown_penalty),
        turnover_penalty=float(-turnover_penalty),
        tail_penalty=float(-tail_penalty),
        clipped=clipped,
    )


def reward_for_step(outcome, config: RewardConfig | None = None) -> RewardBreakdown:
    """Reward from a :class:`research.backtest.StepOutcome`."""
    return compute_reward(
        log_return=outcome.log_return,
        drawdown_pct=outcome.drawdown_pct,
        drawdown_before_pct=outcome.drawdown_before,
        position_changed=outcome.position_changed,
        config=config,
    )
