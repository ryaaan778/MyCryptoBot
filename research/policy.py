"""The decision seam: one interface for hand-written strategies and learned policies.

``backend.strategies.Strategy`` is already the right shape — a pure function of
recent bars to a decision, with the risk engine sitting above it — but its
signature only sees the market. A learned policy also needs account state:
whether it is in a position, how deep the current drawdown is, how long it has
been holding. Adding those arguments to ``Strategy`` would mean editing all five
existing strategies, so instead ``Policy`` is a *superset* interface and
:class:`StrategyPolicy` adapts the existing ones into it **without touching a
line of strategy code**. They keep working in the live bot exactly as they do
today, and become baselines here for free.

The baselines matter more than they look. "The RL agent made 12%" means nothing
on its own; "the RL agent made 12% where buy-and-hold made 31%" means the agent
is a worse way to hold the asset. :class:`BuyAndHoldPolicy`,
:class:`FlatPolicy` and :class:`RandomPolicy` exist so no learned result can be
reported without something to lose to.

**Window parity.** ``StrategyPolicy`` passes ``min_bars + 60`` bars, which is
exactly what ``BotAgent`` passes in the live loop. This is deliberate: the
recursive indicators (Wilder RSI/ATR, SMA-seeded EMA) give slightly different
values on a longer history, so a backtest that fed them the full series would be
measuring a strategy the live bot never runs. It also keeps the backtest O(n)
rather than O(n²).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np

from backend.models import Position, PositionSide, SignalAction
from backend.strategies import create_strategy
from backend.strategies.base import Series, Strategy

from .datasets import Dataset

logger = logging.getLogger("research.policy")

POLICY_API_VERSION = "v1"

#: Matches ``BotAgent``: ``self.strategy.min_bars + 60``.
LIVE_WINDOW_MARGIN = 60


class PolicyAction(IntEnum):
    """The four-action space. Stable integer codes — they are stored in the DB."""

    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

    @property
    def label(self) -> str:
        return self.name


_FROM_SIGNAL = {
    SignalAction.HOLD: PolicyAction.HOLD,
    SignalAction.LONG: PolicyAction.LONG,
    SignalAction.SHORT: PolicyAction.SHORT,
    SignalAction.CLOSE: PolicyAction.CLOSE,
}


@dataclass(frozen=True)
class AccountState:
    """What the policy knows about its own position and equity.

    Everything is relative — ratios and percentages, never absolute currency —
    for the same reason the market features are: a policy trained at $10,000 of
    equity should behave identically at $100,000, and one fed absolute numbers
    will not.
    """

    equity: float
    starting_equity: float
    position_side: int = 0          # -1 short, 0 flat, +1 long
    position_qty: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    unrealised_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    exposure_pct: float = 0.0
    bars_in_position: int = 0
    bars_since_trade: int = 0
    trades_taken: int = 0
    realised_pnl_pct: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.position_side == 0

    @property
    def is_long(self) -> bool:
        return self.position_side > 0

    @property
    def is_short(self) -> bool:
        return self.position_side < 0

    #: Order is part of the observation contract — appending is safe, reordering
    #: silently invalidates every policy trained before the change.
    VECTOR_NAMES: tuple[str, ...] = (
        "position_side", "unrealised_pnl_pct", "drawdown_pct", "exposure_pct",
        "bars_in_position_log", "bars_since_trade_log", "equity_ratio",
    )

    def as_vector(self) -> np.ndarray:
        """The account features the RL environment appends to the market ones."""
        equity_ratio = (
            self.equity / self.starting_equity - 1.0 if self.starting_equity > 1e-12 else 0.0
        )
        return np.array([
            float(self.position_side),
            self.unrealised_pnl_pct / 100.0,
            self.drawdown_pct / 100.0,
            self.exposure_pct / 100.0,
            # Log-compressed: the difference between holding 1 bar and 10 matters
            # far more than between 500 and 510.
            float(np.log1p(max(0, self.bars_in_position))),
            float(np.log1p(max(0, self.bars_since_trade))),
            equity_ratio,
        ], dtype=np.float64)

    def to_position(self, bot_id: str, symbol: str) -> Position | None:
        """A ``Position`` for strategies that expect one. ``None`` when flat.

        Built with ``model_construct`` to skip pydantic validation — this runs
        once per bar of every backtest, and the fields are already trusted.
        """
        if self.is_flat:
            return None
        return Position.model_construct(
            id="bt", bot_id=bot_id, symbol=symbol,
            side=PositionSide.LONG if self.is_long else PositionSide.SHORT,
            quantity=self.position_qty,
            entry_price=self.entry_price,
            mark_price=self.mark_price or self.entry_price,
            leverage=1.0, stop_loss=None, take_profit=None,
            unrealized_pnl=0.0, unrealized_pnl_pct=self.unrealised_pnl_pct,
            margin=0.0, fees_paid=0.0, opened_at=0, updated_at=0,
        )


@dataclass(frozen=True)
class Decision:
    action: PolicyAction
    confidence: float = 0.0
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyContext:
    """Everything a policy may look at on one bar — and nothing beyond it.

    ``index`` is the bar being decided on, and every accessor here is bounded by
    it. There is no method that returns a future bar, which is the cheapest
    possible structural defence against look-ahead: a policy cannot peek at data
    the context will not hand it.
    """

    dataset: Dataset
    index: int
    account: AccountState
    bot_id: str = "research"
    features: np.ndarray | None = None      # scaled market features for this bar

    @property
    def price(self) -> float:
        return float(self.dataset.close[self.index])

    @property
    def symbol(self) -> str:
        return self.dataset.manifest.symbol

    def series(self, lookback: int) -> Series:
        """Trailing ``lookback`` bars ending at (and including) ``index``.

        Slices are numpy views, so this costs nothing per call.
        """
        start = max(0, self.index + 1 - lookback)
        stop = self.index + 1
        d = self.dataset
        return Series(
            ts=d.ts[start:stop], open=d.open[start:stop], high=d.high[start:stop],
            low=d.low[start:stop], close=d.close[start:stop], volume=d.volume[start:stop],
        )

    def bars_available(self) -> int:
        return self.index + 1


@runtime_checkable
class Policy(Protocol):
    """The interface every decision-maker implements — hand-written or learned."""

    name: str
    min_bars: int

    def reset(self, seed: int | None = None) -> None: ...

    def decide(self, ctx: PolicyContext) -> Decision: ...


class BasePolicy:
    """Shared plumbing. Subclasses implement :meth:`decide`."""

    name: str = "base"
    min_bars: int = 1
    kind: str = "baseline"

    def reset(self, seed: int | None = None) -> None:
        """Clear per-episode state. Called before every backtest run."""

    def decide(self, ctx: PolicyContext) -> Decision:   # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "min_bars": self.min_bars,
                "api": POLICY_API_VERSION}

    @staticmethod
    def hold(reason: str = "") -> Decision:
        return Decision(action=PolicyAction.HOLD, confidence=0.0, reason=reason)


# --------------------------------------------------------------------------
# The adapter — existing strategies, unmodified
# --------------------------------------------------------------------------


class StrategyPolicy(BasePolicy):
    """Wraps a ``backend.strategies.Strategy`` so it satisfies :class:`Policy`.

    The strategy is called through ``compute`` rather than ``evaluate`` so the
    numpy ``Series`` can be built from dataset slices directly. ``evaluate``
    would demand a ``list[Candle]`` — several million pydantic objects over a
    walk-forward run, for no benefit — and its only other job is the
    ``min_bars`` guard, which is reproduced here exactly.
    """

    kind = "strategy"

    def __init__(self, strategy: Strategy, *, lookback: int | None = None) -> None:
        self.strategy = strategy
        self.name = strategy.name
        self.min_bars = strategy.min_bars
        self.lookback = lookback if lookback is not None else strategy.min_bars + LIVE_WINDOW_MARGIN

    @classmethod
    def from_name(
        cls, name: str, params: dict[str, float] | None = None, *, lookback: int | None = None
    ) -> "StrategyPolicy":
        return cls(create_strategy(name, params or {}), lookback=lookback)

    def decide(self, ctx: PolicyContext) -> Decision:
        if ctx.bars_available() < self.min_bars:
            return self.hold(f"warming up ({ctx.bars_available()}/{self.min_bars} bars)")

        signal = self.strategy.compute(
            ctx.bot_id,
            ctx.symbol,
            ctx.series(self.lookback),
            ctx.account.to_position(ctx.bot_id, ctx.symbol),
        )
        return Decision(
            action=_FROM_SIGNAL.get(signal.action, PolicyAction.HOLD),
            confidence=float(signal.confidence),
            reason=signal.reason,
            diagnostics=dict(signal.indicators),
        )

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "strategy": self.strategy.name,
                "params": dict(self.strategy.params), "lookback": self.lookback}


# --------------------------------------------------------------------------
# Baselines — what a learned policy has to beat
# --------------------------------------------------------------------------


class FlatPolicy(BasePolicy):
    """Never trades.

    The most important baseline in the set, and the one most often left out. It
    costs nothing, risks nothing, and returns exactly zero — so any policy with
    a negative expectancy loses to it. A learned policy that cannot beat doing
    nothing has not learned anything worth deploying.
    """

    name = "flat"

    def decide(self, ctx: PolicyContext) -> Decision:
        return self.hold("flat baseline")


class BuyAndHoldPolicy(BasePolicy):
    """Long from the first eligible bar, and never sells.

    In a bull sample this is very hard to beat, which is the point: it exposes
    the difference between a policy with an edge and a policy that is simply
    long a rising asset.
    """

    name = "buy_and_hold"

    def decide(self, ctx: PolicyContext) -> Decision:
        if ctx.account.is_flat:
            return Decision(PolicyAction.LONG, confidence=1.0, reason="buy and hold entry")
        return self.hold("holding")


class RandomPolicy(BasePolicy):
    """Seeded coin-flipping, for the null hypothesis.

    Its real job is to calibrate cost drag: a random policy's return is roughly
    the fees and slippage it paid, so if a learned policy scores near this line
    it has learned to trade, not to trade *well*.
    """

    name = "random"

    def __init__(self, seed: int = 0, trade_probability: float = 0.02) -> None:
        if not 0.0 <= trade_probability <= 1.0:
            raise ValueError("trade_probability must be in [0, 1]")
        self.seed = seed
        self.trade_probability = trade_probability
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        # Re-seeding here is what makes a run reproducible: without it a second
        # backtest of the same policy would take different actions.
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def decide(self, ctx: PolicyContext) -> Decision:
        if self._rng.random() >= self.trade_probability:
            return self.hold("no action")
        if not ctx.account.is_flat:
            return Decision(PolicyAction.CLOSE, confidence=0.5, reason="random close")
        action = PolicyAction.LONG if self._rng.random() < 0.5 else PolicyAction.SHORT
        return Decision(action, confidence=0.5, reason="random entry")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "seed": self.seed,
                "trade_probability": self.trade_probability}


class AlwaysShortPolicy(BasePolicy):
    """The mirror of buy-and-hold, so a bear sample has a matching baseline."""

    name = "always_short"

    def decide(self, ctx: PolicyContext) -> Decision:
        if ctx.account.is_flat:
            return Decision(PolicyAction.SHORT, confidence=1.0, reason="always short entry")
        return self.hold("holding")


BASELINE_POLICIES: tuple[str, ...] = ("flat", "buy_and_hold", "random", "always_short")
STRATEGY_POLICIES: tuple[str, ...] = (
    "scalping", "volatility_breakout", "momentum", "mean_reversion", "hybrid",
)


def make_policy(
    name: str, params: dict[str, float] | None = None, *, seed: int = 0
) -> BasePolicy:
    """Build any baseline or strategy policy by name."""
    if name == "flat":
        return FlatPolicy()
    if name == "buy_and_hold":
        return BuyAndHoldPolicy()
    if name == "always_short":
        return AlwaysShortPolicy()
    if name == "random":
        probability = float((params or {}).get("trade_probability", 0.02))
        return RandomPolicy(seed=seed, trade_probability=probability)
    if name in STRATEGY_POLICIES:
        return StrategyPolicy.from_name(name, params)
    raise ValueError(
        f"unknown policy {name!r}; available: "
        f"{sorted(BASELINE_POLICIES + STRATEGY_POLICIES)}"
    )


def all_baseline_policies(seed: int = 0) -> list[BasePolicy]:
    return [make_policy(name, seed=seed) for name in BASELINE_POLICIES]


def all_strategy_policies(params: dict[str, float] | None = None) -> list[BasePolicy]:
    return [make_policy(name, params) for name in STRATEGY_POLICIES]
