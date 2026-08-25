"""JOJO's job: move capital toward whoever is actually winning.

Five agents trade their own markets with their own money. This decides how much
money each one gets, and revises it as evidence comes in. An agent that is
making money is given more to work with; one that is losing is given less.

**The trap this has to avoid.** Naively allocating in proportion to recent P&L
is a well-known way to lose money: you buy into a streak at its peak and cut an
agent right before it recovers, and with five agents and a handful of trades
each, most of what looks like skill is noise. Two things keep that in check, and
neither is a cap on how freely an agent trades:

*Shrinkage toward equal weight.* An agent's measured edge is pulled toward zero
by how little evidence supports it — ``trades / (trades + prior_trades)``. Three
winning trades barely move an allocation; sixty move it a lot. This is the whole
difference between "who is winning" and "who won recently", and it costs nothing
but patience.

*A floor and a ceiling.* Nobody is starved to zero, because an agent with no
capital can never generate the evidence that would win it back — that is an
absorbing state, and absorbing states are how a portfolio quietly becomes one
strategy. And nobody takes the whole desk, because five correlated agents
holding one view is a concentration risk that P&L alone will happily walk you
into.

Allocation is a *budget*, not an instruction. It never tells an agent what to
trade, when, or which way. It only decides how much its convictions are worth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("jojo.allocator")


@dataclass(frozen=True)
class AllocatorConfig:
    """How aggressively capital chases performance."""

    #: Below this many closed trades an agent is scored, but very weakly.
    prior_trades: int = 20
    #: An agent cannot hold *more* than an even share until it has closed this
    #: many trades. Shrinkage alone does not handle a small, lucky sample: three
    #: trades averaging +300 produce a genuinely high point estimate, and no
    #: amount of pulling it toward zero changes that it beats a hundred trades
    #: averaging +0.50. What is missing at n=3 is not magnitude, it is grounds.
    #: So a big budget has to be *earned over time*, not won in an afternoon.
    min_trades_for_conviction: int = 25
    #: No agent falls below this share, however badly it is doing.
    floor: float = 0.05
    #: No agent rises above this share, however well it is doing.
    ceiling: float = 0.40
    #: Total share handed out across all agents.
    total: float = 1.0
    #: How much of the way to move toward the new target each round, 0..1.
    #: Allocation on noisy samples is itself noisy, and a desk that jumps to
    #: every fresh estimate spends its edge on churn. Moving part of the way
    #: keeps the direction and drops most of the whipsaw.
    smoothing: float = 0.35

    def validate(self) -> None:
        if not 0.0 <= self.floor <= self.ceiling <= 1.0:
            raise ValueError("need 0 <= floor <= ceiling <= 1")
        if self.total <= 0:
            raise ValueError("total must be positive")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")


@dataclass(frozen=True)
class AgentPerformance:
    """What the desk knows about one agent's results so far."""

    bot_id: str
    trades: int = 0
    wins: int = 0
    realized_pnl: float = 0.0
    #: Average notional per closed trade — the scale the results were actually
    #: produced at. Used to make edge scale-free, so an agent holding a big
    #: budget does not look skilful merely for betting bigger.
    #:
    #: This must NOT be derived from the agent's current allocation. An earlier
    #: version used ``equity * allocation`` and created a feedback loop: cutting
    #: an agent shrank its denominator, inflated its apparent edge, and won the
    #: money straight back — the worst agent on the desk took the ceiling within
    #: a few rounds. Trade notional is a property of the trades themselves and
    #: cannot move when the budget does.
    capital: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy(self) -> float:
        """Average P&L per closed trade."""
        return self.realized_pnl / self.trades if self.trades else 0.0

    @property
    def edge(self) -> float:
        """Expectancy as a fraction of the capital that produced it.

        Without this, the agent holding the largest allocation looks the most
        skilful simply because its trades are bigger — and allocation would then
        feed on itself regardless of results.
        """
        if self.trades <= 0 or self.capital <= 0:
            return 0.0
        return self.expectancy / self.capital


@dataclass(frozen=True)
class Allocation:
    """One agent's new share, and why."""

    bot_id: str
    previous: float
    allocation: float
    score: float
    confidence: float
    performance: AgentPerformance
    reason: str

    @property
    def delta(self) -> float:
        return self.allocation - self.previous

    def as_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "previous": round(self.previous, 4),
            "allocation": round(self.allocation, 4),
            "delta": round(self.delta, 4),
            "score": round(self.score, 8),
            "evidence": round(self.confidence, 3),
            "trades": self.performance.trades,
            "win_rate": round(self.performance.win_rate, 4),
            "realized_pnl": round(self.performance.realized_pnl, 2),
            "reason": self.reason,
        }


class PerformanceAllocator:
    """Turns results into budgets."""

    def __init__(self, config: AllocatorConfig | None = None) -> None:
        self.config = config or AllocatorConfig()
        self.config.validate()

    # ---- scoring -----------------------------------------------------------

    def evidence_weight(self, trades: int) -> float:
        """How much to believe a measured edge, given how little we have seen.

        Rises from 0 toward 1 as trades accumulate. At ``prior_trades`` it is
        exactly 0.5 — the sample and the prior carry equal weight.
        """
        if trades <= 0:
            return 0.0
        return trades / (trades + self.config.prior_trades)

    def score(self, performance: AgentPerformance) -> float:
        """A shrunk edge. Positive means "give this agent more"."""
        return performance.edge * self.evidence_weight(performance.trades)

    # ---- allocation --------------------------------------------------------

    def allocate(
        self,
        performances: dict[str, AgentPerformance],
        previous: dict[str, float] | None = None,
    ) -> list[Allocation]:
        """Split ``total`` across the agents according to demonstrated edge."""
        previous = previous or {}
        if not performances:
            return []

        cfg = self.config
        ids = sorted(performances)
        scores = {bid: self.score(performances[bid]) for bid in ids}
        even = cfg.total / len(ids)

        untested = all(performances[bid].trades == 0 for bid in ids)
        positive = {bid: s for bid, s in scores.items() if s > 0}

        shares: dict[str, float]
        forced_reason = ""
        if untested:
            # Nothing has been tried yet. Everyone gets an equal chance to
            # generate the evidence that will later move their budget.
            shares = {bid: even for bid in ids}
            forced_reason = "no trades closed yet — equal starting budget"
        elif not positive:
            # Agents have traded and none of them is making money. Splitting the
            # desk evenly here would deploy full capital behind a roster with no
            # demonstrated edge. Everyone drops to the floor instead, which keeps
            # them alive and still gathering evidence while most of the desk
            # sits in cash.
            shares = {bid: cfg.floor for bid in ids}
            forced_reason = "no agent is profitable — minimum budget, rest held as cash"
        else:
            weight_total = sum(positive.values())
            shares = {
                bid: cfg.total * positive.get(bid, 0.0) / weight_total for bid in ids
            }
            shares = self._apply_bounds(shares, self._ceilings(performances, even))

        shares = self._smooth(shares, previous)

        out: list[Allocation] = []
        for bid in ids:
            performance = performances[bid]
            out.append(Allocation(
                bot_id=bid,
                previous=previous.get(bid, 0.0),
                allocation=shares[bid],
                score=scores[bid],
                confidence=self.evidence_weight(performance.trades),
                performance=performance,
                reason=forced_reason or self._reason(performance, scores[bid], shares[bid]),
            ))
        return out

    def _smooth(
        self, target: dict[str, float], previous: dict[str, float]
    ) -> dict[str, float]:
        """Move part of the way from the current budget to the new one."""
        alpha = self.config.smoothing
        if alpha >= 1.0:
            return target
        out = {}
        for bid, want in target.items():
            have = previous.get(bid)
            # No previous budget on record means this is the first allocation
            # for this agent; there is nothing to ease away from.
            out[bid] = want if have is None else have + alpha * (want - have)

        # Blending is a convex combination, so it preserves bounds *if* the
        # previous budget respected them. A hand-edited config need not have,
        # so re-clamp and re-normalise rather than trust the input.
        cfg = self.config
        out = {bid: min(max(v, cfg.floor), cfg.ceiling) for bid, v in out.items()}
        total = sum(out.values())
        if total > cfg.total and total > 0:
            out = {bid: v * cfg.total / total for bid, v in out.items()}
        return out

    def _ceilings(
        self, performances: dict[str, AgentPerformance], even: float
    ) -> dict[str, float]:
        """Per-agent upper bound on budget.

        Normally the configured ceiling. An agent that has not yet closed
        ``min_trades_for_conviction`` trades is held to an even share instead —
        and this has to be a real ceiling carried through the redistribution
        rather than a pre-pass, or the agent is simply handed the slack again
        when everyone else is pinned to the floor.
        """
        threshold = self.config.min_trades_for_conviction
        return {
            bid: (self.config.ceiling if perf.trades >= threshold
                  else min(even, self.config.ceiling))
            for bid, perf in performances.items()
        }

    def _apply_bounds(
        self, shares: dict[str, float], ceilings: dict[str, float]
    ) -> dict[str, float]:
        """Clamp each share into [floor, its own ceiling], redistributing slack.

        Iterative because pinning one agent changes what is left for the rest,
        which can push another through a bound in turn. Whatever cannot be
        placed without breaking a bound is simply not allocated — the desk holds
        it as cash. That is the honest outcome when one agent is the only one
        earning and is already at its limit: the alternative is handing the
        remainder to agents the evidence says are losing money.
        """
        cfg = self.config
        ids = list(shares)
        if not ids:
            return {}
        if cfg.floor * len(ids) > cfg.total:
            return {bid: cfg.total / len(ids) for bid in ids}

        current = dict(shares)
        for _ in range(len(ids) + 2):
            pinned = {
                bid: min(max(v, cfg.floor), ceilings[bid])
                for bid, v in current.items()
                if v < cfg.floor or v > ceilings[bid]
            }
            if not pinned:
                break
            free = [bid for bid in ids if bid not in pinned]
            remaining = cfg.total - sum(pinned.values())
            free_total = sum(current[bid] for bid in free)
            if not free or remaining <= 0 or free_total <= 0:
                current = {**current, **pinned}
                break
            current = {
                **pinned,
                **{bid: remaining * current[bid] / free_total for bid in free},
            }

        current = {
            bid: min(max(v, cfg.floor), ceilings[bid]) for bid, v in current.items()
        }
        overshoot = sum(current.values())
        if overshoot > cfg.total and overshoot > 0:
            current = {bid: v * cfg.total / overshoot for bid, v in current.items()}
        return current

    @staticmethod
    def _reason(performance: AgentPerformance, score: float, share: float) -> str:
        if performance.trades == 0:
            return "no closed trades yet"
        direction = "winning" if performance.realized_pnl > 0 else "losing"
        return (
            f"{direction}: {performance.realized_pnl:+,.2f} over "
            f"{performance.trades} trades ({performance.win_rate:.0%} won) "
            f"-> {share:.1%} of the desk"
        )
