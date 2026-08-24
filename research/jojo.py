"""JOJO: ranking, champion/challenger, allocation, retirement.

JOJO manages the research ecosystem. It decides which agent's policy is that
agent's champion, how much **paper** capital each agent gets, and when a
champion has decayed enough to retire. It does not trade, and it cannot make
anything safer or riskier than the risk engine already allows.

**What JOJO cannot do, structurally.** This module never imports ``RiskEngine``,
never touches ``settings.risk``, and never writes to any live-trading gate. Its
only lever is an entry in the ``allocations`` table, which is paper capital and
nothing else. A test asserts the absence of those imports rather than trusting
the docstring — "JOJO must not be able to override hard risk controls" is a
property of the dependency graph here, not a promise.

**Correlation is treated as a risk, not a win.** Five agents that all discovered
the same edge are not five edges; they are one bet with five names on it, and a
desk allocating equally across them is far more concentrated than its own
paperwork suggests. So allocation applies a haircut for how correlated an
agent's returns are with the rest of the desk, and a desk whose average pairwise
correlation is high is flagged loudly.

Correlation is measured on **per-bar equity returns** rather than on positions.
Positions would answer "are they holding the same thing"; returns answer "do
they lose money at the same time", which is the question a portfolio actually
cares about.

**Promotion is evidence-driven; deployment is not automatic.** A challenger that
clears the gate becomes that agent's champion — inside the research system. It
does not reach live trading, which stays behind the existing three-part gate and
an explicit human decision, per policy and per venue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .backtest import BacktestResult
from .evaluate import GateConfig, GateResult, evaluate_gate
from .experiments import ExperimentStore

logger = logging.getLogger("research.jojo")

JOJO_VERSION = "v1"


@dataclass(frozen=True)
class AllocationConfig:
    """How paper capital is spread across the desk.

    ``reserve_fraction`` is unallocated on purpose. A research desk that always
    deploys 100% has no capacity to add a promising newcomer without first
    taking capital from someone, which biases it toward whoever is already
    there.
    """

    total_paper_capital: float = 50_000.0
    max_share: float = 0.40
    min_share: float = 0.0
    reserve_fraction: float = 0.20
    correlation_penalty: float = 0.5
    #: Agents scoring at or below this get nothing. Zero is the natural line:
    #: it is what doing nothing scores, and paying an agent to underperform
    #: doing nothing is how a desk funds its own losses.
    min_score: float = 0.0

    def validate(self) -> None:
        if self.total_paper_capital <= 0:
            raise ValueError("total_paper_capital must be positive")
        if not 0 < self.max_share <= 1:
            raise ValueError("max_share must be in (0, 1]")
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if self.correlation_penalty < 0:
            raise ValueError("correlation_penalty cannot be negative")
        if self.min_share < 0 or self.min_share > self.max_share:
            raise ValueError("min_share must be between 0 and max_share")


@dataclass(frozen=True)
class RetirementConfig:
    """When a champion has decayed enough to stand down."""

    lookback_windows: int = 3
    #: Fraction of its promoted score a champion may fall to before retiring.
    decay_tolerance: float = 0.5
    #: Retire outright if it loses to the best baseline this many recent windows.
    max_windows_below_baseline: int = 3

    def validate(self) -> None:
        if self.lookback_windows < 1:
            raise ValueError("lookback_windows must be at least 1")
        if not 0 <= self.decay_tolerance <= 1:
            raise ValueError("decay_tolerance must be in [0, 1]")


@dataclass(frozen=True)
class AgentRanking:
    agent: str
    score: float
    drawdown_pct: float
    trades: int
    runs: int
    correlation: float = 0.0
    allocation: float = 0.0
    reason: str = ""

    def describe(self) -> str:
        return (
            f"{self.agent:<14}{self.score:>9.3f}{self.drawdown_pct:>9.2f}%"
            f"{self.trades:>8}{self.correlation:>+8.2f}{self.allocation:>12,.0f}   {self.reason}"
        )


@dataclass(frozen=True)
class RetirementDecision:
    agent: str
    policy_id: str
    retire: bool
    reason: str


def _median(results: Sequence[BacktestResult], metric: str) -> float:
    values = [r.metrics.get(metric) for r in results]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def return_series(results: Sequence[BacktestResult]) -> np.ndarray:
    """One agent's per-bar log returns, chained across windows.

    Windows each restart at the same capital, so the boundary bar between them
    is dropped rather than counted — the reset is not a return anyone earned.
    """
    ordered = sorted(results, key=lambda r: (r.window if r.window is not None else 0,
                                             r.start_index))
    parts = []
    for result in ordered:
        equity = np.maximum(np.asarray(result.equity, dtype=np.float64), 1e-12)
        if equity.size > 1:
            parts.append(np.log(equity[1:] / equity[:-1]))
    return np.concatenate(parts) if parts else np.zeros(0)


def correlation_matrix(
    by_agent: Mapping[str, Sequence[BacktestResult]]
) -> tuple[list[str], np.ndarray]:
    """Pairwise correlation of agents' return series.

    An agent that never traded has a constant equity curve and therefore no
    correlation with anything — reported as 0 rather than ``nan``, because a
    ``nan`` propagating into the allocation maths silently zeroes somebody's
    capital.
    """
    names = sorted(by_agent)
    series = [return_series(by_agent[name]) for name in names]
    if not series:
        return names, np.zeros((0, 0))

    length = min((s.size for s in series if s.size), default=0)
    if length < 2:
        return names, np.eye(len(names))

    stacked = np.vstack([s[:length] for s in series])
    matrix = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = stacked[i], stacked[j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                value = 0.0
            else:
                value = float(np.corrcoef(a, b)[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            matrix[i, j] = matrix[j, i] = value
    return names, matrix


def average_correlations(
    by_agent: Mapping[str, Sequence[BacktestResult]]
) -> dict[str, float]:
    """Each agent's mean correlation with the rest of the desk."""
    names, matrix = correlation_matrix(by_agent)
    if len(names) < 2:
        return {name: 0.0 for name in names}
    out: dict[str, float] = {}
    for index, name in enumerate(names):
        others = [matrix[index, j] for j in range(len(names)) if j != index]
        out[name] = float(np.mean(others))
    return out


class JojoManager:
    """Ranks agents, allocates paper capital, promotes and retires champions."""

    def __init__(
        self,
        store: ExperimentStore,
        *,
        allocation: AllocationConfig | None = None,
        retirement: RetirementConfig | None = None,
        metric: str = "sortino",
    ) -> None:
        self.store = store
        self.allocation_config = allocation or AllocationConfig()
        self.allocation_config.validate()
        self.retirement_config = retirement or RetirementConfig()
        self.retirement_config.validate()
        self.metric = metric

    # ---- ranking and allocation -------------------------------------------

    def rank(
        self, by_agent: Mapping[str, Sequence[BacktestResult]]
    ) -> list[AgentRanking]:
        """Rank on median score, with a haircut for correlation to the desk."""
        correlations = average_correlations(by_agent)
        rankings = [
            AgentRanking(
                agent=agent,
                score=_median(results, self.metric),
                drawdown_pct=_median(results, "max_drawdown_pct"),
                trades=int(sum(r.metrics["trades"] for r in results)),
                runs=len(results),
                correlation=correlations.get(agent, 0.0),
            )
            for agent, results in by_agent.items()
        ]
        rankings.sort(key=lambda r: (-r.score if np.isfinite(r.score) else np.inf, r.agent))
        return rankings

    def allocate(
        self, by_agent: Mapping[str, Sequence[BacktestResult]]
    ) -> list[AgentRanking]:
        """Assign paper capital. Never more than the cap, never past the reserve."""
        cfg = self.allocation_config
        rankings = self.rank(by_agent)
        deployable = cfg.total_paper_capital * (1.0 - cfg.reserve_fraction)

        shares: dict[str, float] = {}
        haircuts: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for ranking in rankings:
            if not np.isfinite(ranking.score) or ranking.score <= cfg.min_score:
                shares[ranking.agent] = 0.0
                haircuts[ranking.agent] = 0.0
                reasons[ranking.agent] = (
                    f"score {ranking.score:.3f} at or below {cfg.min_score:.2f} "
                    "— beaten by doing nothing"
                )
                continue
            shares[ranking.agent] = ranking.score
            # Only positive correlation is penalised. An agent that is genuinely
            # uncorrelated, or negatively so, is diversification and should not
            # be charged for it.
            haircuts[ranking.agent] = max(
                0.05, 1.0 - cfg.correlation_penalty * max(0.0, ranking.correlation))
            reasons[ranking.agent] = (
                f"score {ranking.score:.3f}, corr {ranking.correlation:+.2f} "
                f"-> x{haircuts[ranking.agent]:.2f}"
            )

        total_share = sum(shares.values())
        allocations: dict[str, float] = {agent: 0.0 for agent in shares}
        if total_share <= 0:
            logger.warning("no agent scores above %.2f; allocating nothing", cfg.min_score)
        else:
            cap = deployable * cfg.max_share
            for agent, share in shares.items():
                # The haircut scales the agent's allocation *down*, it does not
                # merely reshuffle shares between agents. That distinction is the
                # whole point: on a desk where everyone has found the same edge,
                # every haircut is large and the desk deploys less capital — which
                # is the correct response to concentration. Applying it only to
                # relative weights would leave a fully converged desk deploying
                # exactly as much as a diversified one.
                allocations[agent] = min(deployable * (share / total_share)
                                         * haircuts[agent], cap)
            spare = deployable - sum(allocations.values())
            if spare > 1e-9:
                logger.info("%.0f of paper capital held back (correlation haircut "
                            "and/or share cap)", spare)

        return [
            AgentRanking(
                agent=r.agent, score=r.score, drawdown_pct=r.drawdown_pct,
                trades=r.trades, runs=r.runs, correlation=r.correlation,
                allocation=round(allocations.get(r.agent, 0.0), 2),
                reason=reasons.get(r.agent, ""),
            )
            for r in rankings
        ]

    def record_allocations(self, rankings: Sequence[AgentRanking]) -> None:
        for ranking in rankings:
            self.store.record_allocation(
                agent=ranking.agent, paper_capital=ranking.allocation,
                reason=ranking.reason or "ranked allocation",
            )

    # ---- champion and challenger ------------------------------------------

    def consider_promotion(
        self,
        agent: str,
        policy_id: str,
        challenger: Sequence[BacktestResult],
        *,
        baselines: Mapping[str, Sequence[BacktestResult]],
        champion_results: Sequence[BacktestResult] | None = None,
        regime_metrics: Mapping[str, object] | None = None,
        overall_drawdown_pct: float | None = None,
        gate_config: GateConfig | None = None,
    ) -> GateResult:
        """Run the gate and, if it passes, make this policy the agent's champion.

        Promotion here is a *research* status. It does not deploy anything: live
        trading stays behind the existing three-part gate and an explicit human
        decision, per policy and per venue.
        """
        gate = evaluate_gate(
            policy_id, challenger, baselines=baselines,
            champion=champion_results, regime_metrics=regime_metrics,
            overall_drawdown_pct=overall_drawdown_pct, config=gate_config,
        )
        if gate.passed:
            self.store.promote(agent=agent, policy_id=policy_id, reason=gate.reason())
            logger.info("%s: %s promoted to champion", agent, policy_id)
        else:
            logger.info("%s: %s stays a challenger — %s", agent, policy_id, gate.reason())
        return gate

    def champion(self, agent: str) -> dict | None:
        return self.store.champion(agent)

    def consider_retirement(
        self,
        agent: str,
        recent: Sequence[BacktestResult],
        *,
        baselines: Mapping[str, Sequence[BacktestResult]],
        promoted_score: float | None = None,
    ) -> RetirementDecision:
        """Retire a champion whose edge has decayed.

        Retirement is automatic where promotion is not, and the asymmetry is
        deliberate: standing down reduces exposure, so erring toward it is
        cheap, while deploying is the direction that costs money when wrong.
        """
        cfg = self.retirement_config
        champion = self.store.champion(agent)
        if champion is None:
            return RetirementDecision(agent, "", False, "no champion to retire")

        policy_id = champion["policy_id"]
        windows = sorted({r.window for r in recent if r.window is not None})
        recent_windows = windows[-cfg.lookback_windows:]
        scored = [r for r in recent if r.window in recent_windows]
        if not scored:
            return RetirementDecision(agent, policy_id, False,
                                      "no recent results to judge")

        current = _median(scored, self.metric)
        below = 0
        for window in recent_windows:
            mine = _median([r for r in scored if r.window == window], self.metric)
            best = max((_median([r for r in runs if r.window == window], self.metric)
                        for runs in baselines.values()), default=float("-inf"))
            if np.isfinite(mine) and np.isfinite(best) and mine < best:
                below += 1

        if below >= cfg.max_windows_below_baseline:
            reason = (f"lost to the best baseline in {below} of the last "
                      f"{len(recent_windows)} window(s)")
            self.store.retire(agent=agent, policy_id=policy_id, reason=reason)
            return RetirementDecision(agent, policy_id, True, reason)

        if promoted_score is not None and np.isfinite(current):
            floor = promoted_score * cfg.decay_tolerance
            # A champion promoted on a positive score has decayed if it falls
            # below a fraction of it; one promoted on a negative score should
            # never have been promoted, so any further decline retires it.
            decayed = current < floor if promoted_score > 0 else current < promoted_score
            if decayed:
                reason = (f"{self.metric} fell to {current:.3f} from {promoted_score:.3f} "
                          f"at promotion")
                self.store.retire(agent=agent, policy_id=policy_id, reason=reason)
                return RetirementDecision(agent, policy_id, True, reason)

        return RetirementDecision(agent, policy_id, False,
                                  f"{self.metric} {current:.3f} still within tolerance")

    # ---- reporting ---------------------------------------------------------

    def desk_report(self, by_agent: Mapping[str, Sequence[BacktestResult]]) -> str:
        rankings = self.allocate(by_agent)
        names, matrix = correlation_matrix(by_agent)
        lines = [
            f"{'agent':<14}{self.metric:>9}{'maxDD':>10}{'trades':>8}{'corr':>8}"
            f"{'paper $':>12}   note",
            "-" * 100,
        ]
        lines.extend(r.describe() for r in rankings)

        if len(names) > 1:
            off_diagonal = matrix[np.triu_indices(len(names), k=1)]
            mean_correlation = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
            lines.append("")
            lines.append(f"desk mean pairwise correlation: {mean_correlation:+.2f}")
            if mean_correlation > 0.7:
                lines.append(
                    "  WARNING: the agents have converged. Five policies this correlated "
                    "are one bet with five names on it, and the desk is far more "
                    "concentrated than an equal split suggests."
                )
            deployed = sum(r.allocation for r in rankings)
            lines.append(
                f"deployed {deployed:,.0f} of {self.allocation_config.total_paper_capital:,.0f} "
                f"paper capital ({self.allocation_config.reserve_fraction:.0%} reserved"
                + (", plus capping spare)" if deployed < self.allocation_config.total_paper_capital
                   * (1 - self.allocation_config.reserve_fraction) - 1 else ")")
            )
        return "\n".join(lines)
