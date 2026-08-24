"""Research agents: five identities, five search distributions, one method.

Each agent starts with a **bias** — JOLYAN explores, JONATHAN looks for trend,
JOSEPH watches regime, JOTARO minds risk, KIRA fades the crowd. The important
word is *starts*. A bias here is a prior over the search distribution, not a
strategy and not a constraint:

* it skews which features are **likely** to be sampled, never which are possible;
* it is blended with observed evidence, and the blend shifts toward evidence as
  experiments accumulate;
* after enough contrary results an agent's effective weights can invert
  completely, which is the point. JONATHAN is *allowed to discover that trend
  following does not work here*, and if the evidence says so, it will stop
  proposing trend features without anyone editing its definition.

The alternative — hard-coding each agent to a strategy family — is the thing
this whole phase exists to replace. An agent that cannot be wrong cannot learn.

**Memory is what stops the search going in circles.** Every experiment records
its spec, its gate result and its score. The generator rejects proposals that
are near-duplicates of known failures, and the rejection distance widens as the
failure count grows, so a search that keeps rediscovering the same dead end is
pushed further away from it each time.

Hypothesis generation here is **programmatic, not an LLM** — a seeded search over
a typed space. That keeps the whole pipeline deterministic and testable before
any language model is involved; the LLM enters in Phase 6 as an additional
proposer subject to identical gates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .datasets import Dataset
from .evaluate import GateConfig, GateResult, evaluate_gate, regime_metrics_across_seeds
from .experiments import EvaluationStage, ExperimentStatus, ExperimentStore, MemoryKind
from .features import available_features
from .hypothesis import ExperimentSpec, SearchRanges, sample_spec
from .regimes import Regime, label_regimes
from .splits import SplitPlan

logger = logging.getLogger("research.agents")

AGENT_VERSION = "v1"

#: How much evidence it takes to half-outweigh the starting bias. Low enough
#: that a genuinely wrong prior is overturned within a normal campaign, high
#: enough that three unlucky experiments do not flip an agent's identity.
PRIOR_STRENGTH = 8.0

#: A proposal closer than this to a known failure is rejected and redrawn.
BASE_REJECT_DISTANCE = 0.15
MAX_REJECT_DISTANCE = 0.45


@dataclass(frozen=True)
class AgentBias:
    """A starting lean, expressed as feature affinities and range preferences."""

    name: str
    description: str
    favoured: tuple[str, ...] = ()
    disfavoured: tuple[str, ...] = ()
    favour_weight: float = 3.0
    disfavour_weight: float = 0.4
    ranges: SearchRanges = field(default_factory=SearchRanges)

    def initial_weights(self) -> dict[str, float]:
        weights = {name: 1.0 for name in available_features()}
        for name in self.favoured:
            if name in weights:
                weights[name] = self.favour_weight
        for name in self.disfavoured:
            if name in weights:
                weights[name] = self.disfavour_weight
        return weights


TREND_FEATURES = ("ema_spread", "trend_slope", "macd_hist", "ret_60", "ret_15",
                  "dist_from_sma")
REVERSION_FEATURES = ("rsi_14", "rsi_7", "bb_position", "dist_from_sma", "upper_wick",
                      "lower_wick")
REGIME_FEATURES = ("vol_20", "vol_60", "vol_ratio", "atr_pct", "bb_width", "volume_ratio")


AGENT_BIASES: dict[str, AgentBias] = {
    "JOLYAN": AgentBias(
        name="JOLYAN",
        description="explorer — no feature preference, high entropy, widest search",
        favoured=(), disfavoured=(),
        # The explorer's edge is exploration, so it starts with more entropy and
        # a wider network range rather than a feature opinion.
        ranges=SearchRanges(ent_coef=(0.005, 0.08)),
    ),
    "JONATHAN": AgentBias(
        name="JONATHAN",
        description="trend — leans on directional persistence",
        favoured=TREND_FEATURES,
        disfavoured=("rsi_7", "bb_position"),
        ranges=SearchRanges(gamma=(0.98, 0.9995), episode_bars_choices=(1_000, 2_000, 4_000)),
    ),
    "JOSEPH": AgentBias(
        name="JOSEPH",
        description="adaptive — leans on volatility and regime context",
        favoured=REGIME_FEATURES,
        ranges=SearchRanges(net_arch_choices=((64, 64), (128, 64), (128, 128))),
    ),
    "JOTARO": AgentBias(
        name="JOTARO",
        description="risk-aware — small size, heavy drawdown and tail penalties",
        favoured=("atr_pct", "vol_20", "vol_60", "vol_ratio"),
        ranges=SearchRanges(
            risk_per_trade=(0.002, 0.02),
            leverage=(1.0, 3.0),
            lambda_drawdown=(1.0, 4.0),
            lambda_tail=(0.5, 3.0),
        ),
    ),
    "KIRA": AgentBias(
        name="KIRA",
        description="contrarian — fades extension, expects reversion",
        favoured=REVERSION_FEATURES,
        disfavoured=("trend_slope", "ema_spread"),
        ranges=SearchRanges(
            episode_bars_choices=(500, 1_000, 2_000),
            take_profit_choices=(0.5, 0.75, 1.0, 1.5, 2.0),
        ),
    ),
}


@dataclass
class ExperimentOutcome:
    spec: ExperimentSpec
    experiment_id: str
    score: float
    gate: GateResult | None
    trades: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def promoted(self) -> bool:
        return bool(self.gate and self.gate.passed)


class AgentMemory:
    """Structured recall, backed by the ``experiments`` and ``agent_memory`` tables."""

    def __init__(self, store: ExperimentStore, agent: str) -> None:
        self.store = store
        self.agent = agent

    def past_specs(self) -> list[tuple[ExperimentSpec, float, bool]]:
        """``(spec, score, passed)`` for everything this agent has run."""
        out: list[tuple[ExperimentSpec, float, bool]] = []
        for record in self.store.recall(self.agent, MemoryKind.OUTCOME, limit=1_000):
            content = record["content"]
            try:
                spec = ExperimentSpec.from_dict(content["spec"])
            except Exception:
                logger.warning("skipping unreadable memory row for %s", self.agent)
                continue
            out.append((spec, float(content.get("score", 0.0)),
                        bool(content.get("passed", False))))
        return out

    def failures(self) -> list[ExperimentSpec]:
        return [spec for spec, _, passed in self.past_specs() if not passed]

    def remember_outcome(self, outcome: ExperimentOutcome) -> None:
        self.store.remember(
            agent=self.agent, kind=MemoryKind.OUTCOME,
            content={
                "spec": outcome.spec.as_dict(),
                "spec_id": outcome.spec.spec_id,
                "score": outcome.score,
                "passed": outcome.promoted,
                "trades": outcome.trades,
                "error": outcome.error,
                "gate": outcome.gate.as_dict() if outcome.gate else None,
            },
            experiment_id=outcome.experiment_id,
        )
        if not outcome.promoted and outcome.gate:
            self.store.remember(
                agent=self.agent, kind=MemoryKind.DEAD_END,
                content={"spec_id": outcome.spec.spec_id,
                         "features": sorted(outcome.spec.features),
                         "why": outcome.gate.reason()},
                experiment_id=outcome.experiment_id,
            )

    def feature_evidence(self) -> dict[str, float]:
        """Mean score of experiments that included each feature, centred on 0.

        Crude on purpose. Attributing a policy's score to individual features is
        not identifiable from this data — features appear in bundles — so this is
        a weak signal, and it is deliberately given weak authority: it moves the
        prior, it does not replace it.
        """
        history = self.past_specs()
        if not history:
            return {}
        scores = np.array([score for _, score, _ in history], dtype=np.float64)
        baseline = float(np.median(scores))
        spread = float(np.std(scores)) or 1.0

        totals: dict[str, list[float]] = {}
        for spec, score, _ in history:
            for name in spec.features:
                totals.setdefault(name, []).append((score - baseline) / spread)
        return {name: float(np.mean(values)) for name, values in totals.items()}


class ResearchAgent:
    """One identity: a bias, a memory, a generator and a runner."""

    def __init__(
        self,
        bias: AgentBias,
        store: ExperimentStore,
        *,
        seed: int = 0,
        prior_strength: float = PRIOR_STRENGTH,
    ) -> None:
        self.bias = bias
        self.name = bias.name
        self.store = store
        self.memory = AgentMemory(store, bias.name)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.prior_strength = prior_strength

    # ---- the bias, and how it stops being one ------------------------------

    def effective_weights(self) -> dict[str, float]:
        """Prior blended with evidence, shifting toward evidence over time.

        With no history this is exactly the starting bias. As outcomes
        accumulate the blend moves: at ``prior_strength`` observations the two
        contribute equally, and beyond that evidence dominates. An agent whose
        starting assumption is wrong therefore stops acting on it — without
        anyone editing its definition, which is the whole point of calling it a
        prior instead of a strategy.
        """
        prior = self.bias.initial_weights()
        evidence = self.memory.feature_evidence()
        if not evidence:
            return prior

        observations = float(len(self.memory.past_specs()))
        evidence_share = observations / (observations + self.prior_strength)
        weights: dict[str, float] = {}
        for name, prior_weight in prior.items():
            # Evidence is a z-like score; map it to a multiplicative factor that
            # stays positive so no feature is ever ruled out entirely.
            factor = float(np.exp(np.clip(evidence.get(name, 0.0), -2.0, 2.0)))
            weights[name] = float(
                prior_weight ** (1.0 - evidence_share) * factor ** evidence_share
            )
        return weights

    def bias_drift(self) -> float:
        """How far the effective weights have moved from the starting bias, in [0, 1]."""
        prior = self.bias.initial_weights()
        current = self.effective_weights()
        names = sorted(prior)
        a = np.array([prior[n] for n in names]); a = a / a.sum()
        b = np.array([current[n] for n in names]); b = b / b.sum()
        return float(np.abs(a - b).sum() / 2.0)

    # ---- proposing ---------------------------------------------------------

    def reject_distance(self) -> float:
        """Widen the exclusion zone as dead ends pile up."""
        failures = len(self.memory.failures())
        return float(min(MAX_REJECT_DISTANCE,
                         BASE_REJECT_DISTANCE + 0.01 * failures))

    def propose(self, *, total_timesteps: int = 200_000, attempts: int = 40) -> ExperimentSpec:
        """Draw a spec that is not a near-duplicate of something already tried."""
        known = [spec for spec, _, _ in self.memory.past_specs()]
        threshold = self.reject_distance()
        weights = self.effective_weights()

        best: ExperimentSpec | None = None
        best_distance = -1.0
        for _ in range(attempts):
            candidate = sample_spec(
                self.rng, feature_weights=weights, ranges=self.bias.ranges,
                total_timesteps=total_timesteps,
            )
            if not known:
                return candidate
            nearest = min(candidate.distance(other) for other in known)
            if nearest >= threshold:
                return candidate
            if nearest > best_distance:
                best, best_distance = candidate, nearest

        # Nothing far enough turned up. Returning the most distant candidate
        # beats looping forever, and it is honest about the search having
        # saturated rather than pretending it found something novel.
        logger.info("%s: search saturating (nearest %.3f < %.3f after %d draws)",
                    self.name, best_distance, threshold, attempts)
        return best  # type: ignore[return-value]

    # ---- running -----------------------------------------------------------

    def run_experiment(
        self,
        spec: ExperimentSpec,
        dataset: Dataset,
        plan: SplitPlan,
        *,
        runner: Callable[..., tuple[float, GateResult | None, int]],
        hypothesis: str | None = None,
    ) -> ExperimentOutcome:
        """Record, execute, gate and remember one experiment.

        ``runner`` does the expensive part (train + evaluate) and is injected so
        the agent logic can be tested without a GPU-hour. It returns
        ``(score, gate_result, trades)``.
        """
        experiment_id = self.store.create_experiment(
            agent=self.name,
            hypothesis=hypothesis or self._hypothesis_for(spec),
            spec=spec.as_dict(), seed=self.seed,
        )
        self.store.set_experiment_status(experiment_id, ExperimentStatus.TRAINING)

        try:
            score, gate, trades = runner(spec=spec, dataset=dataset, plan=plan,
                                         experiment_id=experiment_id)
        except Exception as exc:                      # a failed run is data too
            logger.exception("%s: experiment %s failed", self.name, experiment_id)
            self.store.set_experiment_status(
                experiment_id, ExperimentStatus.REJECTED, f"run failed: {exc}"[:400])
            outcome = ExperimentOutcome(spec=spec, experiment_id=experiment_id,
                                        score=float("-inf"), gate=None, trades=0,
                                        error=str(exc))
            self.memory.remember_outcome(outcome)
            return outcome

        self.store.set_experiment_status(experiment_id, ExperimentStatus.VALIDATING)
        passed = bool(gate and gate.passed)
        # The store refuses a rejection with no reason, and rightly so — a
        # rejection that teaches nothing gets proposed again next week. Make sure
        # there is always something to record even when the gate came back terse.
        reason = ""
        if not passed:
            reason = (gate.reason() if gate else "").strip() or (
                f"rejected with score {score:.4f} and {trades} trade(s); "
                "the gate returned no failing condition"
            )
        self.store.set_experiment_status(
            experiment_id,
            ExperimentStatus.CANDIDATE if passed else ExperimentStatus.REJECTED,
            reason[:400],
        )

        outcome = ExperimentOutcome(spec=spec, experiment_id=experiment_id, score=score,
                                    gate=gate, trades=trades)
        self.memory.remember_outcome(outcome)
        logger.info("%s %s: score %.3f %s", self.name, spec.spec_id, score,
                    "CANDIDATE" if passed else "rejected")
        return outcome

    def campaign(
        self,
        dataset: Dataset,
        plan: SplitPlan,
        *,
        runner: Callable[..., tuple[float, GateResult | None, int]],
        experiments: int = 5,
        total_timesteps: int = 200_000,
    ) -> list[ExperimentOutcome]:
        """Propose, run and learn from ``experiments`` hypotheses in sequence.

        Sequential rather than batched so each proposal sees the previous
        outcome — an agent that cannot react to its own last result is running a
        random search with extra steps.
        """
        outcomes: list[ExperimentOutcome] = []
        for index in range(experiments):
            spec = self.propose(total_timesteps=total_timesteps)
            logger.info("%s experiment %d/%d: %s", self.name, index + 1, experiments,
                        spec.describe())
            outcomes.append(self.run_experiment(spec, dataset, plan, runner=runner))
        return outcomes

    def _hypothesis_for(self, spec: ExperimentSpec) -> str:
        leaning = ", ".join(sorted(set(spec.features) & set(self.bias.favoured))[:4])
        return (
            f"{self.bias.description}. {len(spec.features)} features"
            + (f", incl. {leaning}" if leaning else "")
            + f"; risk {spec.risk_per_trade:.1%} at {spec.leverage:.0f}x, "
            f"reward dd={spec.reward.lambda_drawdown:.2f} tail={spec.reward.lambda_tail:.2f}"
        )

    def summary(self) -> str:
        history = self.memory.past_specs()
        passed = sum(1 for _, _, ok in history if ok)
        drift = self.bias_drift()
        return (
            f"{self.name:<9} {self.bias.description}\n"
            f"          {len(history)} experiments, {passed} candidate(s), "
            f"bias drift {drift:.0%}"
        )


def build_agents(store: ExperimentStore, *, seed: int = 0,
                 names: Sequence[str] | None = None) -> list[ResearchAgent]:
    chosen = list(names) if names else list(AGENT_BIASES)
    unknown = [n for n in chosen if n not in AGENT_BIASES]
    if unknown:
        raise ValueError(f"unknown agent(s) {unknown}; available: {sorted(AGENT_BIASES)}")
    return [
        ResearchAgent(AGENT_BIASES[name], store, seed=seed + index)
        for index, name in enumerate(chosen)
    ]
