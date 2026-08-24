"""Turning a proposed spec into a scored, gated result.

This is the expensive half of an agent's loop: train the spec's policy on the
training window, walk it forward over every validation window, compare it to the
baselines, and put the whole thing through the promotion gate.

**Baselines run under a fixed, neutral envelope — never the agent's.** That is a
deliberate safeguard rather than a convenience. If the baselines inherited the
spec's proposed risk and stop parameters, an agent could clear the
"beats_best_baseline" condition by proposing an envelope that cripples
buy-and-hold rather than by learning anything, and the gate would wave it
through. Holding the baselines fixed means the comparison always asks the
question that matters: can this policy, using the envelope it wants, beat a
sensible strategy using a sensible one?

Baselines are also cached per (dataset, plan, seeds) because they do not depend
on the spec at all, and recomputing nine of them for every experiment would
dominate the runtime of a campaign.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.config import load_settings

from .backtest import BacktestConfig, BacktestResult, run_backtest, run_walk_forward
from .datasets import Dataset
from .evaluate import GateConfig, GateResult, evaluate_gate, regime_metrics_across_seeds
from .experiments import EvaluationStage, ExperimentStore
from .hypothesis import ExperimentSpec
from .policy import BASELINE_POLICIES, STRATEGY_POLICIES, make_policy
from .regimes import Regime, label_regimes
from .splits import SplitPlan, Window

logger = logging.getLogger("research.campaign")

#: The envelope every baseline is measured under, whatever an agent proposes.
REFERENCE_CONFIG = BacktestConfig(
    starting_equity=10_000.0, risk_per_trade=0.02, leverage=1.0,
    stop_loss_pct=1.0, take_profit_pct=7.0,
)
NO_EXIT_POLICIES = {"flat", "buy_and_hold", "always_short"}


def backtest_config_for(spec: ExperimentSpec) -> BacktestConfig:
    return BacktestConfig(
        starting_equity=REFERENCE_CONFIG.starting_equity,
        risk_per_trade=spec.risk_per_trade,
        leverage=spec.leverage,
        stop_loss_pct=spec.stop_loss_pct,
        take_profit_pct=spec.take_profit_pct,
        sizing_stop_pct=spec.stop_loss_pct or 5.0,
    )


def baseline_config_for(name: str) -> BacktestConfig:
    """A hold-forever baseline with a 1% stop is not a hold-forever baseline."""
    if name in NO_EXIT_POLICIES:
        from dataclasses import replace as dc_replace

        return dc_replace(REFERENCE_CONFIG, stop_loss_pct=None, take_profit_pct=None,
                          sizing_stop_pct=5.0)
    return REFERENCE_CONFIG


class BaselineCache:
    """Baseline walk-forwards, computed once per (dataset, plan, seeds)."""

    def __init__(self) -> None:
        self._cache: dict[tuple, dict[str, list[BacktestResult]]] = {}

    def get(
        self, dataset: Dataset, plan: SplitPlan, seeds: Sequence[int],
        *, names: Sequence[str] | None = None,
    ) -> dict[str, list[BacktestResult]]:
        chosen = tuple(names) if names else tuple(BASELINE_POLICIES) + tuple(STRATEGY_POLICIES)
        key = (dataset.manifest.sha256, plan.split_id, tuple(seeds), chosen)
        if key in self._cache:
            return self._cache[key]

        settings = load_settings()
        logger.info("computing %d baselines over %d windows x %d seed(s)",
                    len(chosen), len(plan), len(seeds))
        results = {
            name: run_walk_forward(
                lambda n=name: make_policy(n, settings.strategy_parameters, seed=0),
                dataset, plan, config=baseline_config_for(name), seeds=tuple(seeds),
            )
            for name in chosen
        }
        self._cache[key] = results
        return results


@dataclass
class CampaignConfig:
    """How much compute one experiment gets."""

    train_window: int = 0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    total_timesteps: int = 200_000
    n_envs: int = 4
    min_regime_bars: int = 200
    gate: GateConfig = None            # type: ignore[assignment]
    root: str | Path = "data"

    def __post_init__(self) -> None:
        if self.gate is None:
            self.gate = GateConfig(min_seeds=len(self.seeds))
        if not self.seeds:
            raise ValueError("a campaign needs at least one seed")


def make_runner(
    *,
    config: CampaignConfig | None = None,
    baselines: BaselineCache | None = None,
    store: ExperimentStore | None = None,
    agent_name: str = "agent",
):
    """Build the ``runner`` callable :meth:`ResearchAgent.run_experiment` expects.

    Returns ``(score, gate_result, trades)``. The score is the median Sortino
    across every window and seed — the same statistic the gate's primary
    condition reads, so an agent's search is optimising the thing it is judged
    on rather than a proxy for it.
    """
    from .learned import LearnedPolicy
    from .train import TrainConfig, train_seed_ensemble

    cfg = config or CampaignConfig()
    cache = baselines or BaselineCache()

    def run(*, spec: ExperimentSpec, dataset: Dataset, plan: SplitPlan,
            experiment_id: str) -> tuple[float, GateResult | None, int]:
        window: Window = plan.windows[cfg.train_window]
        backtest_config = backtest_config_for(spec)

        train_config = TrainConfig(
            total_timesteps=spec.total_timesteps or cfg.total_timesteps,
            n_steps=spec.n_steps,
            batch_size=min(256, spec.n_steps * cfg.n_envs),
            learning_rate=spec.learning_rate,
            gamma=spec.gamma,
            ent_coef=spec.ent_coef,
            net_arch=spec.net_arch,
            episode_bars=min(spec.episode_bars, len(window.train) - 1),
            n_envs=cfg.n_envs,
        )

        ensemble = train_seed_ensemble(
            dataset, agent=agent_name,
            train_start=window.train.start, train_stop=window.train.stop,
            seeds=cfg.seeds, version=1, train_config=train_config,
            root=cfg.root, feature_names=spec.features,
            reward_config=spec.reward, backtest_config=backtest_config,
        )
        # The policy id carries the experiment, not just the agent, so two
        # experiments by the same agent cannot overwrite each other's artefacts.
        for trained in ensemble:
            trained.metadata["experiment_id"] = experiment_id

        challenger: list[BacktestResult] = []
        for w in plan.windows:
            for seed, trained in zip(cfg.seeds, ensemble):
                policy = LearnedPolicy.from_directory(trained.directory, dataset)
                challenger.append(run_backtest(
                    policy, dataset, start=w.validate.start, stop=w.validate.stop,
                    config=backtest_config, seed=seed, window=w.index,
                ))

        baseline_results = cache.get(dataset, plan, cfg.seeds)
        labels = label_regimes(dataset)
        regime_metrics, stitched_drawdown = regime_metrics_across_seeds(
            challenger, labels, bars_per_year=dataset.bars_per_day * 365.0,
            min_bars=cfg.min_regime_bars, regime_name=lambda code: Regime(code).label,
        )
        regime_metrics.pop(Regime.UNKNOWN.label, None)

        gate = evaluate_gate(
            f"{agent_name}_{spec.spec_id}", challenger, baselines=baseline_results,
            regime_metrics=regime_metrics or None,
            overall_drawdown_pct=stitched_drawdown or None, config=cfg.gate,
        )
        score = float(np.median([r.metrics["sortino"] for r in challenger]))
        trades = int(sum(r.metrics["trades"] for r in challenger))

        if store is not None:
            store.register_dataset(dataset.manifest)
            policy_id = f"{agent_name}_{spec.spec_id}"
            for seed, trained in zip(cfg.seeds, ensemble):
                store.register_policy(
                    agent=agent_name, kind="learned", algo="PPO",
                    experiment_id=experiment_id,
                    hyperparams=spec.as_dict(), feature_set=spec.features,
                    reward_version=spec.reward.version, seed=seed,
                    model_path=str(trained.model_path),
                    weights_path=str(trained.weights_path),
                    policy_id=f"{policy_id}_s{seed}",
                    version=_next_version(store, agent_name),
                )
            for result in challenger:
                store.record_metrics(
                    policy_id=f"{policy_id}_s{result.seed}",
                    dataset_id=dataset.manifest.dataset_id, metrics=result.metrics,
                    split_id=plan.split_id, window=result.window, seed=result.seed,
                )
            store.record_evaluation(
                policy_id=f"{policy_id}_s{cfg.seeds[0]}",
                dataset_id=dataset.manifest.dataset_id, split_id=plan.split_id,
                stage=EvaluationStage.WALK_FORWARD, passed=gate.passed,
                gates=gate.as_dict(), reason=gate.reason(),
            )

        return score, gate, trades

    return run


def _next_version(store: ExperimentStore, agent: str) -> int:
    row = store.db.execute(
        "SELECT COALESCE(MAX(version), 0) AS v FROM policies WHERE agent = ?", (agent,)
    ).fetchone()
    return int(row["v"]) + 1
