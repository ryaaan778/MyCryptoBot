"""JOJO: ranking, allocation, correlation, promotion and retirement.

Two properties carry the most weight here. JOJO must not be able to reach the
risk engine — asserted against the import graph rather than a docstring — and
correlation must be treated as a risk rather than a win, because five agents
that found the same edge are one bet with five names on it.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from research.backtest import BacktestResult
from research.evaluate import Condition, GateResult
from research.experiments import ExperimentStore
from research.jojo import (
    AllocationConfig, JojoManager, RetirementConfig, average_correlations,
    correlation_matrix, return_series,
)
from research.metrics import MetricSet


@pytest.fixture
def store(tmp_path):
    with ExperimentStore(tmp_path / "research.db") as opened:
        yield opened


@pytest.fixture
def jojo(store):
    return JojoManager(store)


def run(equity, *, window=0, seed=0, sortino=1.0, drawdown=5.0, trades=20):
    array = np.asarray(equity, dtype=np.float64)
    values = {"sortino": sortino, "max_drawdown_pct": drawdown,
              "trades": float(trades), "total_return_pct": float(
                  (array[-1] / array[0] - 1) * 100)}
    return BacktestResult(
        policy="p", dataset_id="d", dataset_source="SYNTHETIC", symbol="BTC/USDT",
        timeframe="5m", start_index=window * 100, stop_index=window * 100 + array.size,
        ts=np.arange(array.size, dtype=np.int64) + window * 100,
        equity=array, trades=[],
        metrics=MetricSet(values=values, trades=trades, bars=array.size),
        actions=np.zeros(array.size, dtype=np.int8), seed=seed, window=window,
    )


def walk(returns, *, sortino=1.0, drawdown=5.0, trades=20, windows=3, seeds=(0,)):
    """One agent's results: the same return pattern repeated over windows/seeds."""
    out = []
    for window in range(windows):
        for seed in seeds:
            equity = 10_000.0 * np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
            equity = np.concatenate([[10_000.0], equity])
            out.append(run(equity, window=window, seed=seed, sortino=sortino,
                           drawdown=drawdown, trades=trades))
    return out


# ---- the structural guarantee ---------------------------------------------


def test_jojo_cannot_reach_the_risk_engine():
    """Asserted against the import graph, not a promise in a docstring."""
    tree = ast.parse(pathlib.Path("research/jojo.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{alias.name}" for alias in node.names)

    forbidden = {"backend.risk", "backend.config", "backend.orchestrator",
                 "backend.execution", "backend.app"}
    assert not (imported & forbidden), sorted(imported & forbidden)
    assert not any(name.endswith("RiskEngine") for name in imported)


def test_jojos_only_lever_is_paper_capital(jojo, store):
    """Allocation writes to `allocations` and nothing else changes."""
    before = store.counts()
    rankings = jojo.allocate({"A": walk([0.001] * 20, sortino=2.0)})
    jojo.record_allocations(rankings)
    after = store.counts()
    changed = {k for k in after if after[k] != before.get(k, 0)}
    assert changed == {"allocations"}, changed


# ---- correlation -----------------------------------------------------------


def test_identical_agents_are_perfectly_correlated():
    pattern = list(np.random.default_rng(0).normal(0, 0.01, 200))
    by_agent = {"A": walk(pattern), "B": walk(pattern)}
    names, matrix = correlation_matrix(by_agent)
    assert names == ["A", "B"]
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_opposite_agents_are_negatively_correlated():
    pattern = list(np.random.default_rng(1).normal(0, 0.01, 200))
    by_agent = {"A": walk(pattern), "B": walk([-x for x in pattern])}
    _, matrix = correlation_matrix(by_agent)
    assert matrix[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_an_idle_agent_correlates_with_nothing_rather_than_producing_nan():
    """A nan propagating into the allocation maths silently zeroes someone."""
    pattern = list(np.random.default_rng(2).normal(0, 0.01, 200))
    by_agent = {"A": walk(pattern), "IDLE": walk([0.0] * 200)}
    _, matrix = correlation_matrix(by_agent)
    assert np.all(np.isfinite(matrix))
    assert matrix[0, 1] == 0.0


def test_the_window_reset_is_not_counted_as_a_return():
    """Each window restarts at the same capital; that jump is nobody's return."""
    results = walk([0.01] * 5, windows=3)
    series = return_series(results)
    assert series.size == 15                       # 5 returns per window, resets dropped
    assert np.allclose(series, 0.01)


def test_average_correlation_excludes_self():
    pattern = list(np.random.default_rng(3).normal(0, 0.01, 200))
    by_agent = {"A": walk(pattern), "B": walk(pattern), "C": walk([-x for x in pattern])}
    averages = average_correlations(by_agent)
    assert averages["A"] == pytest.approx(0.0, abs=1e-6)     # +1 with B, -1 with C
    assert averages["C"] == pytest.approx(-1.0, abs=1e-6)


def test_a_single_agent_desk_has_no_correlation():
    assert average_correlations({"A": walk([0.001] * 20)}) == {"A": 0.0}


# ---- ranking and allocation ------------------------------------------------


def test_ranking_orders_by_score(jojo):
    by_agent = {"WEAK": walk([0.0001] * 50, sortino=0.2),
                "STRONG": walk([0.001] * 50, sortino=3.0),
                "MIDDLING": walk([0.0005] * 50, sortino=1.0)}
    assert [r.agent for r in jojo.rank(by_agent)] == ["STRONG", "MIDDLING", "WEAK"]


def test_an_agent_beaten_by_doing_nothing_gets_nothing(jojo):
    by_agent = {"GOOD": walk([0.001] * 50, sortino=2.0),
                "LOSER": walk([-0.001] * 50, sortino=-1.5)}
    allocations = {r.agent: r.allocation for r in jojo.allocate(by_agent)}
    assert allocations["LOSER"] == 0.0
    assert allocations["GOOD"] > 0
    loser = next(r for r in jojo.allocate(by_agent) if r.agent == "LOSER")
    assert "doing nothing" in loser.reason


def test_correlation_is_penalised_not_rewarded(jojo):
    """Two agents with the same edge must not be funded as if they were two edges."""
    rng = np.random.default_rng(4)
    shared = list(rng.normal(0.0005, 0.01, 300))
    independent = list(rng.normal(0.0005, 0.01, 300))

    twins = jojo.allocate({"A": walk(shared, sortino=2.0), "B": walk(shared, sortino=2.0)})
    diverse = jojo.allocate({"A": walk(shared, sortino=2.0),
                             "B": walk(independent, sortino=2.0)})
    assert sum(r.allocation for r in twins) < sum(r.allocation for r in diverse)


def test_negative_correlation_is_not_charged_for(jojo):
    """Genuine diversification should not be taxed as if it were crowding."""
    rng = np.random.default_rng(5)
    pattern = list(rng.normal(0.0005, 0.01, 300))
    hedged = jojo.allocate({"A": walk(pattern, sortino=2.0),
                            "B": walk([-x for x in pattern], sortino=2.0)})
    for ranking in hedged:
        assert "x1.00" in ranking.reason


def test_no_agent_exceeds_the_share_cap(jojo, store):
    manager = JojoManager(store, allocation=AllocationConfig(
        total_paper_capital=100_000.0, max_share=0.3, reserve_fraction=0.2))
    by_agent = {"DOMINANT": walk([0.002] * 50, sortino=50.0),
                "TINY": walk([0.0001] * 50, sortino=0.1)}
    allocations = manager.allocate(by_agent)
    deployable = 100_000.0 * 0.8
    for ranking in allocations:
        assert ranking.allocation <= deployable * 0.3 + 1e-6


def test_the_reserve_is_never_deployed(jojo, store):
    manager = JojoManager(store, allocation=AllocationConfig(
        total_paper_capital=100_000.0, reserve_fraction=0.25, max_share=1.0))
    by_agent = {f"A{i}": walk([0.001] * 50, sortino=1.0 + i) for i in range(4)}
    total = sum(r.allocation for r in manager.allocate(by_agent))
    assert total <= 100_000.0 * 0.75 + 1e-6


def test_capping_returns_capital_to_the_reserve_rather_than_redistributing(jojo, store):
    manager = JojoManager(store, allocation=AllocationConfig(
        total_paper_capital=100_000.0, max_share=0.3, reserve_fraction=0.0))
    by_agent = {"A": walk([0.001] * 50, sortino=10.0),
                "B": walk([0.001] * 50, sortino=0.5)}
    allocations = {r.agent: r.allocation for r in manager.allocate(by_agent)}
    assert allocations["A"] <= 30_000.0 + 1e-6
    assert sum(allocations.values()) < 100_000.0


def test_a_desk_where_nobody_earns_anything_allocates_nothing(jojo):
    by_agent = {"A": walk([-0.001] * 50, sortino=-2.0),
                "B": walk([-0.002] * 50, sortino=-3.0)}
    assert all(r.allocation == 0.0 for r in jojo.allocate(by_agent))


@pytest.mark.parametrize("override", [
    {"total_paper_capital": 0}, {"max_share": 0}, {"max_share": 1.5},
    {"reserve_fraction": 1.0}, {"correlation_penalty": -1}, {"min_share": 0.9},
])
def test_malformed_allocation_configs_are_rejected(override):
    with pytest.raises(ValueError):
        AllocationConfig(**override).validate()


# ---- promotion and retirement ----------------------------------------------


def passing_gate(passed: bool) -> list[Condition]:
    return [Condition("beats_best_baseline", passed, 1.0, 0.5,
                      "sortino 1.000 vs 0.500" if passed else "sortino 0.100 vs 0.500")]


def test_a_challenger_that_clears_the_gate_becomes_champion(jojo, store, monkeypatch):
    import research.jojo as jojo_module

    store.register_policy(agent="KIRA", kind="learned", policy_id="KIRA_v2", version=2)
    monkeypatch.setattr(jojo_module, "evaluate_gate",
                        lambda *a, **k: GateResult("KIRA_v2", True, passing_gate(True), {}))
    assert store.champion("KIRA") is None
    gate = jojo.consider_promotion("KIRA", "KIRA_v2", walk([0.001] * 20),
                                   baselines={"flat": walk([0.0] * 20)})
    assert gate.passed
    assert store.champion("KIRA")["policy_id"] == "KIRA_v2"


def test_a_challenger_that_fails_stays_a_challenger(jojo, store, monkeypatch):
    import research.jojo as jojo_module

    monkeypatch.setattr(jojo_module, "evaluate_gate",
                        lambda *a, **k: GateResult("KIRA_v2", False, passing_gate(False), {}))
    jojo.consider_promotion("KIRA", "KIRA_v2", walk([0.001] * 20),
                            baselines={"flat": walk([0.0] * 20)})
    assert store.champion("KIRA") is None


def test_promotion_is_a_research_status_and_deploys_nothing(jojo, store, monkeypatch):
    import research.jojo as jojo_module

    store.register_policy(agent="KIRA", kind="learned", policy_id="K_v2", version=2)
    monkeypatch.setattr(jojo_module, "evaluate_gate",
                        lambda *a, **k: GateResult("K_v2", True, passing_gate(True), {}))
    before = store.counts()
    jojo.consider_promotion("KIRA", "K_v2", walk([0.001] * 20),
                            baselines={"flat": walk([0.0] * 20)})
    after = store.counts()
    assert {k for k in after if after[k] != before.get(k, 0)} == {"champion_history"}


def test_a_champion_losing_to_baselines_is_retired(jojo, store):
    store.register_policy(agent="KIRA", kind="learned", policy_id="KIRA_v1", version=1)
    store.promote(agent="KIRA", policy_id="KIRA_v1", reason="first")
    decision = jojo.consider_retirement(
        "KIRA", walk([-0.001] * 20, sortino=-1.0, windows=3),
        baselines={"flat": walk([0.0] * 20, sortino=0.0, windows=3)})
    assert decision.retire and "lost to the best baseline" in decision.reason
    assert store.champion("KIRA") is None


def test_a_decayed_champion_is_retired(jojo, store):
    store.register_policy(agent="JOSEPH", kind="learned", policy_id="JOSEPH_v1", version=1)
    store.promote(agent="JOSEPH", policy_id="JOSEPH_v1", reason="promoted at 3.0")
    decision = jojo.consider_retirement(
        "JOSEPH", walk([0.0001] * 20, sortino=0.4, windows=3),
        baselines={"flat": walk([0.0] * 20, sortino=0.0, windows=3)},
        promoted_score=3.0)
    assert decision.retire and "fell to" in decision.reason


def test_a_champion_still_performing_is_kept(jojo, store):
    store.register_policy(agent="JOSEPH", kind="learned", policy_id="JOSEPH_v1", version=1)
    store.promote(agent="JOSEPH", policy_id="JOSEPH_v1", reason="promoted at 3.0")
    decision = jojo.consider_retirement(
        "JOSEPH", walk([0.001] * 20, sortino=2.8, windows=3),
        baselines={"flat": walk([0.0] * 20, sortino=0.0, windows=3)},
        promoted_score=3.0)
    assert not decision.retire and "within tolerance" in decision.reason
    assert store.champion("JOSEPH") is not None


def test_retiring_with_no_champion_is_a_no_op(jojo):
    decision = jojo.consider_retirement("NOBODY", walk([0.001] * 20), baselines={})
    assert not decision.retire and "no champion" in decision.reason


def test_retirement_is_automatic_where_promotion_is_not(jojo, store):
    """Standing down reduces exposure; deploying is the direction that costs money."""
    store.register_policy(agent="KIRA", kind="learned", policy_id="KIRA_v1", version=1)
    store.promote(agent="KIRA", policy_id="KIRA_v1", reason="first")
    jojo.consider_retirement("KIRA", walk([-0.01] * 20, sortino=-5.0, windows=3),
                             baselines={"flat": walk([0.0] * 20, sortino=0.0, windows=3)})
    history = store.champion_history("KIRA")
    assert history[0]["action"] == "RETIRED"


# ---- reporting -------------------------------------------------------------


def test_the_desk_report_warns_when_the_agents_converge(jojo):
    rng = np.random.default_rng(6)
    shared = list(rng.normal(0.0005, 0.01, 300))
    report = jojo.desk_report({name: walk(shared, sortino=2.0)
                               for name in ("A", "B", "C")})
    assert "converged" in report
    assert "one bet with five names" in report or "one bet" in report


def test_the_desk_report_is_quiet_when_they_are_diverse(jojo):
    rng = np.random.default_rng(7)
    report = jojo.desk_report({
        name: walk(list(rng.normal(0.0005, 0.01, 300)), sortino=2.0)
        for name in ("A", "B", "C")})
    assert "WARNING" not in report
    assert "desk mean pairwise correlation" in report


def test_promoting_an_unregistered_policy_is_refused_clearly(store, jojo):
    """The foreign key would catch it, with an error naming neither party."""
    with pytest.raises(ValueError, match="not a registered policy"):
        store.promote(agent="KIRA", policy_id="never_trained", reason="oops")


def test_a_fully_converged_desk_deploys_less_capital(jojo):
    """The haircut must reduce deployment, not merely reshuffle it.

    If it only changed relative weights, five agents that all found the same
    edge would receive exactly as much capital as five independent ones — the
    concentration would be invisible in the only number that matters.
    """
    rng = np.random.default_rng(11)
    shared = list(rng.normal(0.0005, 0.01, 300))
    twins = {name: walk(shared, sortino=2.0) for name in ("A", "B", "C")}
    diverse = {name: walk(list(rng.normal(0.0005, 0.01, 300)), sortino=2.0)
               for name in ("A", "B", "C")}
    deployed_twins = sum(r.allocation for r in jojo.allocate(twins))
    deployed_diverse = sum(r.allocation for r in jojo.allocate(diverse))
    assert deployed_twins < deployed_diverse * 0.75, (deployed_twins, deployed_diverse)
