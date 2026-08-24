"""Research agents: bias as a prior, memory as a brake, evidence as the authority.

The test this file exists for is
:func:`test_an_agent_can_discover_its_starting_bias_is_wrong`. Everything else
is scaffolding around that one requirement — an agent that cannot be wrong about
its own premise is a hard-coded strategy wearing a costume, which is precisely
what this phase replaces.

Training is stubbed throughout. What is under test is the search logic, not
whether PPO converges.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.agents import (
    AGENT_BIASES, AgentBias, ExperimentOutcome, PRIOR_STRENGTH, ResearchAgent,
    REGIME_FEATURES, REVERSION_FEATURES, TREND_FEATURES, build_agents,
)
from research.collect import generate_synthetic
from research.evaluate import GateResult
from research.experiments import ExperimentStatus, ExperimentStore, MemoryKind
from research.features import available_features
from research.hypothesis import (
    MAX_LEVERAGE, MAX_RISK_PER_TRADE, MIN_FEATURES, ExperimentSpec, SearchRanges, sample_spec,
)
from research.reward import RewardConfig
from research.splits import default_config, make_walk_forward, scale_config


@pytest.fixture
def store(tmp_path):
    with ExperimentStore(tmp_path / "research.db") as opened:
        yield opened


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=20_000, seed=3)


@pytest.fixture(scope="module")
def plan(dataset):
    return make_walk_forward(dataset, scale_config(
        default_config(dataset, warmup_bars=600), 0.05))


def gate_result(passed: bool) -> GateResult:
    """A realistic gate result: a failure always carries a failing condition."""
    from research.evaluate import Condition

    conditions = [
        Condition(name="beats_best_baseline", passed=passed, actual=1.0, threshold=0.5,
                  detail="sortino 1.000 vs best baseline flat 0.500" if passed
                         else "sortino -1.000 vs best baseline flat 0.500"),
    ]
    return GateResult(policy_id="p", passed=passed, conditions=conditions, summary={})


def scoring_runner(score_fn):
    """A stub runner that scores a spec however the test wants."""
    def run(*, spec, dataset, plan, experiment_id):
        score = score_fn(spec)
        return score, gate_result(score > 5.0), 50
    return run


# ---- the hypothesis space --------------------------------------------------


def test_a_spec_round_trips_and_is_content_addressed():
    spec = sample_spec(np.random.default_rng(0))
    assert ExperimentSpec.from_dict(spec.as_dict()).spec_id == spec.spec_id
    other = sample_spec(np.random.default_rng(1))
    assert other.spec_id != spec.spec_id


def test_risk_proposals_are_clamped_to_the_hard_caps():
    """An agent may propose anything; it may not propose past the risk engine."""
    spec = sample_spec(np.random.default_rng(0),
                       overrides={"risk_per_trade": 0.9, "leverage": 500.0})
    assert spec.risk_per_trade == MAX_RISK_PER_TRADE
    assert spec.leverage == MAX_LEVERAGE


def test_a_spec_with_too_few_features_is_refused():
    base = sample_spec(np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least"):
        ExperimentSpec.from_dict({**base.as_dict(),
                                  "features": list(base.features[:MIN_FEATURES - 1])})


def test_unknown_or_duplicated_features_are_refused():
    base = sample_spec(np.random.default_rng(0)).as_dict()
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSpec.from_dict({**base, "features": ["rsi_14", "nope", "atr_pct", "vol_20"]})
    with pytest.raises(ValueError, match="duplicate"):
        ExperimentSpec(**{**ExperimentSpec.from_dict(base).__dict__,
                          "features": ("rsi_14", "rsi_14", "atr_pct", "vol_20")})


def test_distance_is_zero_to_itself_and_bounded():
    rng = np.random.default_rng(2)
    specs = [sample_spec(rng) for _ in range(12)]
    for spec in specs:
        assert spec.distance(spec) == 0.0
    for a in specs:
        for b in specs:
            assert 0.0 <= a.distance(b) <= 1.0
            assert a.distance(b) == pytest.approx(b.distance(a))


def test_feature_weights_skew_sampling_without_forbidding_anything():
    rng = np.random.default_rng(7)
    weights = {name: 1.0 for name in available_features()}
    for name in TREND_FEATURES:
        weights[name] = 20.0
    ranges = SearchRanges(n_features=(4, 6))
    drawn = [sample_spec(rng, feature_weights=weights, ranges=ranges) for _ in range(200)]

    favoured_rate = np.mean([
        len(set(s.features) & set(TREND_FEATURES)) / len(s.features) for s in drawn])
    assert favoured_rate > 0.5, favoured_rate
    # but nothing is excluded — every feature still shows up somewhere
    seen = set().union(*(set(s.features) for s in drawn))
    assert len(seen) == len(available_features()), sorted(set(available_features()) - seen)


# ---- agent identities ------------------------------------------------------


def test_all_five_agents_exist_with_distinct_leanings(store):
    agents = build_agents(store)
    assert [a.name for a in agents] == ["JOLYAN", "JONATHAN", "JOSEPH", "JOTARO", "KIRA"]
    weights = {a.name: a.effective_weights() for a in agents}
    assert weights["JONATHAN"]["trend_slope"] > weights["KIRA"]["trend_slope"]
    assert weights["KIRA"]["rsi_14"] > weights["JONATHAN"]["rsi_14"]
    assert weights["JOSEPH"]["vol_ratio"] > weights["JOLYAN"]["vol_ratio"]


def test_every_favoured_feature_actually_exists():
    """A typo in a bias would silently become no bias at all."""
    catalogue = set(available_features())
    for bias in AGENT_BIASES.values():
        assert set(bias.favoured) <= catalogue, (bias.name, set(bias.favoured) - catalogue)
        assert set(bias.disfavoured) <= catalogue, bias.name


def test_the_explorer_starts_with_no_feature_opinion(store):
    jolyan = build_agents(store, names=["JOLYAN"])[0]
    assert len(set(jolyan.effective_weights().values())) == 1


def test_jotaro_proposes_smaller_size_than_the_others(store):
    jotaro, jolyan = build_agents(store, names=["JOTARO", "JOLYAN"])
    risk = lambda agent: np.median([agent.propose().risk_per_trade for _ in range(30)])
    assert risk(jotaro) < risk(jolyan)


def test_an_unknown_agent_is_refused(store):
    with pytest.raises(ValueError, match="unknown agent"):
        build_agents(store, names=["DIO"])


# ---- THE requirement: a bias is a prior, not a cage ------------------------


def test_an_agent_can_discover_its_starting_bias_is_wrong(store, dataset, plan):
    """JONATHAN starts trend-biased. Feed it evidence that reversion works.

    After enough contrary outcomes its effective weights must favour the
    reversion features over the trend ones it began with — without anyone
    editing its definition. If this fails, the agents are hard-coded strategies
    and the whole phase is decoration.
    """
    jonathan = build_agents(store, names=["JONATHAN"])[0]
    before = jonathan.effective_weights()
    assert np.mean([before[f] for f in TREND_FEATURES]) > \
           np.mean([before[f] for f in REVERSION_FEATURES])

    def score(spec):
        reversion = len(set(spec.features) & set(REVERSION_FEATURES))
        trend = len(set(spec.features) & set(TREND_FEATURES))
        return 10.0 * reversion - 8.0 * trend

    jonathan.campaign(dataset, plan, runner=scoring_runner(score), experiments=40)

    after = jonathan.effective_weights()
    trend_weight = float(np.mean([after[f] for f in TREND_FEATURES]))
    reversion_weight = float(np.mean([after[f] for f in REVERSION_FEATURES]))
    assert reversion_weight > trend_weight, (
        f"JONATHAN kept its trend prior against 40 contrary results: "
        f"trend {trend_weight:.2f} vs reversion {reversion_weight:.2f}")
    assert jonathan.bias_drift() > 0.05


def test_evidence_needs_to_accumulate_before_it_moves_the_prior(store, dataset, plan):
    """Three unlucky results must not flip an agent's identity."""
    jonathan = build_agents(store, names=["JONATHAN"])[0]
    start = jonathan.effective_weights()
    jonathan.campaign(dataset, plan, experiments=2,
                      runner=scoring_runner(lambda spec: -50.0))
    small = jonathan.bias_drift()
    jonathan.campaign(dataset, plan, experiments=30,
                      runner=scoring_runner(
                          lambda spec: 20.0 * len(set(spec.features) & set(REGIME_FEATURES))))
    assert jonathan.bias_drift() > small
    assert small < 0.35, f"two experiments moved the prior by {small:.0%}"


def test_no_feature_is_ever_ruled_out_entirely(store, dataset, plan):
    """Even a comprehensively discredited feature keeps a non-zero weight."""
    kira = build_agents(store, names=["KIRA"])[0]
    kira.campaign(dataset, plan, experiments=30,
                  runner=scoring_runner(
                      lambda spec: -30.0 * len(set(spec.features) & set(REVERSION_FEATURES))))
    weights = kira.effective_weights()
    assert all(value > 0 for value in weights.values())
    assert min(weights.values()) > 1e-6


def test_with_no_history_the_effective_weights_are_exactly_the_prior(store):
    for agent in build_agents(store):
        assert agent.effective_weights() == agent.bias.initial_weights()
        assert agent.bias_drift() == 0.0


# ---- memory ----------------------------------------------------------------


def test_the_generator_avoids_re_proposing_known_failures(store, dataset, plan):
    joseph = build_agents(store, names=["JOSEPH"])[0]
    joseph.campaign(dataset, plan, experiments=12,
                    runner=scoring_runner(lambda spec: -1.0))
    tried = [spec for spec, _, _ in joseph.memory.past_specs()]
    assert len(tried) == 12
    assert len({spec.spec_id for spec in tried}) == 12, "the same spec was run twice"

    fresh = joseph.propose()
    nearest = min(fresh.distance(spec) for spec in tried)
    assert nearest >= joseph.reject_distance() or nearest > 0.05


def test_the_exclusion_zone_widens_as_dead_ends_accumulate(store, dataset, plan):
    joseph = build_agents(store, names=["JOSEPH"])[0]
    start = joseph.reject_distance()
    joseph.campaign(dataset, plan, experiments=15,
                    runner=scoring_runner(lambda spec: -1.0))
    assert joseph.reject_distance() > start


def test_outcomes_and_dead_ends_are_both_recorded(store, dataset, plan):
    kira = build_agents(store, names=["KIRA"])[0]
    kira.campaign(dataset, plan, experiments=3, runner=scoring_runner(lambda spec: -2.0))
    assert len(store.recall("KIRA", MemoryKind.OUTCOME)) == 3
    dead_ends = store.recall("KIRA", MemoryKind.DEAD_END)
    assert len(dead_ends) == 3
    assert "why" in dead_ends[0]["content"]


def test_each_agent_keeps_its_own_memory(store, dataset, plan):
    jonathan, kira = build_agents(store, names=["JONATHAN", "KIRA"])
    jonathan.campaign(dataset, plan, experiments=4, runner=scoring_runner(lambda s: 1.0))
    kira.campaign(dataset, plan, experiments=2, runner=scoring_runner(lambda s: 1.0))
    assert len(jonathan.memory.past_specs()) == 4
    assert len(kira.memory.past_specs()) == 2
    assert len(store.experiments("JONATHAN")) == 4
    assert len(store.experiments("KIRA")) == 2


# ---- the experiment lifecycle ---------------------------------------------


def test_a_passing_experiment_reaches_candidate(store, dataset, plan):
    agent = build_agents(store, names=["JOLYAN"])[0]
    outcome = agent.campaign(dataset, plan, experiments=1,
                             runner=scoring_runner(lambda spec: 99.0))[0]
    assert outcome.promoted
    assert store.experiment(outcome.experiment_id)["status"] == ExperimentStatus.CANDIDATE.value


def test_a_failing_experiment_is_rejected_with_a_reason(store, dataset, plan):
    agent = build_agents(store, names=["JOLYAN"])[0]

    def runner(*, spec, dataset, plan, experiment_id):
        return -1.0, gate_result(False), 3

    outcome = agent.run_experiment(agent.propose(), dataset, plan, runner=runner)
    record = store.experiment(outcome.experiment_id)
    assert record["status"] == ExperimentStatus.REJECTED.value
    assert record["decision_reason"]


def test_a_crashing_run_is_recorded_rather_than_lost(store, dataset, plan):
    """A failed experiment is data. Losing it means proposing it again."""
    agent = build_agents(store, names=["JOLYAN"])[0]

    def exploding(*, spec, dataset, plan, experiment_id):
        raise RuntimeError("CUDA out of memory")

    outcome = agent.run_experiment(agent.propose(), dataset, plan, runner=exploding)
    assert not outcome.succeeded and "CUDA" in outcome.error
    record = store.experiment(outcome.experiment_id)
    assert record["status"] == ExperimentStatus.REJECTED.value
    assert "CUDA" in record["decision_reason"]
    assert len(agent.memory.past_specs()) == 1


def test_the_hypothesis_is_human_readable(store, dataset, plan):
    agent = build_agents(store, names=["JOTARO"])[0]
    outcome = agent.campaign(dataset, plan, experiments=1,
                             runner=scoring_runner(lambda spec: 1.0))[0]
    hypothesis = store.experiment(outcome.experiment_id)["hypothesis"]
    assert "risk-aware" in hypothesis and "risk" in hypothesis


def test_a_campaign_reacts_to_its_own_previous_result(store, dataset, plan):
    """Sequential, not batched — otherwise it is random search with extra steps."""
    agent = build_agents(store, names=["JOLYAN"])[0]
    seen: list[int] = []

    def runner(*, spec, dataset, plan, experiment_id):
        seen.append(len(agent.memory.past_specs()))
        return 1.0, gate_result(False), 10

    agent.campaign(dataset, plan, runner=runner, experiments=4)
    assert seen == [0, 1, 2, 3]


def test_agents_are_reproducible_from_their_seed(store, tmp_path, dataset, plan):
    first = ResearchAgent(AGENT_BIASES["KIRA"], store, seed=11)
    with ExperimentStore(tmp_path / "other.db") as other_store:
        second = ResearchAgent(AGENT_BIASES["KIRA"], other_store, seed=11)
        third = ResearchAgent(AGENT_BIASES["KIRA"], other_store, seed=12)
        a = [first.propose().spec_id for _ in range(5)]
        b = [second.propose().spec_id for _ in range(5)]
        c = [third.propose().spec_id for _ in range(5)]
    assert a == b
    assert a != c


def test_the_summary_reports_drift(store, dataset, plan):
    agent = build_agents(store, names=["JONATHAN"])[0]
    agent.campaign(dataset, plan, experiments=6, runner=scoring_runner(lambda s: 2.0))
    text = agent.summary()
    assert "JONATHAN" in text and "6 experiments" in text and "drift" in text


def test_a_terse_gate_still_produces_a_recorded_reason(store, dataset, plan):
    """The store refuses a reason-less rejection; the agent must not crash on it."""
    agent = build_agents(store, names=["JOLYAN"])[0]

    def terse(*, spec, dataset, plan, experiment_id):
        return -1.0, GateResult(policy_id="p", passed=False, conditions=[], summary={}), 7

    outcome = agent.run_experiment(agent.propose(), dataset, plan, runner=terse)
    record = store.experiment(outcome.experiment_id)
    assert record["status"] == ExperimentStatus.REJECTED.value
    assert record["decision_reason"], "a rejection was recorded with no reason at all"
    assert "7 trade" in record["decision_reason"]


# ---- the baseline-gaming safeguard ----------------------------------------


def test_baselines_are_measured_under_a_fixed_envelope_not_the_agents(dataset):
    """Otherwise an agent clears the gate by crippling the comparison.

    "beats_best_baseline" is only meaningful if the baseline is allowed a
    sensible envelope. If the baselines inherited the spec's proposed risk and
    stop parameters, an agent could satisfy the condition by proposing settings
    that wreck buy-and-hold rather than by learning anything.
    """
    from research.campaign import REFERENCE_CONFIG, backtest_config_for, baseline_config_for

    hostile = sample_spec(np.random.default_rng(0),
                          overrides={"risk_per_trade": 0.1, "leverage": 10.0,
                                     "stop_loss_pct": 0.3})
    challenger_config = backtest_config_for(hostile)
    assert challenger_config.risk_per_trade == pytest.approx(0.1)
    assert challenger_config.leverage == pytest.approx(10.0)

    for name in ("momentum", "mean_reversion", "random"):
        assert baseline_config_for(name).risk_per_trade == REFERENCE_CONFIG.risk_per_trade
        assert baseline_config_for(name).leverage == REFERENCE_CONFIG.leverage
        assert baseline_config_for(name).stop_loss_pct == REFERENCE_CONFIG.stop_loss_pct


def test_hold_forever_baselines_get_no_protective_stop():
    from research.campaign import baseline_config_for

    for name in ("flat", "buy_and_hold", "always_short"):
        config = baseline_config_for(name)
        assert config.stop_loss_pct is None and config.take_profit_pct is None
    assert baseline_config_for("scalping").stop_loss_pct is not None


def test_the_baseline_cache_computes_once_per_dataset_and_plan(dataset, plan, monkeypatch):
    from research import campaign as campaign_module

    calls = {"n": 0}
    real = campaign_module.run_walk_forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "run_walk_forward", counting)
    cache = campaign_module.BaselineCache()
    first = cache.get(dataset, plan, (0,), names=("flat", "buy_and_hold"))
    after_first = calls["n"]
    second = cache.get(dataset, plan, (0,), names=("flat", "buy_and_hold"))
    assert calls["n"] == after_first, "the cache recomputed identical baselines"
    assert first is second
    cache.get(dataset, plan, (0, 1), names=("flat", "buy_and_hold"))
    assert calls["n"] > after_first, "a different seed set must not reuse the cache"


def test_a_spec_maps_onto_the_backtest_envelope_it_asked_for():
    from research.campaign import backtest_config_for

    spec = sample_spec(np.random.default_rng(4),
                       overrides={"risk_per_trade": 0.03, "leverage": 2.0,
                                  "stop_loss_pct": 1.5, "take_profit_pct": 2.0})
    config = backtest_config_for(spec)
    assert config.risk_per_trade == pytest.approx(0.03)
    assert config.stop_loss_pct == pytest.approx(1.5)
    assert config.take_profit_pct == pytest.approx(2.0)
    config.validate()


def test_a_spec_without_a_take_profit_still_sizes_positions():
    from research.campaign import backtest_config_for

    spec = sample_spec(np.random.default_rng(5),
                       overrides={"take_profit_pct": None, "stop_loss_pct": 2.0})
    config = backtest_config_for(spec)
    assert config.take_profit_pct is None
    assert config.sizing_stop_pct > 0
    config.validate()
