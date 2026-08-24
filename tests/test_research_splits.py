"""Split integrity: purge, embargo, coverage, and the holdout seal."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.collect import generate_synthetic
from research.splits import (
    HOLDOUT_CONFIRMATION, HoldoutSealed, Range, SplitConfig, SplitPlan,
    default_config, make_walk_forward, scale_config,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=60_000, seed=1234)


@pytest.fixture(scope="module")
def config(dataset):
    return scale_config(default_config(dataset, warmup_bars=600), 0.08)


@pytest.fixture
def plan(dataset, config):
    return make_walk_forward(dataset, config)


def test_plan_passes_its_own_invariants(plan):
    plan.check()
    assert len(plan) >= 5


def test_training_stops_at_least_one_purge_before_validation(plan, config):
    for window in plan:
        assert window.validate.start - window.train.stop >= config.purge_bars


def test_validation_tiles_the_evaluated_span_exactly_once(plan):
    covered = np.zeros(plan.dataset_rows, dtype=int)
    for window in plan:
        covered[window.validate.start:window.validate.stop] += 1
    first = plan.windows[0].validate.start
    last = plan.windows[-1].validate.stop
    assert covered.max() == 1
    assert covered[first:last].min() == 1


def test_embargoed_bars_are_removed_from_later_training_sets(plan):
    """The embargo only bites from window 2 onward, and it must actually bite."""
    zones = plan.embargo_zones()
    dropped = []
    for window in plan:
        indices = plan.train_indices(window)
        for zone in zones:
            assert not np.any((indices >= zone.start) & (indices < zone.stop))
        dropped.append(len(window.train) - indices.size)
    assert dropped[0] == 0 and dropped[1] == 0
    assert sum(dropped) > 0, "the embargo never removed a bar"


def test_nothing_reaches_the_holdout(plan):
    for window in plan:
        assert not window.train.overlaps(plan._holdout)
        assert not window.validate.overlaps(plan._holdout)


def test_no_window_trains_inside_the_feature_warmup(plan, config):
    assert min(w.train.start for w in plan) >= config.warmup_bars


def test_holdout_is_sealed_until_deliberately_opened(plan):
    assert not plan.is_unlocked
    assert plan.holdout_size > 0          # knowing the size leaks nothing
    with pytest.raises(HoldoutSealed):
        plan.holdout()


@pytest.mark.parametrize("phrase", ["", "yes", "I_UNDERSTAND_THE_RISK",
                                    HOLDOUT_CONFIRMATION.lower()])
def test_wrong_confirmation_does_not_open_the_holdout(plan, phrase):
    with pytest.raises(HoldoutSealed):
        plan.unlock_holdout(actor="me", reason="curiosity", confirmation=phrase)


@pytest.mark.parametrize("actor,reason", [("", "r"), ("a", ""), ("  ", "  ")])
def test_unlocking_demands_an_actor_and_a_reason(plan, actor, reason):
    with pytest.raises(ValueError):
        plan.unlock_holdout(actor=actor, reason=reason, confirmation=HOLDOUT_CONFIRMATION)


def test_unlocks_are_audited_and_cumulative(plan, tmp_path):
    audit = tmp_path / "holdout.jsonl"
    plan.unlock_holdout(actor="ryan", reason="final evaluation",
                        confirmation=HOLDOUT_CONFIRMATION, audit_path=audit)
    assert plan.is_unlocked and plan.holdout() == plan._holdout
    plan.unlock_holdout(actor="ryan", reason="second look",
                        confirmation=HOLDOUT_CONFIRMATION, audit_path=audit)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(records) == 2
    assert records[1]["prior_unlocks"] == 1
    assert records[0]["reason"] == "final evaluation"


def test_a_reloaded_plan_is_still_sealed(plan, tmp_path):
    plan.save(tmp_path)
    assert not SplitPlan.load(tmp_path, plan.split_id).is_unlocked


def test_save_load_round_trips(plan, tmp_path):
    plan.save(tmp_path)
    reloaded = SplitPlan.load(tmp_path, plan.split_id)
    assert reloaded.to_dict() == plan.to_dict()
    reloaded.check()


def test_a_plan_refuses_bars_it_was_not_built_from(plan, dataset):
    plan.bind(dataset)
    with pytest.raises(ValueError):
        plan.bind(generate_synthetic(bars=60_000, seed=999))
    with pytest.raises(ValueError):
        plan.bind(generate_synthetic(bars=60_001, seed=1234))


def test_split_id_is_deterministic_and_config_sensitive(dataset, config):
    assert make_walk_forward(dataset, config).split_id == make_walk_forward(
        dataset, config).split_id
    changed = SplitConfig(**{**config.__dict__, "purge_bars": config.purge_bars + 1})
    assert make_walk_forward(dataset, changed).split_id != make_walk_forward(
        dataset, config).split_id


@pytest.mark.parametrize("override", [
    {"train_bars": 0}, {"validate_bars": 0}, {"step_bars": 0}, {"embargo_bars": -1},
    {"holdout_fraction": 0.6}, {"holdout_fraction": -0.1},
])
def test_malformed_configs_are_rejected(dataset, config, override):
    with pytest.raises(ValueError):
        make_walk_forward(dataset, SplitConfig(**{**config.__dict__, **override}))


def test_purge_may_not_consume_the_training_window(dataset, config):
    with pytest.raises(ValueError):
        make_walk_forward(dataset, SplitConfig(
            **{**config.__dict__, "purge_bars": config.train_bars}))


def test_step_larger_than_validation_would_skip_data(dataset, config):
    with pytest.raises(ValueError):
        make_walk_forward(dataset, SplitConfig(
            **{**config.__dict__, "step_bars": config.validate_bars * 2}))


def test_overlapping_validation_requires_opting_in(dataset, config):
    overlapping = {**config.__dict__, "step_bars": max(1, config.validate_bars // 2)}
    with pytest.raises(ValueError, match="independent"):
        make_walk_forward(dataset, SplitConfig(**overlapping))
    opted_in = make_walk_forward(
        dataset, SplitConfig(**{**overlapping, "allow_overlapping_validation": True}))
    assert len(opted_in) > len(make_walk_forward(dataset, config))
    assert "OVERLAPPING" in opted_in.summary()


def test_a_dataset_too_short_for_one_window_says_so():
    short = generate_synthetic(bars=5_000, seed=2)
    with pytest.raises(ValueError, match="usable development"):
        make_walk_forward(short, default_config(short, warmup_bars=600))


def test_expanding_training_grows_while_rolling_stays_fixed(dataset, config):
    rolling = [len(w.train) for w in make_walk_forward(dataset, config)]
    expanding = [len(w.train) for w in make_walk_forward(
        dataset, SplitConfig(**{**config.__dict__, "expanding": True}))]
    assert len(set(rolling)) == 1
    assert expanding == sorted(expanding) and expanding[-1] > expanding[0]


def test_range_is_half_open():
    r = Range(10, 20)
    assert len(r) == 10 and 10 in r and 20 not in r
    assert np.array_equal(r.indices(), np.arange(10, 20))
    assert r.overlaps(Range(19, 25)) and not r.overlaps(Range(20, 25))
    with pytest.raises(ValueError):
        Range(20, 10)
