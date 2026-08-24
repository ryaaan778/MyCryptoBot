"""Provenance and source segregation.

The rule these tests defend: a number produced from a simulator must never be
able to appear in the same figure as a number produced from a real exchange —
not by accident, not by omission, not by a helpful default.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace as dc_replace

import numpy as np
import pytest

from research.collect import generate_synthetic
from research.datasets import (
    DataQualityError, DataSource, Dataset, DatasetManifest, SYNTHETIC_ID,
    assert_single_source, known_exchanges, venues_of,
)
from research.experiments import (
    EvaluationStage, ExperimentStatus, ExperimentStore, MemoryKind,
)
from research.metrics import TradeRecord, compute_metrics


@pytest.fixture(scope="module")
def bars():
    return generate_synthetic(bars=800, seed=1)


@pytest.fixture
def store(tmp_path):
    with ExperimentStore(tmp_path / "research.db") as opened:
        yield opened


def manifest_for(source: str, symbol: str = "BTC/USDT") -> DatasetManifest:
    return DatasetManifest(
        dataset_id=f"{source}_{symbol}", source=str(DataSource(source)), symbol=symbol,
        timeframe="5m", start_ts=0, end_ts=1, rows=10, sha256="x",
        collector_version="1", created_at=0,
    )


def sample_metrics():
    equity = np.array([10_000.0, 10_500.0])
    return compute_metrics(
        equity, np.array([0, 300_000], dtype=np.int64),
        [TradeRecord(0, 1, "LONG", 100.0, 105.0, 1.0, 500.0, 0.4)],
        bars_per_year=288 * 365,
    )


# ---- the source type ------------------------------------------------------


@pytest.mark.parametrize("name", ["binance", "Bybit", "OKX", "kraken", "coinbase",
                                  "kucoin", "bitget", "gate", "mexc", "htx"])
def test_real_venues_are_accepted_and_normalised(name):
    venue = DataSource(name)
    assert venue.is_real_market and str(venue) == name.lower()
    assert venue.exchange_id == name.lower()


@pytest.mark.parametrize("name", ["binanace", "byb1t", "NASDAQ", "my_exchange", "", "  "])
def test_invented_venues_are_rejected(name):
    with pytest.raises(ValueError):
        DataSource(name)


def test_ccxt_is_the_authority_on_what_a_venue_is():
    assert len(known_exchanges()) > 50
    assert "binance" in known_exchanges()


def test_synthetic_has_no_exchange():
    assert DataSource.SYNTHETIC.is_synthetic
    assert not DataSource.SYNTHETIC.is_real_market
    assert str(DataSource(SYNTHETIC_ID)) == SYNTHETIC_ID
    with pytest.raises(ValueError):
        DataSource.SYNTHETIC.exchange_id


# ---- aggregation rules ----------------------------------------------------


def test_a_single_source_aggregates():
    assert assert_single_source([manifest_for("binance")] * 3) == "binance"
    assert assert_single_source([manifest_for(SYNTHETIC_ID)]).is_synthetic


@pytest.mark.parametrize("allow", [False, True])
def test_synthetic_never_aggregates_with_real(allow):
    with pytest.raises(ValueError, match="(?i)synthetic"):
        assert_single_source(
            [manifest_for(SYNTHETIC_ID), manifest_for("binance")], allow_cross_venue=allow
        )


def test_venues_do_not_blend_by_default():
    with pytest.raises(ValueError, match="(?i)venue"):
        assert_single_source([manifest_for("binance"), manifest_for("bybit")])


def test_cross_venue_comparison_is_available_when_asked_for():
    manifests = [manifest_for("binance"), manifest_for("bybit"), manifest_for("okx")]
    assert assert_single_source(manifests, allow_cross_venue=True).is_real_market
    assert venues_of(manifests) == ["binance", "bybit", "okx"]


def test_aggregating_nothing_is_an_error():
    with pytest.raises(ValueError):
        assert_single_source([])


# ---- dataset integrity ----------------------------------------------------


def test_checksum_catches_a_single_altered_bar(bars, tmp_path):
    bars.save(tmp_path)
    path = tmp_path / "datasets" / bars.manifest.dataset_id / "candles.npz"
    with np.load(path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["close"][400] *= 1.000001
    np.savez_compressed(path, **arrays)
    with pytest.raises(DataQualityError, match="checksum"):
        Dataset.load(tmp_path, bars.manifest.dataset_id)


def test_the_source_label_is_metadata_beside_the_bars_not_inside_them(bars):
    relabelled = Dataset.from_arrays(
        ts=bars.ts, open=bars.open, high=bars.high, low=bars.low,
        close=bars.close, volume=bars.volume,
        source=DataSource("bybit"), symbol="BTC/USDT", timeframe="5m",
    )
    assert relabelled.manifest.sha256 == bars.manifest.sha256
    assert relabelled.manifest.dataset_id.startswith("bybit_")
    assert bars.manifest.dataset_id.startswith("synthetic_")
    assert relabelled.manifest.is_real_market and bars.manifest.is_synthetic


@pytest.mark.parametrize("mutate", [
    lambda d: d.ts.__setitem__(100, d.ts[99]),          # non-monotonic
    lambda d: d.close.__setitem__(100, np.nan),         # non-finite
    lambda d: d.close.__setitem__(100, -1.0),           # non-positive
    lambda d: d.high.__setitem__(100, d.low[100] - 1),  # high below the body
])
def test_broken_invariants_are_refused_at_construction(bars, mutate):
    copy = generate_synthetic(bars=800, seed=1)
    mutate(copy)
    with pytest.raises(DataQualityError):
        Dataset.from_arrays(
            ts=copy.ts, open=copy.open, high=copy.high, low=copy.low,
            close=copy.close, volume=copy.volume,
            source=DataSource.SYNTHETIC, symbol="BTC/USDT", timeframe="5m",
        )


# ---- the store ------------------------------------------------------------


def test_foreign_keys_are_enabled(store):
    assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_a_metric_cannot_exist_without_a_registered_dataset(store):
    policy = store.register_policy(agent="A", kind="baseline")
    with pytest.raises(ValueError, match="provenance"):
        store.record_metrics(policy_id=policy, dataset_id="never_registered",
                             metrics=sample_metrics())


def test_a_metric_cannot_exist_without_a_registered_policy(store, bars):
    dataset_id = store.register_dataset(bars.manifest)
    with pytest.raises(ValueError):
        store.record_metrics(policy_id="ghost", dataset_id=dataset_id,
                             metrics=sample_metrics())


def test_registering_identical_bytes_twice_is_a_noop(store, bars):
    first = store.register_dataset(bars.manifest)
    assert store.register_dataset(bars.manifest) == first
    assert len(store.datasets()) == 1


def test_the_same_id_with_different_bytes_is_refused(store, bars):
    store.register_dataset(bars.manifest)
    with pytest.raises(ValueError, match="different checksum"):
        store.register_dataset(dc_replace(bars.manifest, sha256="0" * 64))


def _seed_two_sources(store, bars):
    synthetic_id = store.register_dataset(bars.manifest)
    ids = {}
    for venue in ("binance", "bybit"):
        dataset = Dataset.from_arrays(
            ts=bars.ts, open=bars.open, high=bars.high, low=bars.low,
            close=bars.close, volume=bars.volume,
            source=DataSource(venue), symbol="BTC/USDT", timeframe="5m")
        ids[venue] = store.register_dataset(dataset.manifest)

    policies = {}
    for label, dataset_id in (("syn", synthetic_id), ("bin", ids["binance"]),
                              ("byb", ids["bybit"])):
        policies[label] = store.register_policy(agent=label.upper(), kind="baseline")
        store.record_metrics(policy_id=policies[label], dataset_id=dataset_id,
                             metrics=sample_metrics(), seed=1)
    return policies


def test_compare_refuses_synthetic_against_real(store, bars):
    policies = _seed_two_sources(store, bars)
    assert store.compare([policies["syn"]], "sortino")
    with pytest.raises(ValueError, match="(?i)synthetic"):
        store.compare([policies["syn"], policies["bin"]], "total_return_pct")
    with pytest.raises(ValueError, match="(?i)synthetic"):
        store.compare([policies["syn"], policies["bin"]], "total_return_pct",
                      allow_cross_venue=True)


def test_compare_refuses_two_venues_unless_asked(store, bars):
    policies = _seed_two_sources(store, bars)
    with pytest.raises(ValueError, match="(?i)venue"):
        store.compare([policies["bin"], policies["byb"]], "total_return_pct")
    assert len(store.compare([policies["bin"], policies["byb"]], "total_return_pct",
                             allow_cross_venue=True)) == 2


# ---- experiment lifecycle -------------------------------------------------


def test_promotion_cannot_skip_stages(store):
    experiment = store.create_experiment(agent="JOLYAN", hypothesis="h", spec={})
    with pytest.raises(ValueError, match="not a legal transition"):
        store.set_experiment_status(experiment, ExperimentStatus.PROMOTED)
    for status in (ExperimentStatus.TRAINING, ExperimentStatus.VALIDATING,
                   ExperimentStatus.CANDIDATE, ExperimentStatus.PAPER_TRADING,
                   ExperimentStatus.PROMOTED):
        store.set_experiment_status(experiment, status)
    assert store.experiment(experiment)["status"] == "PROMOTED"


def test_terminal_states_are_terminal(store):
    experiment = store.create_experiment(agent="KIRA", hypothesis="h", spec={})
    store.set_experiment_status(experiment, ExperimentStatus.REJECTED, "no edge")
    with pytest.raises(ValueError):
        store.set_experiment_status(experiment, ExperimentStatus.TRAINING)


def test_rejection_requires_a_reason(store):
    experiment = store.create_experiment(agent="KIRA", hypothesis="h", spec={})
    with pytest.raises(ValueError, match="reason"):
        store.set_experiment_status(experiment, ExperimentStatus.REJECTED)


def test_policy_versions_are_unique_per_agent(store):
    assert store.register_policy(agent="JOTARO", kind="learned") == "JOTARO_v1"
    assert store.register_policy(agent="JOTARO", kind="learned") == "JOTARO_v2"
    with pytest.raises(sqlite3.IntegrityError):
        store.register_policy(agent="JOTARO", kind="learned", version=1)


def test_leaderboard_ranks_by_median_not_by_best_window(store, bars):
    dataset_id = store.register_dataset(bars.manifest)
    lucky = store.register_policy(agent="LUCKY", kind="baseline")
    steady = store.register_policy(agent="STEADY", kind="baseline")
    for value in (-8.0, -6.0, -5.0, -4.0, 90.0):
        store.record_metrics(policy_id=lucky, dataset_id=dataset_id, seed=int(value),
                             metrics=compute_metrics(
                                 np.array([10_000.0, 10_000.0 * (1 + value / 100)]),
                                 np.array([0, 300_000], dtype=np.int64), [],
                                 bars_per_year=288 * 365))
    for value in (2.0, 2.5, 3.0, 3.5, 4.0):
        store.record_metrics(policy_id=steady, dataset_id=dataset_id, seed=int(value),
                             metrics=compute_metrics(
                                 np.array([10_000.0, 10_000.0 * (1 + value / 100)]),
                                 np.array([0, 300_000], dtype=np.int64), [],
                                 bars_per_year=288 * 365))
    order = [row["policy_id"] for row in store.leaderboard("total_return_pct")]
    assert order.index(steady) < order.index(lucky)


def test_champion_history_records_what_it_replaced(store):
    v1 = store.register_policy(agent="JOSEPH", kind="learned")
    v2 = store.register_policy(agent="JOSEPH", kind="learned")
    assert store.champion("JOSEPH") is None
    store.promote(agent="JOSEPH", policy_id=v1, reason="first")
    store.promote(agent="JOSEPH", policy_id=v2, reason="better sortino")
    assert store.champion("JOSEPH")["previous_policy_id"] == v1
    store.retire(agent="JOSEPH", policy_id=v2, reason="edge decayed")
    assert store.champion("JOSEPH") is None
    assert len(store.champion_history("JOSEPH")) == 3


def test_memory_round_trips_structured_content(store):
    store.remember(agent="KIRA", kind=MemoryKind.DEAD_END,
                   content={"features": ["rsi_14"], "why": "no edge without volume"})
    assert store.recall("KIRA", MemoryKind.DEAD_END)[0]["content"]["why"].startswith("no edge")


def test_evaluations_keep_their_gate_breakdown(store, bars):
    dataset_id = store.register_dataset(bars.manifest)
    policy = store.register_policy(agent="JONATHAN", kind="learned")
    store.record_evaluation(policy_id=policy, dataset_id=dataset_id,
                            stage=EvaluationStage.WALK_FORWARD, passed=False,
                            gates={"beats_baseline": False, "min_trades": True},
                            reason="lost to buy_and_hold on 3 of 5 windows")
    row = store.evaluations(policy)[0]
    assert row["passed"] == 0 and "buy_and_hold" in row["reason"]


def test_research_store_does_not_create_the_trading_tables(store):
    names = {r[0] for r in store.db.execute(
        "select name from sqlite_master where type='table' and name not like 'sqlite_%'")}
    assert not (names & {"orders", "trades", "positions", "events",
                         "equity_curve", "audit_log"})
    assert names == {"datasets", "experiments", "policies", "policy_metrics",
                     "evaluations", "agent_memory", "champion_history", "allocations"}


def test_five_seeds_of_one_version_can_coexist(store):
    """Seed is part of a policy's identity, not an attribute of it.

    Inference is argmax and therefore deterministic, so seed diversity can only
    come from separate training runs — five of which legitimately share one
    (agent, version).
    """
    for seed in range(5):
        store.register_policy(agent="JOLYAN", kind="learned", algo="PPO",
                              version=1, seed=seed, policy_id=f"JOLYAN_v1_s{seed}")
    registered = store.policies("JOLYAN")
    assert len(registered) == 5
    assert {row["seed"] for row in registered} == {0, 1, 2, 3, 4}
    # the same seed twice is still a duplicate
    with pytest.raises(sqlite3.IntegrityError):
        store.register_policy(agent="JOLYAN", kind="learned", version=1, seed=0,
                              policy_id="JOLYAN_v1_s0_again")


def test_an_older_database_is_migrated_rather_than_rejected(tmp_path):
    """A schema change must not cost someone their research history."""
    import sqlite3 as sql

    path = tmp_path / "legacy.db"
    legacy = sql.connect(path)
    legacy.executescript("""
        CREATE TABLE policies (
            policy_id TEXT PRIMARY KEY, agent TEXT NOT NULL, version INTEGER NOT NULL,
            kind TEXT NOT NULL, algo TEXT, experiment_id TEXT, hyperparams_json TEXT,
            feature_set TEXT, feature_set_version TEXT, reward_version TEXT,
            seed INTEGER, model_path TEXT, weights_path TEXT, status TEXT NOT NULL,
            created_at INTEGER NOT NULL, UNIQUE (agent, version)
        );
        INSERT INTO policies VALUES
            ('OLD_v1','OLD',1,'learned',NULL,NULL,NULL,NULL,NULL,NULL,0,NULL,NULL,'CANDIDATE',0);
    """)
    legacy.commit()
    legacy.close()

    with ExperimentStore(path) as migrated:
        assert [row["policy_id"] for row in migrated.policies("OLD")] == ["OLD_v1"]
        migrated.register_policy(agent="OLD", kind="learned", version=1, seed=1,
                                 policy_id="OLD_v1_s1")
        assert len(migrated.policies("OLD")) == 2
