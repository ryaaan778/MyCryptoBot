"""Training and export.

The load-bearing test here is :func:`test_numpy_inference_reproduces_sb3`. The
trading server runs the numpy forward pass; validation runs whichever path the
evaluator uses. If those two ever disagree, the policy that trades is not the
policy that was measured, and no amount of careful backtesting means anything.

Training is slow, so these use tiny budgets. They test the *plumbing* — that
weights come out, that they load, that the two implementations agree — not that
PPO learns anything, which no unit test could honestly assert.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from research.backtest import BacktestConfig, run_backtest
from research.collect import generate_synthetic
from research.learned import LearnedPolicy
from research.policy import AccountState, PolicyAction
from research.train import TrainConfig, export_weights, train_policy

TINY = TrainConfig(total_timesteps=1_500, n_steps=128, batch_size=32, n_envs=2,
                   episode_bars=600, seed=0)


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=6_000, seed=5)


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    root = tmp_path_factory.mktemp("policies")
    return train_policy(dataset, agent="TEST", train_start=0, train_stop=4_000,
                        root=root, train_config=TINY, version=1)


def test_training_writes_both_artefacts(trained):
    assert trained.model_path.is_file(), "SB3 checkpoint missing"
    assert trained.weights_path.is_file(), "numpy weights missing"
    assert (trained.directory / "metadata.json").is_file()


def test_numpy_inference_reproduces_sb3(trained):
    """The guarantee that lets the trading server skip torch entirely."""
    from stable_baselines3 import PPO

    model = PPO.load(trained.model_path, device="cpu")
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(300, trained.weights.observation_size)).astype(np.float32)

    torch_actions, _ = model.predict(observations, deterministic=True)
    numpy_actions = np.array([trained.weights.act(o) for o in observations])
    assert np.array_equal(torch_actions, numpy_actions)

    import torch

    with torch.no_grad():
        features = model.policy.extract_features(torch.as_tensor(observations))
        latent = model.policy.mlp_extractor.forward_actor(features)
        torch_logits = model.policy.action_net(latent).cpu().numpy()
    numpy_logits = trained.weights.logits(observations.astype(np.float64))
    assert np.abs(torch_logits - numpy_logits).max() < 1e-6


def test_the_metadata_records_the_full_validation_contract(trained):
    metadata = json.loads((trained.directory / "metadata.json").read_text())
    for key in ("feature_names", "feature_set_version", "account_feature_names",
                "reward_version", "env_version", "algo", "seed", "scaler",
                "train_config", "reward_config", "trained_on"):
        assert key in metadata, key
    assert metadata["account_feature_names"] == list(AccountState.VECTOR_NAMES)
    assert metadata["actions"] == [a.name for a in PolicyAction]


def test_the_metadata_pins_the_exact_bytes_trained_on(trained, dataset):
    """An experiment must be able to say which data produced it."""
    trained_on = json.loads((trained.directory / "metadata.json").read_text())["trained_on"]
    assert trained_on["dataset_sha256"] == dataset.manifest.sha256
    assert trained_on["source"] == dataset.manifest.source
    assert trained_on["train_start"] == 0 and trained_on["train_stop"] == 4_000


def test_the_saved_scaler_is_the_training_one(trained, dataset):
    from research.features import Scaler, build_features

    metadata = json.loads((trained.directory / "metadata.json").read_text())
    saved = Scaler.from_dict(metadata["scaler"])
    matrix = build_features(dataset)
    expected = Scaler.fit(matrix, np.arange(matrix.warmup, 4_000))
    assert np.allclose(saved.mean, expected.mean)
    # and it is genuinely not the whole-dataset scaler
    whole = Scaler.fit(matrix, np.arange(matrix.warmup, len(dataset)))
    assert not np.allclose(saved.mean, whole.mean)


def test_a_learned_policy_loads_and_backtests_like_any_other(trained, dataset):
    policy = LearnedPolicy.from_directory(trained.directory, dataset)
    result = run_backtest(policy, dataset, start=4_000, stop=6_000,
                          config=BacktestConfig())
    assert result.equity.size == 2_000
    assert result.equity[-1] - 10_000.0 == pytest.approx(
        sum(trade.pnl for trade in result.trades), abs=1e-6)
    assert set(np.unique(result.actions)) <= {0, 1, 2, 3}


def test_a_learned_policy_is_deterministic(trained, dataset):
    def run():
        policy = LearnedPolicy.from_directory(trained.directory, dataset)
        return run_backtest(policy, dataset, start=4_000, stop=5_000,
                            config=BacktestConfig())

    first, second = run(), run()
    assert np.array_equal(first.equity, second.equity)
    assert first.trades == second.trades


def test_a_learned_policy_holds_through_the_feature_warmup(trained, dataset):
    from research.policy import PolicyContext

    policy = LearnedPolicy.from_directory(trained.directory, dataset)
    account = AccountState(equity=10_000.0, starting_equity=10_000.0)
    for index in range(policy.features.warmup):
        decision = policy.decide(PolicyContext(dataset=dataset, index=index, account=account))
        assert decision.action is PolicyAction.HOLD


def test_the_risk_engine_still_governs_a_learned_policy(trained, dataset):
    policy = LearnedPolicy.from_directory(trained.directory, dataset)
    result = run_backtest(policy, dataset, start=4_000, stop=6_000,
                          config=BacktestConfig(max_exposure_pct=0.001))
    assert result.trades == []
    assert result.blocked_by_risk >= 0


def test_a_feature_set_mismatch_is_refused(trained, dataset):
    from research.features import Scaler, build_features
    from backend.policy.numpy_infer import load_weights

    weights = load_weights(trained.weights_path)
    smaller = build_features(dataset, ["rsi_14", "atr_pct"])
    scaler = Scaler.fit(smaller, np.arange(smaller.warmup, 4_000))
    with pytest.raises(ValueError, match="inputs"):
        LearnedPolicy(weights, scaler, smaller)


def test_an_unexportable_architecture_is_refused_rather_than_truncated(dataset, tmp_path):
    """Silently skipping a layer would export a policy that is not the one trained."""
    from stable_baselines3 import PPO
    from torch import nn

    from research.env import make_training_env

    env = make_training_env(dataset, train_start=0, train_stop=4_000, episode_bars=600)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu", seed=0)
    model.policy.mlp_extractor.policy_net = nn.Sequential(
        nn.Linear(31, 8), nn.Tanh(), nn.LSTM(8, 8))
    with pytest.raises(ValueError, match="unexpected layer"):
        export_weights(model, activation="tanh")


@pytest.mark.parametrize("override", [
    {"total_timesteps": 0}, {"n_envs": 0}, {"batch_size": 10 ** 9},
    {"activation": "gelu"}, {"net_arch": ()},
])
def test_malformed_training_configs_are_rejected(override):
    with pytest.raises(ValueError):
        TrainConfig(**override).validate()


def test_an_episode_longer_than_the_training_range_is_refused(dataset, tmp_path):
    with pytest.raises(ValueError, match="episode_bars"):
        train_policy(dataset, agent="TOOBIG", train_start=0, train_stop=800,
                     root=tmp_path, train_config=TrainConfig(episode_bars=5_000))


def test_evaluation_seeds_do_not_create_real_diversity(trained, dataset):
    """The flaw this design avoids: inference is argmax, so it is deterministic.

    Running one policy under five evaluation seeds produces five identical runs.
    If the gate counted those as five seeds, a single lucky training run would
    satisfy a requirement meant to prove robustness.
    """
    runs = []
    for seed in range(3):
        policy = LearnedPolicy.from_directory(trained.directory, dataset, seed=seed)
        runs.append(run_backtest(policy, dataset, start=4_000, stop=5_000,
                                 config=BacktestConfig(), seed=seed))
    assert all(np.array_equal(runs[0].equity, other.equity) for other in runs[1:])
    assert all(runs[0].trades == other.trades for other in runs[1:])


def test_training_seeds_do_create_real_diversity(dataset, tmp_path):
    """Which is why seeds come from training several policies instead."""
    from research.train import train_seed_ensemble

    ensemble = train_seed_ensemble(
        dataset, agent="ENS", train_start=0, train_stop=4_000, seeds=(0, 1),
        train_config=TINY, root=tmp_path)
    assert [t.policy_id for t in ensemble] == ["ENS_v1_s0", "ENS_v1_s1"]

    curves = []
    for trained_seed in ensemble:
        policy = LearnedPolicy.from_directory(trained_seed.directory, dataset)
        curves.append(run_backtest(policy, dataset, start=4_000, stop=5_000,
                                   config=BacktestConfig()).equity)
    assert not np.array_equal(curves[0], curves[1]), (
        "two training seeds produced identical policies — the seed is not being used")
