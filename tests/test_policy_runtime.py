"""The numpy inference path — what the trading server will actually run.

The point of this module existing at all is that a process holding real
positions must not import torch. These tests pin the contract that makes that
safe: the artefacts load, the forward pass is correct, and a mismatched or
under-specified bundle is refused rather than silently mis-run.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from backend.models import SignalAction
from backend.policy import MLPPolicyWeights, NumpyPolicy, load_policy, load_weights
from backend.policy.adapter import ACTION_ORDER, PolicyRunner
from backend.policy.numpy_infer import WEIGHTS_FORMAT_VERSION


def make_weights(observation_size=31, action_size=4, activation="tanh", seed=0,
                 hidden=(64, 64)):
    rng = np.random.default_rng(seed)
    layers, previous = [], observation_size
    for width in hidden:
        layers.append((rng.normal(scale=0.3, size=(previous, width)),
                       rng.normal(scale=0.1, size=width)))
        previous = width
    return MLPPolicyWeights(
        layers=layers,
        action_weight=rng.normal(scale=0.3, size=(previous, action_size)),
        action_bias=rng.normal(scale=0.1, size=action_size),
        activation=activation, observation_size=observation_size, action_size=action_size,
    )


# ---- the isolation guarantee ----------------------------------------------


def test_the_trading_policy_path_does_not_import_torch():
    """A broken torch install must not be able to take down a process with positions.

    Run in a clean interpreter on purpose. Asserting on ``sys.modules`` inside
    the suite proves nothing: another test file imports torch, and this one then
    passes or fails on collection order rather than on the property it claims to
    check.
    """
    import subprocess

    probe = (
        "import backend.policy, backend.policy.adapter, backend.policy.loader, "
        "backend.policy.numpy_infer, backend.app, backend.orchestrator, sys;"
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'torch', 'stable_baselines3', 'gymnasium'});"
        "print(','.join(leaked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=".", check=True
    )
    leaked = completed.stdout.strip()
    assert leaked == "", f"the trading path pulled in research dependencies: {leaked}"


def test_no_backend_module_imports_research():
    """The dependency runs one way: research imports backend, never the reverse."""
    import pathlib

    for path in pathlib.Path("backend").rglob("*.py"):
        source = path.read_text()
        assert "import research" not in source, path
        assert "from research" not in source, path


# ---- the forward pass ------------------------------------------------------


def test_the_forward_pass_matches_an_explicit_computation():
    weights = make_weights(observation_size=5, action_size=4, hidden=(3,), seed=1)
    observation = np.array([0.5, -1.0, 0.25, 2.0, -0.75])
    (w0, b0) = weights.layers[0]
    expected = np.tanh(observation @ w0 + b0) @ weights.action_weight + weights.action_bias
    assert np.allclose(weights.logits(observation), expected)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_both_activations_work(activation):
    weights = make_weights(activation=activation)
    logits = weights.logits(np.zeros(31))
    assert logits.shape == (4,) and np.all(np.isfinite(logits))


def test_an_unsupported_activation_is_refused():
    with pytest.raises(ValueError):
        make_weights(activation="gelu")


def test_probabilities_are_a_distribution():
    weights = make_weights()
    probabilities = weights.probabilities(np.random.default_rng(2).normal(size=31))
    assert probabilities.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities >= 0)


def test_extreme_logits_do_not_overflow():
    weights = make_weights(observation_size=3, action_size=4, hidden=(2,))
    weights.layers[0] = (weights.layers[0][0] * 0, weights.layers[0][1] * 0)
    weights.action_weight = weights.action_weight * 0
    weights.action_bias = np.array([1000.0, -1000.0, 0.0, 500.0])
    probabilities = weights.probabilities(np.zeros(3))
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum() == pytest.approx(1.0)
    assert weights.act(np.zeros(3)) == 0


def test_the_action_is_the_argmax_and_never_sampled():
    """Production inference is deterministic, so an incident can be reconstructed."""
    weights = make_weights()
    observation = np.random.default_rng(3).normal(size=31)
    chosen = {weights.act(observation) for _ in range(20)}
    assert len(chosen) == 1
    assert chosen.pop() == int(np.argmax(weights.logits(observation)))


def test_batched_and_single_observations_agree():
    weights = make_weights()
    rng = np.random.default_rng(4)
    batch = rng.normal(size=(6, 31))
    assert np.allclose(weights.logits(batch), [weights.logits(row) for row in batch])


def test_a_wrongly_sized_observation_is_rejected():
    weights = make_weights(observation_size=31)
    with pytest.raises(ValueError, match="trained on"):
        weights.logits(np.zeros(24))


@pytest.mark.parametrize("broken", [
    lambda w: setattr(w, "action_weight", np.zeros((5, 4))),
    lambda w: w.layers.__setitem__(1, (np.zeros((7, 64)), np.zeros(64))),
])
def test_inconsistent_shapes_are_caught_at_construction(broken):
    weights = make_weights()
    broken(weights)
    with pytest.raises(ValueError):
        MLPPolicyWeights(
            layers=weights.layers, action_weight=weights.action_weight,
            action_bias=weights.action_bias, activation=weights.activation,
            observation_size=weights.observation_size, action_size=weights.action_size,
        )


# ---- persistence -----------------------------------------------------------


def test_weights_round_trip_through_disk(tmp_path):
    weights = make_weights()
    weights.obs_clip = 10.0
    weights.metadata = {"policy_id": "JOLYAN_v1", "agent": "JOLYAN"}
    weights.save(tmp_path / "weights.npz")
    reloaded = load_weights(tmp_path / "weights.npz")
    observation = np.random.default_rng(5).normal(size=31)
    assert np.allclose(reloaded.logits(observation), weights.logits(observation))
    assert reloaded.metadata["agent"] == "JOLYAN"
    assert reloaded.obs_clip == 10.0


def test_a_future_weights_format_is_refused_rather_than_guessed(tmp_path):
    weights = make_weights()
    weights.save(tmp_path / "weights.npz")
    with np.load(tmp_path / "weights.npz") as payload:
        arrays = {k: payload[k] for k in payload.files}
    arrays["format_version"] = np.asarray(["99"])
    np.savez_compressed(tmp_path / "weights.npz", **arrays)
    with pytest.raises(ValueError, match="format"):
        load_weights(tmp_path / "weights.npz")


# ---- the bundle contract ---------------------------------------------------


def write_bundle(tmp_path, **overrides):
    weights = make_weights()
    weights.save(tmp_path / "weights.npz")
    metadata = {
        "policy_id": "JOLYAN_v1", "agent": "JOLYAN", "version": 1, "algo": "PPO",
        "feature_names": [f"f{i}" for i in range(24)],
        "feature_set_version": "v1",
        "account_feature_names": [f"a{i}" for i in range(7)],
        "reward_version": "v1", "env_version": "v1", "seed": 0,
    }
    metadata.update(overrides)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_a_complete_bundle_loads(tmp_path):
    bundle = load_policy(write_bundle(tmp_path))
    assert bundle.policy_id == "JOLYAN_v1"
    assert bundle.observation_size == 31
    assert bundle.describe()["reward_version"] == "v1"


@pytest.mark.parametrize("missing", ["feature_names", "reward_version", "env_version",
                                     "feature_set_version", "algo"])
def test_a_bundle_without_its_validation_contract_is_refused(tmp_path, missing):
    directory = write_bundle(tmp_path)
    metadata = json.loads((directory / "metadata.json").read_text())
    del metadata[missing]
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing"):
        load_policy(directory)


def test_metadata_that_disagrees_with_the_weights_is_refused(tmp_path):
    directory = write_bundle(tmp_path, feature_names=[f"f{i}" for i in range(12)])
    with pytest.raises(ValueError, match="do not match"):
        load_policy(directory)


def test_a_missing_artefact_is_reported_clearly(tmp_path):
    directory = write_bundle(tmp_path)
    (directory / "weights.npz").unlink()
    with pytest.raises(FileNotFoundError, match="not a policy bundle"):
        load_policy(directory)


def test_a_reordered_observation_is_refused(tmp_path):
    bundle = load_policy(write_bundle(tmp_path))
    correct = bundle.feature_names + bundle.account_feature_names
    bundle.check_observation(correct)
    with pytest.raises(ValueError):
        bundle.check_observation(tuple(reversed(correct)))


# ---- the runtime adapter ---------------------------------------------------


def test_the_action_codes_match_the_research_side():
    from research.policy import PolicyAction

    assert [a.name for a in PolicyAction] == [a.value for a in ACTION_ORDER]


def test_the_runner_maps_indices_to_signal_actions(tmp_path):
    runner = PolicyRunner(load_policy(write_bundle(tmp_path)))
    action = runner.decide(np.random.default_rng(6).normal(size=31))
    assert action in ACTION_ORDER
    assert isinstance(action, SignalAction)


def test_a_non_finite_observation_holds_rather_than_guessing(tmp_path):
    runner = PolicyRunner(load_policy(write_bundle(tmp_path)))
    observation = np.random.default_rng(7).normal(size=31)
    observation[3] = np.nan
    assert runner.decide(observation) is SignalAction.HOLD


def test_the_runner_rejects_a_wrongly_sized_observation(tmp_path):
    runner = PolicyRunner(load_policy(write_bundle(tmp_path)))
    with pytest.raises(ValueError):
        runner.decide(np.zeros(12))


def test_the_runner_reports_a_probability_per_action(tmp_path):
    runner = PolicyRunner(load_policy(write_bundle(tmp_path)))
    probabilities = runner.probabilities(np.zeros(31))
    assert set(probabilities) == {a.value for a in ACTION_ORDER}
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_an_action_count_mismatch_is_refused(tmp_path):
    """A three-action network cannot drive a four-action runtime."""
    directory = write_bundle(tmp_path)
    make_weights(action_size=3).save(directory / "weights.npz")   # after, so it sticks
    with pytest.raises(ValueError, match="actions"):
        PolicyRunner(load_policy(directory))
