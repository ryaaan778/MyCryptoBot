"""Running an exported policy through the same evaluation path as everything else.

A trained policy is only meaningful if it can be measured against the baselines
on identical terms. :class:`LearnedPolicy` implements the same ``Policy``
interface the hand-written strategies use, so ``run_backtest`` and
``run_walk_forward`` treat a PPO network and ``mean_reversion`` the same way —
same fills, same risk engine, same metrics, same promotion gate.

It deliberately loads the **numpy** artefact rather than the SB3 checkpoint.
Evaluating through the same code path the trading server would use means the
inference implementation is exercised on every evaluation, not just in a unit
test — so a divergence between torch and numpy shows up as a bad backtest rather
than as a surprise in production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from backend.policy.numpy_infer import MLPPolicyWeights, load_weights

from .datasets import Dataset
from .env import OBSERVATION_CLIP
from .features import FeatureMatrix, Scaler, build_features
from .policy import AccountState, BasePolicy, Decision, PolicyAction, PolicyContext

logger = logging.getLogger("research.learned")


class LearnedPolicy(BasePolicy):
    """An exported network, wearing the same interface as a hand-written strategy."""

    kind = "learned"

    def __init__(
        self,
        weights: MLPPolicyWeights,
        scaler: Scaler,
        features: FeatureMatrix,
        *,
        name: str = "learned",
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        self.weights = weights
        self.scaler = scaler
        self.features = features
        self.name = name
        self.deterministic = deterministic
        self.min_bars = max(1, features.warmup)
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        if features.names != scaler.names:
            raise ValueError("scaler and feature matrix disagree on column order")

        # Scale once: the matrix is causal, so precomputing every row is safe and
        # turns per-bar inference into an array lookup plus four matrix products.
        self._scaled = scaler.transform(features).values
        np.clip(self._scaled, -OBSERVATION_CLIP, OBSERVATION_CLIP, out=self._scaled)
        np.nan_to_num(self._scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        expected = features.n_features + len(AccountState.VECTOR_NAMES)
        if weights.observation_size not in (expected, features.n_features):
            raise ValueError(
                f"{name} takes {weights.observation_size} inputs, but this feature set "
                f"gives {features.n_features} market features "
                f"(+{len(AccountState.VECTOR_NAMES)} account = {expected})"
            )
        self._uses_account_state = weights.observation_size == expected

    @classmethod
    def from_directory(
        cls, directory: str | Path, dataset: Dataset, *, deterministic: bool = True,
        seed: int | None = None,
    ) -> "LearnedPolicy":
        """Load weights, scaler and feature list from a training run's output.

        The feature names come from the metadata rather than from defaults, so a
        policy trained on a subset is rebuilt on that subset. Feeding a network
        the full feature set when it was trained on eight columns would not
        error anywhere — it would just produce confident nonsense.
        """
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text())
        weights = load_weights(path / "weights.npz")
        scaler = Scaler.from_dict(metadata["scaler"])
        features = build_features(dataset, metadata["feature_names"])
        if features.names != tuple(metadata["feature_names"]):
            raise ValueError("feature builder returned a different column order than trained on")
        return cls(weights, scaler, features, name=metadata["policy_id"],
                   deterministic=deterministic, seed=seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self._seed if seed is None else seed)

    def observation(self, index: int, account: AccountState) -> np.ndarray:
        market = self._scaled[index]
        if not self._uses_account_state:
            return market
        state = np.clip(account.as_vector(), -OBSERVATION_CLIP, OBSERVATION_CLIP)
        return np.concatenate([market, state])

    def decide(self, ctx: PolicyContext) -> Decision:
        if ctx.index < self.features.warmup:
            return self.hold(f"warming up ({ctx.index}/{self.features.warmup} bars)")

        observation = self.observation(ctx.index, ctx.account)
        probabilities = self.weights.probabilities(observation)
        if self.deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self._rng.choice(len(probabilities), p=probabilities))

        return Decision(
            action=PolicyAction(action),
            confidence=float(probabilities[action]),
            reason=f"{self.name} p={probabilities[action]:.2f}",
            diagnostics={
                f"p_{PolicyAction(i).name.lower()}": float(p)
                for i, p in enumerate(probabilities)
            },
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "observation_size": self.weights.observation_size,
            "uses_account_state": self._uses_account_state,
            "deterministic": self.deterministic,
            "features": list(self.features.names),
        }
