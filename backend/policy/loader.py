"""Loading a validated policy for the trading path.

A policy artefact on disk is not enough to trade with. What the trading server
needs is the policy *plus* the contract it was validated under: which features,
in which order, scaled how, and under which reward. Loading the weights without
that contract is how a policy ends up fed a differently-ordered observation
vector and quietly making nonsense decisions that look like a strategy.

So the loader refuses a bundle whose metadata is missing or inconsistent, and it
never guesses a default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .numpy_infer import NumpyPolicy, load_weights

logger = logging.getLogger("jojo.policy.loader")

BUNDLE_FILENAME = "metadata.json"
WEIGHTS_FILENAME = "weights.npz"


@dataclass
class PolicyBundle:
    """A policy and the contract it was validated under."""

    policy_id: str
    agent: str
    version: int
    policy: NumpyPolicy
    feature_names: tuple[str, ...]
    feature_set_version: str
    account_feature_names: tuple[str, ...]
    reward_version: str
    env_version: str
    algo: str
    seed: int | None = None
    trained_on: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)

    @property
    def observation_size(self) -> int:
        return self.policy.observation_size

    def check_observation(self, names: tuple[str, ...]) -> None:
        """Refuse an observation whose columns are not the ones we trained on.

        Order matters as much as membership: a network fed the same features in
        a different order is not a validated policy, it is a random one.
        """
        expected = tuple(self.feature_names) + tuple(self.account_feature_names)
        if tuple(names) != expected:
            raise ValueError(
                f"{self.policy_id} was validated on {len(expected)} inputs "
                f"{expected[:4]}... but is being given {len(names)} {tuple(names)[:4]}..."
            )

    def act(self, observation: np.ndarray) -> int:
        return self.policy.act(observation)

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "agent": self.agent,
            "version": self.version,
            "algo": self.algo,
            "observation_size": self.observation_size,
            "feature_set_version": self.feature_set_version,
            "reward_version": self.reward_version,
            "env_version": self.env_version,
            "seed": self.seed,
        }


REQUIRED_KEYS = (
    "policy_id", "agent", "version", "feature_names", "feature_set_version",
    "account_feature_names", "reward_version", "env_version", "algo",
)


def load_policy(directory: str | Path) -> PolicyBundle:
    """Load ``weights.npz`` + ``metadata.json`` from a policy directory."""
    path = Path(directory)
    metadata_path = path / BUNDLE_FILENAME
    weights_path = path / WEIGHTS_FILENAME
    for required in (metadata_path, weights_path):
        if not required.is_file():
            raise FileNotFoundError(f"{required} is missing; {path} is not a policy bundle")

    metadata = json.loads(metadata_path.read_text())
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"{metadata_path} is missing {missing}. A policy without its validation "
            "contract cannot be loaded — there is no safe default for which features "
            "it expects."
        )

    weights = load_weights(weights_path)
    expected_size = len(metadata["feature_names"]) + len(metadata["account_feature_names"])
    if weights.observation_size != expected_size:
        raise ValueError(
            f"{metadata['policy_id']}: weights take {weights.observation_size} inputs but "
            f"the metadata lists {expected_size} features. The artefacts do not match."
        )

    bundle = PolicyBundle(
        policy_id=metadata["policy_id"],
        agent=metadata["agent"],
        version=int(metadata["version"]),
        policy=NumpyPolicy(weights, name=metadata["policy_id"]),
        feature_names=tuple(metadata["feature_names"]),
        feature_set_version=metadata["feature_set_version"],
        account_feature_names=tuple(metadata["account_feature_names"]),
        reward_version=metadata["reward_version"],
        env_version=metadata["env_version"],
        algo=metadata["algo"],
        seed=metadata.get("seed"),
        trained_on=metadata.get("trained_on", {}),
        gate=metadata.get("gate", {}),
    )
    logger.info("loaded policy %s (%d inputs, %s)",
                bundle.policy_id, bundle.observation_size, bundle.algo)
    return bundle
