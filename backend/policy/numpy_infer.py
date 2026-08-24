"""A hand-written forward pass for the policy networks SB3 produces.

Stable-Baselines3's ``ActorCriticPolicy`` for a ``Discrete`` action space is, at
inference time, a small deterministic pipeline:

    observation -> [shared extractor] -> policy MLP (tanh) -> action logits

There is no sampling in evaluation mode: the action is ``argmax`` over the
logits. That is a few matrix multiplies, which is why reimplementing it here is
worth doing rather than clever — it removes torch from the process that places
orders, at the cost of about eighty lines that a test pins to SB3's own output.

Only the architectures we actually export are supported, and anything else
raises at load time rather than silently producing wrong actions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("jojo.policy.numpy")

WEIGHTS_FORMAT_VERSION = "1"

SUPPORTED_ACTIVATIONS = {"tanh", "relu"}


def _activate(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(x)
    if kind == "relu":
        return np.maximum(x, 0.0)
    raise ValueError(f"unsupported activation {kind!r}; expected one of {SUPPORTED_ACTIVATIONS}")


@dataclass
class MLPPolicyWeights:
    """Dense layers plus the action head, in evaluation order."""

    layers: list[tuple[np.ndarray, np.ndarray]]     # (weight, bias) per hidden layer
    action_weight: np.ndarray
    action_bias: np.ndarray
    activation: str
    observation_size: int
    action_size: int
    #: Normalisation applied before the network, when the exporter recorded one.
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    obs_clip: float | None = None
    metadata: dict = None                            # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"unsupported activation {self.activation!r}")
        expected = self.observation_size
        for index, (weight, bias) in enumerate(self.layers):
            if weight.shape[0] != expected:
                raise ValueError(
                    f"layer {index} expects {weight.shape[0]} inputs but the previous stage "
                    f"produces {expected}"
                )
            if weight.shape[1] != bias.shape[0]:
                raise ValueError(f"layer {index} weight/bias mismatch: "
                                 f"{weight.shape} vs {bias.shape}")
            expected = weight.shape[1]
        if self.action_weight.shape != (expected, self.action_size):
            raise ValueError(
                f"action head is {self.action_weight.shape}, expected "
                f"({expected}, {self.action_size})"
            )

    def logits(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[1] != self.observation_size:
            raise ValueError(
                f"observation has {x.shape[1]} features, this policy was trained on "
                f"{self.observation_size}"
            )

        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / self.obs_std
        if self.obs_clip is not None:
            x = np.clip(x, -self.obs_clip, self.obs_clip)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        for weight, bias in self.layers:
            x = _activate(x @ weight + bias, self.activation)
        out = x @ self.action_weight + self.action_bias
        return out[0] if single else out

    def act(self, observation: np.ndarray) -> int:
        """Greedy action. Evaluation is deterministic — no sampling, ever.

        A stochastic policy in production would mean the same market state could
        produce a different order on a re-run, which makes an incident
        impossible to reconstruct.
        """
        return int(np.argmax(self.logits(observation)))

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        logits = self.logits(observation)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "format_version": np.asarray([WEIGHTS_FORMAT_VERSION]),
            "n_layers": np.asarray([len(self.layers)]),
            "action_weight": self.action_weight,
            "action_bias": self.action_bias,
            "meta": np.asarray([json.dumps({
                "activation": self.activation,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "obs_clip": self.obs_clip,
                **self.metadata,
            })]),
        }
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
        if self.obs_mean is not None:
            payload["obs_mean"] = self.obs_mean
        if self.obs_std is not None:
            payload["obs_std"] = self.obs_std
        np.savez_compressed(target, **payload)
        return target


def load_weights(path: str | Path) -> MLPPolicyWeights:
    with np.load(Path(path), allow_pickle=False) as payload:
        version = str(payload["format_version"][0])
        if version != WEIGHTS_FORMAT_VERSION:
            raise ValueError(
                f"weights format {version} is not {WEIGHTS_FORMAT_VERSION}; re-export "
                "the policy rather than guessing at the layout"
            )
        meta = json.loads(str(payload["meta"][0]))
        n_layers = int(payload["n_layers"][0])
        layers = [(payload[f"w{i}"], payload[f"b{i}"]) for i in range(n_layers)]
        return MLPPolicyWeights(
            layers=layers,
            action_weight=payload["action_weight"],
            action_bias=payload["action_bias"],
            activation=meta["activation"],
            observation_size=int(meta["observation_size"]),
            action_size=int(meta["action_size"]),
            obs_mean=payload["obs_mean"] if "obs_mean" in payload else None,
            obs_std=payload["obs_std"] if "obs_std" in payload else None,
            obs_clip=meta.get("obs_clip"),
            metadata={k: v for k, v in meta.items()
                      if k not in {"activation", "observation_size", "action_size", "obs_clip"}},
        )


class NumpyPolicy:
    """A loaded policy, ready to answer with an action index."""

    def __init__(self, weights: MLPPolicyWeights, name: str = "learned") -> None:
        self.weights = weights
        self.name = name

    @classmethod
    def from_file(cls, path: str | Path, name: str = "learned") -> "NumpyPolicy":
        return cls(load_weights(path), name=name)

    @property
    def observation_size(self) -> int:
        return self.weights.observation_size

    def act(self, observation: np.ndarray) -> int:
        return self.weights.act(observation)

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        return self.weights.probabilities(observation)
