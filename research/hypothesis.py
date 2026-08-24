"""The space an agent may search, and one point in it.

An :class:`ExperimentSpec` is a complete, hashable description of an experiment:
which features the policy sees, how the reward is weighted, the PPO
hyperparameters, and the risk envelope proposed for the backtest. Everything an
agent decides lives here, which is what makes an experiment reproducible from
its record alone — and what makes "have we tried this already?" a lookup rather
than a judgement call.

**The search is unbounded in kind, bounded in consequence.** An agent may
propose any subset of the feature library, any reward weighting, any
hyperparameters. A bad idea costs CPU and nothing else, and restricting the
search is how you accidentally forbid the one thing that would have worked. What
it may *not* do is propose its way past the risk engine: ``risk_per_trade`` and
``leverage`` are clamped to the configured hard caps at construction, so a spec
that asks for 50% per trade is silently a spec that asks for the cap. The agent
proposes; the risk engine still decides what reaches an order.

**Specs are content-addressed.** ``spec_id`` is a hash of every field, so the
same idea proposed twice by different agents on different days is recognisably
the same idea. Memory uses that to avoid re-running known failures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .features import DEFAULT_FEATURES, available_features
from .reward import RewardConfig

HYPOTHESIS_VERSION = "v1"

#: Hard ceilings the search cannot exceed, whatever an agent proposes. These
#: mirror the risk engine's own limits; the engine is still the authority, this
#: just stops the search wasting time on specs it would refuse anyway.
MAX_RISK_PER_TRADE = 0.10
MAX_LEVERAGE = 10.0
MIN_FEATURES = 4


@dataclass(frozen=True)
class ExperimentSpec:
    """One point in the search space."""

    features: tuple[str, ...]
    reward: RewardConfig
    learning_rate: float
    ent_coef: float
    gamma: float
    n_steps: int
    net_arch: tuple[int, ...]
    episode_bars: int
    risk_per_trade: float
    leverage: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    total_timesteps: int
    version: str = HYPOTHESIS_VERSION

    def __post_init__(self) -> None:
        if len(self.features) < MIN_FEATURES:
            raise ValueError(
                f"a policy needs at least {MIN_FEATURES} features, got {len(self.features)}"
            )
        unknown = [f for f in self.features if f not in available_features()]
        if unknown:
            raise ValueError(f"unknown features {unknown}")
        if len(set(self.features)) != len(self.features):
            raise ValueError("duplicate features in spec")
        # Clamp rather than reject: an over-eager proposal is a normal thing for
        # a search to emit, and turning it into the nearest legal spec keeps the
        # search moving instead of throwing away the rest of a good idea.
        object.__setattr__(self, "risk_per_trade",
                           float(min(max(self.risk_per_trade, 1e-4), MAX_RISK_PER_TRADE)))
        object.__setattr__(self, "leverage",
                           float(min(max(self.leverage, 1.0), MAX_LEVERAGE)))

    @property
    def spec_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": sorted(self.features),
            "reward": self.reward.as_dict(),
            "learning_rate": round(self.learning_rate, 8),
            "ent_coef": round(self.ent_coef, 8),
            "gamma": round(self.gamma, 6),
            "n_steps": self.n_steps,
            "net_arch": list(self.net_arch),
            "episode_bars": self.episode_bars,
            "risk_per_trade": round(self.risk_per_trade, 6),
            "leverage": round(self.leverage, 4),
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "total_timesteps": self.total_timesteps,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSpec":
        reward = payload["reward"]
        return cls(
            features=tuple(payload["features"]),
            reward=RewardConfig(
                lambda_drawdown=reward["lambda_drawdown"],
                lambda_turnover=reward["lambda_turnover"],
                lambda_tail=reward["lambda_tail"],
                clip=reward.get("clip", RewardConfig().clip),
                version=reward.get("version", RewardConfig().version),
            ),
            learning_rate=payload["learning_rate"],
            ent_coef=payload["ent_coef"],
            gamma=payload["gamma"],
            n_steps=payload["n_steps"],
            net_arch=tuple(payload["net_arch"]),
            episode_bars=payload["episode_bars"],
            risk_per_trade=payload["risk_per_trade"],
            leverage=payload["leverage"],
            stop_loss_pct=payload["stop_loss_pct"],
            take_profit_pct=payload["take_profit_pct"],
            total_timesteps=payload["total_timesteps"],
            version=payload.get("version", HYPOTHESIS_VERSION),
        )

    def distance(self, other: "ExperimentSpec") -> float:
        """How different two specs are, in [0, 1].

        Used to decide whether a proposal is "basically the thing that already
        failed". Feature overlap dominates because it is the choice that most
        determines what a policy can possibly learn; the continuous knobs
        contribute a smaller, log-scaled term so that a 3x learning rate counts
        for more than a 3% one.
        """
        mine, theirs = set(self.features), set(other.features)
        union = mine | theirs
        feature_distance = 1.0 - (len(mine & theirs) / len(union)) if union else 0.0

        def ratio(a: float, b: float) -> float:
            if a <= 0 or b <= 0:
                return 0.0 if a == b else 1.0
            return min(1.0, abs(np.log(a / b)) / np.log(10.0))

        knobs = [
            ratio(self.learning_rate, other.learning_rate),
            ratio(max(self.ent_coef, 1e-6), max(other.ent_coef, 1e-6)),
            ratio(self.risk_per_trade, other.risk_per_trade),
            ratio(self.reward.lambda_drawdown or 1e-6, other.reward.lambda_drawdown or 1e-6),
            ratio(self.reward.lambda_tail or 1e-6, other.reward.lambda_tail or 1e-6),
            0.0 if self.net_arch == other.net_arch else 1.0,
        ]
        return float(0.6 * feature_distance + 0.4 * np.mean(knobs))

    def describe(self) -> str:
        return (
            f"{self.spec_id}  {len(self.features):>2} features  "
            f"lr {self.learning_rate:.1e}  ent {self.ent_coef:.3f}  "
            f"risk {self.risk_per_trade:.1%}  lev {self.leverage:.0f}x  "
            f"dd {self.reward.lambda_drawdown:.2f}  tail {self.reward.lambda_tail:.2f}  "
            f"arch {list(self.net_arch)}"
        )


@dataclass(frozen=True)
class SearchRanges:
    """The bounds the sampler draws from.

    Wide on purpose. These are not opinions about what works — they are the
    limits of what is worth spending CPU on, and the only opinion baked in is
    that a policy trained on fewer than four features cannot see enough to
    learn anything.
    """

    n_features: tuple[int, int] = (4, len(DEFAULT_FEATURES))
    learning_rate: tuple[float, float] = (5e-5, 1e-3)
    ent_coef: tuple[float, float] = (0.0, 0.05)
    gamma: tuple[float, float] = (0.95, 0.9995)
    n_steps_choices: tuple[int, ...] = (512, 1_024, 2_048, 4_096)
    net_arch_choices: tuple[tuple[int, ...], ...] = (
        (32,), (64,), (64, 64), (128, 64), (128, 128),
    )
    episode_bars_choices: tuple[int, ...] = (500, 1_000, 2_000, 4_000)
    lambda_drawdown: tuple[float, float] = (0.0, 4.0)
    lambda_turnover: tuple[float, float] = (0.0, 2e-3)
    lambda_tail: tuple[float, float] = (0.0, 3.0)
    risk_per_trade: tuple[float, float] = (0.002, MAX_RISK_PER_TRADE)
    leverage: tuple[float, float] = (1.0, MAX_LEVERAGE)
    stop_loss_pct: tuple[float, float] = (0.3, 5.0)
    take_profit_choices: tuple[float | None, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, None)

    def validate(self) -> None:
        low, high = self.n_features
        if low < MIN_FEATURES or high > len(available_features()) or low > high:
            raise ValueError(f"n_features {self.n_features} is outside the feature library")
        for name in ("learning_rate", "ent_coef", "gamma", "lambda_drawdown",
                     "lambda_turnover", "lambda_tail", "risk_per_trade", "leverage",
                     "stop_loss_pct"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} range {(lo, hi)} runs backwards")


def sample_spec(
    rng: np.random.Generator,
    *,
    feature_weights: Mapping[str, float] | None = None,
    ranges: SearchRanges | None = None,
    total_timesteps: int = 200_000,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentSpec:
    """Draw one spec, optionally skewed by per-feature weights.

    ``feature_weights`` is the only place an agent's bias enters. It skews which
    features are *likely* to be drawn; it never makes any feature impossible, so
    an agent whose prior is wrong can still stumble onto the right inputs and —
    once the evidence accumulates — be steered there.
    """
    bounds = ranges or SearchRanges()
    bounds.validate()
    catalogue = list(available_features())

    weights = np.ones(len(catalogue))
    if feature_weights:
        for index, name in enumerate(catalogue):
            weights[index] = max(1e-3, float(feature_weights.get(name, 1.0)))
    probabilities = weights / weights.sum()

    low, high = bounds.n_features
    count = int(rng.integers(low, high + 1))
    chosen = rng.choice(len(catalogue), size=count, replace=False, p=probabilities)
    features = tuple(sorted(catalogue[i] for i in chosen))

    def log_uniform(pair: tuple[float, float]) -> float:
        lo, hi = pair
        return float(np.exp(rng.uniform(np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12)))))

    spec = ExperimentSpec(
        features=features,
        reward=RewardConfig(
            lambda_drawdown=float(rng.uniform(*bounds.lambda_drawdown)),
            lambda_turnover=float(rng.uniform(*bounds.lambda_turnover)),
            lambda_tail=float(rng.uniform(*bounds.lambda_tail)),
        ),
        learning_rate=log_uniform(bounds.learning_rate),
        ent_coef=float(rng.uniform(*bounds.ent_coef)),
        gamma=float(rng.uniform(*bounds.gamma)),
        n_steps=int(rng.choice(bounds.n_steps_choices)),
        net_arch=tuple(bounds.net_arch_choices[int(rng.integers(len(bounds.net_arch_choices)))]),
        episode_bars=int(rng.choice(bounds.episode_bars_choices)),
        risk_per_trade=log_uniform(bounds.risk_per_trade),
        leverage=float(rng.uniform(*bounds.leverage)),
        stop_loss_pct=float(rng.uniform(*bounds.stop_loss_pct)),
        take_profit_pct=bounds.take_profit_choices[
            int(rng.integers(len(bounds.take_profit_choices)))],
        total_timesteps=total_timesteps,
    )
    return replace(spec, **dict(overrides)) if overrides else spec
