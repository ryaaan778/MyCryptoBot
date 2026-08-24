"""PPO training, and the export that keeps torch out of the trading server.

Training produces two artefacts from every run:

* ``model.zip`` — the SB3 checkpoint, for resuming or fine-tuning.
* ``weights.npz`` + ``metadata.json`` — a plain numpy dump that
  ``backend.policy`` reads with a hand-written forward pass.

The second exists so the process that places orders never imports torch. A
broken CUDA install or a version conflict should not be able to take down
something holding positions, and 500MB of GPU libraries have no business in a
trading loop that does four matrix multiplies per bar.
``tests/test_research_train.py`` asserts the two agree to 1e-6 on identical
observations; if they diverge, the policy that trades is not the policy that was
validated.

**Why PPO.** On-policy, so there is no replay buffer accumulating experience
from a market regime that no longer exists — which is the specific way DQN
tends to fail on non-stationary financial data. Its clipped objective is
forgiving of imperfect hyperparameters, which matters because we cannot afford
large sweeps. Discrete actions are native.

**Honest expectation.** RL on a single price series overfits aggressively, and
PPO will happily memorise a path. Most trained policies *should* fail to beat
the baselines out of sample. That is the harness working, not a bug — and it is
why the baselines and the walk-forward split were built first.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.policy.numpy_infer import MLPPolicyWeights
from research.backtest import BacktestConfig
from research.datasets import Dataset
from research.env import ENV_VERSION, OBSERVATION_CLIP, EnvConfig, TradingEnv
from research.features import FEATURE_SET_VERSION, FeatureMatrix, Scaler, build_features
from research.policy import AccountState, PolicyAction
from research.reward import RewardConfig

logger = logging.getLogger("research.train")

TRAIN_VERSION = "v1"


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:                       # pragma: no cover - env-dependent
        raise ImportError(
            "stable-baselines3 and torch are required for training. They are isolated "
            "in requirements-research.txt so the trading server never imports them:\n"
            "  pip install -r requirements-research.txt "
            "--extra-index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return PPO, Monitor, DummyVecEnv


@dataclass(frozen=True)
class TrainConfig:
    """PPO hyperparameters and episode shape.

    The defaults are SB3's, with three changes. ``n_steps`` is larger because a
    trading episode's reward is dominated by rare events and a short rollout
    sees none of them. ``ent_coef`` is non-zero because with a
    do-nothing action that scores exactly zero, a policy collapses to
    always-HOLD almost immediately without some pressure to keep exploring.
    ``net_arch`` is small on purpose: 31 inputs and a few thousand episodes do
    not support a large network, and a bigger one just memorises faster.
    """

    total_timesteps: int = 200_000
    n_steps: int = 2_048
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    episode_bars: int = 2_000
    n_envs: int = 4
    seed: int = 0
    version: str = TRAIN_VERSION

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError(
                f"batch_size {self.batch_size} exceeds one rollout "
                f"({self.n_steps} x {self.n_envs}); PPO would train on a partial batch"
            )
        if self.activation not in ("tanh", "relu"):
            raise ValueError(f"unsupported activation {self.activation!r}")
        if not self.net_arch:
            raise ValueError("net_arch cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "net_arch": list(self.net_arch)}


@dataclass
class TrainedPolicy:
    """Everything a run produced, including where the artefacts landed."""

    policy_id: str
    agent: str
    version: int
    directory: Path
    weights: MLPPolicyWeights
    scaler: Scaler
    feature_names: tuple[str, ...]
    train_config: TrainConfig
    reward_config: RewardConfig
    backtest_config: BacktestConfig
    trained_on: dict[str, Any]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_path(self) -> Path:
        return self.directory / "weights.npz"

    @property
    def model_path(self) -> Path:
        return self.directory / "model.zip"


def export_weights(
    model, *, activation: str, scaler: Scaler | None = None, obs_clip: float | None = None,
    metadata: dict | None = None,
) -> MLPPolicyWeights:
    """Pull the policy network out of an SB3 model into plain numpy arrays.

    SB3's ``ActorCriticPolicy`` at evaluation time is: flatten the observation,
    run it through ``mlp_extractor.policy_net``, then ``action_net``. The value
    head is not exported — it is used for training and never for choosing an
    action, so shipping it would only be one more thing that could disagree.

    torch stores ``Linear`` weights as ``(out, in)`` and computes ``x @ W.T``;
    here they are transposed once at export so the runtime is a plain ``x @ W``.
    """
    import torch
    from torch import nn

    policy = model.policy
    if policy.action_dist.__class__.__name__ != "CategoricalDistribution":
        raise ValueError(
            f"only discrete-action policies can be exported, got "
            f"{policy.action_dist.__class__.__name__}"
        )

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                layers.append((
                    module.weight.detach().cpu().numpy().T.astype(np.float64),
                    module.bias.detach().cpu().numpy().astype(np.float64),
                ))
        elif isinstance(module, (nn.Tanh, nn.ReLU)):
            continue
        else:
            raise ValueError(
                f"unexpected layer {type(module).__name__} in the policy network; the "
                "numpy runtime only implements Linear + tanh/relu, and silently "
                "skipping a layer would produce a policy that is not the one trained"
            )

    with torch.no_grad():
        action_weight = policy.action_net.weight.detach().cpu().numpy().T.astype(np.float64)
        action_bias = policy.action_net.bias.detach().cpu().numpy().astype(np.float64)

    observation_size = int(np.prod(policy.observation_space.shape))
    return MLPPolicyWeights(
        layers=layers,
        action_weight=action_weight,
        action_bias=action_bias,
        activation=activation,
        observation_size=observation_size,
        action_size=int(action_weight.shape[1]),
        obs_mean=None,      # the environment already scales; see note in train()
        obs_std=None,
        obs_clip=obs_clip,
        metadata=metadata or {},
    )


def train_policy(
    dataset: Dataset,
    *,
    agent: str,
    train_start: int,
    train_stop: int,
    root: str | Path = "data",
    feature_names: Sequence[str] | None = None,
    train_config: TrainConfig | None = None,
    reward_config: RewardConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    version: int = 1,
    policy_id: str | None = None,
    progress: bool = False,
) -> TrainedPolicy:
    """Train one PPO policy on ``dataset[train_start:train_stop]``.

    The scaler is fitted on the training rows only and saved alongside the
    weights. Evaluation must reuse it: refitting on the evaluation window would
    feed the policy normalisation statistics derived from the very bars it is
    being tested on, which is the quietest form of leakage there is.

    One run is one seed. Seed diversity has to come from training several
    policies, not from evaluating one policy several times — inference is
    ``argmax`` and therefore deterministic, so N evaluation seeds of a single
    policy are N identical runs wearing different labels. See
    :func:`train_seed_ensemble`.
    """
    PPO, Monitor, DummyVecEnv = _require_sb3()

    cfg = train_config or TrainConfig()
    cfg.validate()
    reward = reward_config or RewardConfig()
    reward.validate()
    backtest = backtest_config or BacktestConfig()
    backtest.validate()

    matrix: FeatureMatrix = build_features(dataset, feature_names)
    rows = np.arange(max(train_start, matrix.warmup), train_stop)
    if rows.size < cfg.episode_bars:
        raise ValueError(
            f"training range gives {rows.size:,} usable bars but episode_bars is "
            f"{cfg.episode_bars:,}; shorten the episode or collect more history"
        )
    scaler = Scaler.fit(matrix, rows)

    def make_env(rank: int):
        def build():
            env = TradingEnv(
                dataset, scaler=scaler, features=matrix,
                start=train_start, stop=train_stop,
                backtest_config=backtest, reward_config=reward,
                env_config=EnvConfig(episode_bars=cfg.episode_bars, random_start=True),
                label=f"{agent}-train-{rank}",
            )
            return Monitor(env)
        return build

    vec_env = DummyVecEnv([make_env(i) for i in range(cfg.n_envs)])
    vec_env.seed(cfg.seed)

    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm, seed=cfg.seed, device="cpu",
        policy_kwargs={"net_arch": list(cfg.net_arch),
                       "activation_fn": _activation_module(cfg.activation)},
        verbose=1 if progress else 0,
    )

    started = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
    elapsed = time.time() - started
    logger.info("%s v%d: %d timesteps in %.1fs", agent, version, cfg.total_timesteps, elapsed)

    policy_id = policy_id or f"{agent}_v{version}"
    directory = Path(root) / "policies" / policy_id
    directory.mkdir(parents=True, exist_ok=True)

    weights = export_weights(
        model, activation=cfg.activation, scaler=scaler, obs_clip=OBSERVATION_CLIP,
        metadata={"policy_id": policy_id, "agent": agent, "algo": "PPO"},
    )
    weights.save(directory / "weights.npz")
    model.save(directory / "model.zip")

    trained_on = {
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_sha256": dataset.manifest.sha256,
        "source": dataset.manifest.source,
        "symbol": dataset.manifest.symbol,
        "timeframe": dataset.manifest.timeframe,
        "train_start": int(train_start),
        "train_stop": int(train_stop),
        "usable_bars": int(rows.size),
        "wall_seconds": round(elapsed, 1),
    }
    metadata = {
        "policy_id": policy_id,
        "agent": agent,
        "version": version,
        "algo": "PPO",
        "feature_names": list(matrix.names),
        "feature_set_version": FEATURE_SET_VERSION,
        "account_feature_names": list(AccountState.VECTOR_NAMES),
        "reward_version": reward.version,
        "env_version": ENV_VERSION,
        "seed": cfg.seed,
        "actions": [a.name for a in PolicyAction],
        "train_config": cfg.as_dict(),
        "reward_config": reward.as_dict(),
        "scaler": scaler.to_dict(),
        "trained_on": trained_on,
        "created_at": int(time.time() * 1000),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    vec_env.close()
    return TrainedPolicy(
        policy_id=policy_id, agent=agent, version=version, directory=directory,
        weights=weights, scaler=scaler, feature_names=tuple(matrix.names),
        train_config=cfg, reward_config=reward, backtest_config=backtest,
        trained_on=trained_on, seed=cfg.seed, metadata=metadata,
    )


def _activation_module(name: str):
    from torch import nn

    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


def train_seed_ensemble(
    dataset: Dataset,
    *,
    agent: str,
    train_start: int,
    train_stop: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    version: int = 1,
    train_config: TrainConfig | None = None,
    **kwargs,
) -> list[TrainedPolicy]:
    """Train one policy per seed — the only honest source of seed diversity here.

    The promotion gate demands a positive median across at least five seeds, and
    that requirement only means something if the seeds actually change the
    outcome. Evaluating a single deterministic policy five times does not: it
    produces five identical runs and a median that says nothing about
    robustness. Training five policies does, because the seed changes the
    initialisation, the rollout sampling and the batch order — everything the
    result should be robust to.
    """
    base = train_config or TrainConfig()
    trained: list[TrainedPolicy] = []
    for seed in seeds:
        from dataclasses import replace as dc_replace

        trained.append(train_policy(
            dataset, agent=agent, train_start=train_start, train_stop=train_stop,
            train_config=dc_replace(base, seed=seed), version=version,
            policy_id=f"{agent}_v{version}_s{seed}", **kwargs,
        ))
        logger.info("%s seed %d done (%d/%d)", agent, seed, len(trained), len(seeds))
    return trained
