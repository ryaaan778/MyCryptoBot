"""The reinforcement-learning environment.

A Gymnasium environment wrapped around :class:`research.backtest.Simulator`, so
the mechanics an agent learns against are byte-for-byte the mechanics its
results are later measured with. That is the single most important property
here: an environment with its own fill logic would train a policy against rules
that do not exist anywhere else, and the gap between those rules and the real
ones would look exactly like a discovered edge.

**The risk engine is inside the loop, not around it.** ``Simulator`` sizes every
position through the real ``RiskEngine`` and gates it through ``can_open``. The
agent chooses a *direction*, never a size, and an action that would breach an
exposure or drawdown cap simply does not happen — the environment reports it in
``info["blocked"]`` and the episode carries on. So the agent cannot learn its
way around a hard limit, because from inside the environment the limit is not a
penalty to be traded off, it is a wall. Requirement 17 holds structurally rather
than by convention.

**Observations are market features plus account state.** Market features are
precomputed once for the whole dataset, which is safe only because they are
causal — row ``i`` is a function of bars ``0..i`` and nothing else, and
``tests/test_research_leakage.py`` enforces that mechanically. Account state is
appended at each step because it depends on the agent's own actions.

**The scaler is required, not fitted here.** ``TradingEnv`` refuses to
normalise using statistics it computed itself, because the natural place to do
that is over the whole episode range — which leaks the future into every
observation. The caller fits a :class:`research.features.Scaler` on training
rows and passes it in. The inconvenience is the safeguard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:                        # pragma: no cover - env-dependent
    raise ImportError(
        "gymnasium is required for the RL environment. It lives in "
        "requirements-research.txt, isolated from the trading server: "
        "pip install -r requirements-research.txt"
    ) from exc

from backend.config import Settings

from .backtest import BacktestConfig, BacktestResult, Simulator
from .datasets import Dataset
from .features import FEATURE_SET_VERSION, FeatureMatrix, Scaler, build_features
from .policy import AccountState, PolicyAction
from .reward import RewardBreakdown, RewardConfig, reward_for_step

logger = logging.getLogger("research.env")

ENV_VERSION = "v1"

#: Observations are clipped to this many standard deviations. A z-scored
#: feature can spike arbitrarily far when the market leaves its training range,
#: and an unbounded input is a fast way to blow up a policy network's first
#: layer on exactly the bar that matters most.
OBSERVATION_CLIP = 10.0


@dataclass(frozen=True)
class EnvConfig:
    """Episode shape and bookkeeping.

    ``episode_bars`` with ``random_start`` is for training: sampling short
    windows from different points gives the agent varied market conditions and
    stops it memorising one path from one starting equity. Evaluation should use
    ``random_start=False`` and the full range, so every policy is scored on
    exactly the same bars.
    """

    episode_bars: int | None = None
    random_start: bool = True
    bankruptcy_equity_fraction: float = 0.2
    include_account_state: bool = True
    version: str = ENV_VERSION

    def validate(self) -> None:
        if self.episode_bars is not None and self.episode_bars < 2:
            raise ValueError("episode_bars must be at least 2")
        if not 0.0 <= self.bankruptcy_equity_fraction < 1.0:
            raise ValueError("bankruptcy_equity_fraction must be in [0, 1)")


class TradingEnv(gym.Env):
    """``Discrete(4)`` actions over one symbol, one position at a time."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: Dataset,
        *,
        scaler: Scaler,
        start: int = 0,
        stop: int | None = None,
        feature_names: tuple[str, ...] | None = None,
        features: FeatureMatrix | None = None,
        backtest_config: BacktestConfig | None = None,
        reward_config: RewardConfig | None = None,
        env_config: EnvConfig | None = None,
        settings: Settings | None = None,
        label: str = "rl",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.backtest_config = backtest_config or BacktestConfig()
        self.backtest_config.validate()
        self.reward_config = reward_config or RewardConfig()
        self.reward_config.validate()
        self.env_config = env_config or EnvConfig()
        self.env_config.validate()
        self.settings = settings
        self.label = label

        self.range_start = max(0, start)
        self.range_stop = len(dataset) if stop is None else min(stop, len(dataset))

        matrix = features if features is not None else build_features(dataset, feature_names)
        if matrix.names != scaler.names:
            raise ValueError(
                f"scaler was fitted on {scaler.names} but the feature matrix has "
                f"{matrix.names}; a policy fed differently-ordered inputs is not the "
                "policy that was trained"
            )
        self.features = matrix
        self.scaler = scaler
        self._scaled = scaler.transform(matrix).values
        np.clip(self._scaled, -OBSERVATION_CLIP, OBSERVATION_CLIP, out=self._scaled)
        # Warm-up rows are NaN by design; they are never observed because the
        # episode floor is the warm-up bar, but a stray NaN reaching a network
        # is silent and fatal, so they are zeroed as a second line of defence.
        np.nan_to_num(self._scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self.earliest_start = max(self.range_start, matrix.warmup)
        if self.range_stop - self.earliest_start < 2:
            raise ValueError(
                f"no usable bars: range [{self.range_start}, {self.range_stop}) starts "
                f"inside the {matrix.warmup}-bar feature warm-up"
            )

        self.n_market_features = matrix.n_features
        self.n_account_features = (
            len(AccountState.VECTOR_NAMES) if self.env_config.include_account_state else 0
        )
        self.observation_names = tuple(matrix.names) + (
            AccountState.VECTOR_NAMES if self.env_config.include_account_state else ()
        )

        self.action_space = spaces.Discrete(len(PolicyAction))
        self.observation_space = spaces.Box(
            low=-OBSERVATION_CLIP, high=OBSERVATION_CLIP,
            shape=(self.n_market_features + self.n_account_features,), dtype=np.float32,
        )

        self.simulator: Simulator | None = None
        self._episode_start = self.earliest_start
        self._episode_stop = self.range_stop
        self._reward_total = 0.0
        self._step_count = 0

    # ---- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        episode_bars = options.get("episode_bars", self.env_config.episode_bars)
        forced_start = options.get("start")

        if forced_start is not None:
            begin = int(forced_start)
        elif episode_bars is not None and self.env_config.random_start:
            latest = self.range_stop - episode_bars
            if latest <= self.earliest_start:
                begin = self.earliest_start
            else:
                begin = int(self.np_random.integers(self.earliest_start, latest + 1))
        else:
            begin = self.earliest_start

        begin = max(self.earliest_start, min(begin, self.range_stop - 2))
        end = self.range_stop if episode_bars is None else min(begin + episode_bars,
                                                               self.range_stop)
        if end - begin < 2:
            end = min(begin + 2, self.range_stop)

        self._episode_start, self._episode_stop = begin, end
        self.simulator = Simulator(
            self.dataset, self.backtest_config, settings=self.settings,
            start=begin, stop=end, label=self.label,
        )
        self._reward_total = 0.0
        self._step_count = 0
        return self._observe(), self._info()

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"action {action!r} is outside {self.action_space}")

        outcome = self.simulator.step(PolicyAction(int(action)))
        self.simulator.record_action(PolicyAction(int(action)))
        breakdown = reward_for_step(outcome, self.reward_config)
        self._reward_total += breakdown.total
        self._step_count += 1

        bankrupt = outcome.equity <= (
            self.backtest_config.starting_equity * self.env_config.bankruptcy_equity_fraction
        )
        # `terminated` means the episode reached a real end state (ruin);
        # `truncated` means it hit the edge of the data. Conflating them teaches
        # a value function that running out of bars is as bad as going broke.
        truncated = self.simulator.done and not bankrupt
        terminated = bool(bankrupt)
        if terminated or truncated:
            self.simulator.finish()

        return self._observe(), breakdown.total, terminated, truncated, self._info(
            outcome=outcome, breakdown=breakdown
        )

    # ---- observation and info ---------------------------------------------

    def _observe(self) -> np.ndarray:
        assert self.simulator is not None
        market = self._scaled[self.simulator.index]
        if not self.env_config.include_account_state:
            return market.astype(np.float32, copy=True)
        account = np.clip(self.simulator.account.as_vector(),
                          -OBSERVATION_CLIP, OBSERVATION_CLIP)
        return np.concatenate([market, account]).astype(np.float32, copy=False)

    def _info(self, outcome=None, breakdown: RewardBreakdown | None = None) -> dict[str, Any]:
        assert self.simulator is not None
        info: dict[str, Any] = {
            "index": self.simulator.index,
            "equity": float(self.simulator.portfolio.equity),
            "drawdown_pct": float(self.simulator.portfolio.drawdown_pct),
            "position_side": self.simulator.account.position_side,
            "trades": len(self.simulator.trades),
            "episode_start": self._episode_start,
            "episode_stop": self._episode_stop,
            "reward_total": self._reward_total,
            "steps": self._step_count,
        }
        if outcome is not None:
            info["opened"] = outcome.opened
            info["closed"] = outcome.closed is not None
            if outcome.blocked:
                # Surfaced rather than silently swallowed: an agent whose entries
                # are all being refused is a training run worth stopping early.
                info["blocked"] = outcome.blocked
        if breakdown is not None:
            info.update(breakdown.as_dict())
        return info

    # ---- evaluation --------------------------------------------------------

    def result(self, *, seed: int | None = None, window: int | None = None) -> BacktestResult:
        """The finished episode as a :class:`BacktestResult`.

        The same type the baselines produce, so a learned policy and a
        hand-written strategy go through one metric path and one promotion gate.
        """
        if self.simulator is None:
            raise RuntimeError("no episode has been run")
        self.simulator.finish()
        return self.simulator.result(seed=seed, window=window)

    def describe(self) -> dict[str, Any]:
        return {
            "env_version": ENV_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "reward_version": self.reward_config.version,
            "observation_size": int(self.observation_space.shape[0]),
            "market_features": self.n_market_features,
            "account_features": self.n_account_features,
            "actions": [a.name for a in PolicyAction],
            "range": [self.range_start, self.range_stop],
            "warmup": self.features.warmup,
        }


def make_training_env(
    dataset: Dataset,
    *,
    train_start: int,
    train_stop: int,
    feature_names: tuple[str, ...] | None = None,
    episode_bars: int | None = 2_000,
    backtest_config: BacktestConfig | None = None,
    reward_config: RewardConfig | None = None,
    settings: Settings | None = None,
) -> TradingEnv:
    """A training environment with the scaler fitted on training rows only.

    This is the safe path: it fits the scaler on ``[train_start, train_stop)``
    and nothing else, so the normalisation statistics an agent sees cannot carry
    information from the validation period. Use :func:`make_eval_env` with the
    *same* scaler afterwards — refitting on the evaluation window would leak
    exactly what the walk-forward split exists to prevent.
    """
    matrix = build_features(dataset, feature_names)
    rows = np.arange(max(train_start, matrix.warmup), train_stop)
    if rows.size == 0:
        raise ValueError(
            f"training range [{train_start}, {train_stop}) is entirely inside the "
            f"{matrix.warmup}-bar feature warm-up"
        )
    scaler = Scaler.fit(matrix, rows)
    return TradingEnv(
        dataset, scaler=scaler, start=train_start, stop=train_stop, features=matrix,
        backtest_config=backtest_config, reward_config=reward_config,
        env_config=EnvConfig(episode_bars=episode_bars, random_start=True),
        settings=settings, label="rl-train",
    )


def make_eval_env(
    dataset: Dataset,
    *,
    start: int,
    stop: int,
    scaler: Scaler,
    features: FeatureMatrix | None = None,
    feature_names: tuple[str, ...] | None = None,
    backtest_config: BacktestConfig | None = None,
    reward_config: RewardConfig | None = None,
    settings: Settings | None = None,
) -> TradingEnv:
    """A deterministic evaluation environment: full range, fixed start, given scaler."""
    return TradingEnv(
        dataset, scaler=scaler, start=start, stop=stop, features=features,
        feature_names=feature_names,
        backtest_config=backtest_config, reward_config=reward_config,
        env_config=EnvConfig(episode_bars=None, random_start=False),
        settings=settings, label="rl-eval",
    )
