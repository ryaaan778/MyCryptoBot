"""The RL environment: conformance, parity with the backtester, and no look-ahead.

The parity tests are the important ones. If the environment's mechanics drift
from the backtester's, a policy is trained under one set of rules and scored
under another, and the difference between them is indistinguishable from an
edge.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from research.backtest import BacktestConfig, Simulator, run_backtest
from research.collect import generate_synthetic
from research.env import (
    OBSERVATION_CLIP, EnvConfig, TradingEnv, make_eval_env, make_training_env,
)
from research.features import Scaler, build_features
from research.policy import AccountState, BasePolicy, Decision, PolicyAction, make_policy
from research.reward import RewardConfig, compute_reward, reward_for_step

TRAIN_STOP = 4_000


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=6_000, seed=5)


@pytest.fixture(scope="module")
def scaler(dataset):
    matrix = build_features(dataset)
    return Scaler.fit(matrix, np.arange(matrix.warmup, TRAIN_STOP))


@pytest.fixture
def env(dataset, scaler):
    return make_eval_env(dataset, start=0, stop=TRAIN_STOP, scaler=scaler)


# ---- reward ---------------------------------------------------------------


def test_doing_nothing_scores_exactly_zero():
    """Zero must beat negative, so a lazy policy is an honest local optimum."""
    idle = compute_reward(log_return=0.0, drawdown_pct=5.0, drawdown_before_pct=5.0,
                          position_changed=False)
    assert idle.total == 0.0


def test_only_new_drawdown_is_charged():
    deeper = compute_reward(log_return=0.0, drawdown_pct=12.0, drawdown_before_pct=10.0,
                            position_changed=False)
    assert deeper.total == pytest.approx(-0.02)          # 2 percentage points, as a fraction
    recovering = compute_reward(log_return=0.0, drawdown_pct=8.0, drawdown_before_pct=10.0,
                                position_changed=False)
    assert recovering.total == 0.0, "recovering from a drawdown must not be penalised"


def test_the_drawdown_term_is_scaled_from_percent_to_fraction():
    """A percent-scaled term with lambda=1.0 would be 100x too heavy."""
    penalty = compute_reward(log_return=0.0, drawdown_pct=1.0, drawdown_before_pct=0.0,
                             position_changed=False)
    assert penalty.total == pytest.approx(-0.01)
    assert penalty.total != pytest.approx(-1.0)


def test_the_tail_penalty_is_convex():
    def tail(log_return):
        return -compute_reward(log_return=log_return, drawdown_pct=0.0,
                               drawdown_before_pct=0.0,
                               position_changed=False).tail_penalty

    assert tail(-0.02) > 2 * tail(-0.01)
    assert tail(0.05) == 0.0, "upside must not be penalised"


def test_turnover_is_charged_once_per_position_change():
    config = RewardConfig()
    changed = compute_reward(log_return=0.0, drawdown_pct=0.0, drawdown_before_pct=0.0,
                             position_changed=True, config=config)
    assert changed.total == pytest.approx(-config.lambda_turnover)


def test_costs_are_not_charged_twice():
    """Fees are inside equity via PaperExecution; the reward must not re-subtract."""
    import inspect

    import research.reward as reward_module

    source = inspect.getsource(reward_module.compute_reward)
    assert "fee" not in source.lower()


def test_reward_terms_sum_to_the_total():
    breakdown = compute_reward(log_return=-0.03, drawdown_pct=9.0, drawdown_before_pct=4.0,
                               position_changed=True)
    assert breakdown.total == pytest.approx(
        breakdown.log_return + breakdown.drawdown_penalty
        + breakdown.turnover_penalty + breakdown.tail_penalty)


def test_a_catastrophic_bar_is_clipped_not_infinite():
    breakdown = compute_reward(log_return=-50.0, drawdown_pct=100.0, drawdown_before_pct=0.0,
                               position_changed=True)
    assert np.isfinite(breakdown.total)
    assert breakdown.total == -RewardConfig().clip and breakdown.clipped


@pytest.mark.parametrize("override", [{"lambda_drawdown": -1}, {"lambda_tail": -0.1},
                                      {"clip": 0}, {"lambda_turnover": np.inf}])
def test_malformed_reward_configs_are_rejected(override):
    with pytest.raises(ValueError):
        RewardConfig(**override).validate()


# ---- gymnasium conformance -------------------------------------------------


def test_spaces_are_well_formed(env):
    assert env.action_space.n == 4
    assert env.observation_space.shape == (31,)
    assert len(env.observation_names) == 31


def test_reset_returns_an_observation_inside_the_space(env):
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert np.all(np.isfinite(observation))
    assert info["equity"] == pytest.approx(10_000.0)


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_every_action_steps_cleanly(env, action):
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "equity" in info and "reward" in info


def test_an_out_of_range_action_is_rejected(env):
    env.reset(seed=1)
    for action in (-1, 4, 99):
        with pytest.raises(ValueError):
            env.step(action)


def test_stepping_before_reset_is_rejected(dataset, scaler):
    fresh = make_eval_env(dataset, start=0, stop=TRAIN_STOP, scaler=scaler)
    fresh.simulator = None
    with pytest.raises(RuntimeError):
        fresh.step(0)


def test_an_episode_truncates_at_the_end_of_its_range(dataset, scaler):
    env = TradingEnv(dataset, scaler=scaler, start=100, stop=400,
                     env_config=EnvConfig(episode_bars=None, random_start=False))
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(PolicyAction.HOLD)
        steps += 1
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert steps == 400 - max(100, env.features.warmup) - 1


def test_ruin_terminates_rather_than_truncates():
    """Running out of bars and going broke must not look the same to a value function.

    Built on a deliberately falling tape rather than a random one, so the test
    asserts the environment's behaviour instead of the sample's direction.
    """
    from research.datasets import DataSource, Dataset

    n = 2_000
    close = 1_000.0 * np.exp(np.linspace(0.0, -0.9, n))     # a steady, deep decline
    falling = Dataset.from_arrays(
        ts=np.arange(n, dtype=np.int64) * 300_000,
        open=close, high=close * 1.001, low=close * 0.999, close=close,
        volume=np.ones(n), source=DataSource.SYNTHETIC,
        symbol="BTC/USDT", timeframe="5m",
    )
    matrix = build_features(falling)
    fitted = Scaler.fit(matrix, np.arange(matrix.warmup, 1_000))
    env = TradingEnv(
        falling, scaler=fitted, features=matrix, start=matrix.warmup, stop=n,
        # Leverage 3 puts exposure at the 300% cap: any higher and the risk
        # engine refuses the entry outright, which is its job but not what this
        # test is about.
        backtest_config=BacktestConfig(starting_equity=10_000.0, risk_per_trade=0.5,
                                       leverage=3.0, stop_loss_pct=None,
                                       take_profit_pct=None, sizing_stop_pct=1.0),
        env_config=EnvConfig(episode_bars=None, random_start=False,
                             bankruptcy_equity_fraction=0.5),
    )
    env.reset(seed=0)
    terminated = truncated = False
    for _ in range(n):
        _, _, terminated, truncated, info = env.step(PolicyAction.LONG)
        if terminated or truncated:
            break
    assert terminated and not truncated, (
        f"a wiped-out account should terminate, not truncate (equity {info['equity']:.0f})")


def test_observations_are_bounded(dataset, scaler):
    env = make_eval_env(dataset, start=0, stop=TRAIN_STOP, scaler=scaler)
    observation, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(0, 4)))
        assert np.all(np.abs(observation) <= OBSERVATION_CLIP + 1e-6)
        assert np.all(np.isfinite(observation))
        if terminated or truncated:
            break


def test_no_observation_comes_from_the_feature_warmup(dataset, scaler):
    env = TradingEnv(dataset, scaler=scaler, start=0, stop=TRAIN_STOP,
                     env_config=EnvConfig(episode_bars=None, random_start=False))
    env.reset(seed=0)
    assert env.simulator.index >= env.features.warmup


def test_a_range_entirely_inside_the_warmup_is_rejected(dataset, scaler):
    with pytest.raises(ValueError, match="warm-up"):
        TradingEnv(dataset, scaler=scaler, start=0, stop=40)


def test_a_mismatched_scaler_is_rejected(dataset, scaler):
    other = build_features(dataset, ["rsi_14", "atr_pct"])
    with pytest.raises(ValueError, match="scaler"):
        TradingEnv(dataset, scaler=scaler, features=other, start=0, stop=TRAIN_STOP)


# ---- parity with the backtester -------------------------------------------


class Scripted(BasePolicy):
    """Replays a fixed action sequence, so env and backtester get identical input."""

    name = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)
        self.cursor = 0

    def reset(self, seed=None):
        self.cursor = 0

    def decide(self, ctx):
        action = self.actions[self.cursor % len(self.actions)]
        self.cursor += 1
        return Decision(action, 1.0, "scripted")


def test_the_environment_and_the_backtester_agree_bar_for_bar(dataset, scaler):
    """Same actions in, byte-identical equity curve and trades out."""
    rng = np.random.default_rng(11)
    script = [PolicyAction(int(a)) for a in rng.integers(0, 4, size=1_500)]
    config = BacktestConfig(starting_equity=10_000.0, risk_per_trade=0.02)

    start = build_features(dataset).warmup
    stop = 2_000
    backtest = run_backtest(Scripted(script), dataset, start=start, stop=stop, config=config)

    env = TradingEnv(dataset, scaler=scaler, start=start, stop=stop,
                     backtest_config=config,
                     env_config=EnvConfig(episode_bars=None, random_start=False))
    env.reset(seed=0)
    cursor = 0
    while True:
        cursor += 1                                   # bar 0's decision is script[0]
        _, _, terminated, truncated, _ = env.step(script[(cursor - 1) % len(script)])
        if terminated or truncated:
            break
    result = env.result()

    assert np.array_equal(result.equity, backtest.equity)
    assert result.trades == backtest.trades
    assert result.metrics.values == backtest.metrics.values


def test_the_risk_engine_still_refuses_entries_inside_the_environment(dataset, scaler):
    env = TradingEnv(dataset, scaler=scaler, start=100, stop=1_500,
                     backtest_config=BacktestConfig(max_exposure_pct=0.001),
                     env_config=EnvConfig(episode_bars=None, random_start=False))
    env.reset(seed=0)
    blocked = 0
    for _ in range(600):
        _, _, terminated, truncated, info = env.step(PolicyAction.LONG)
        blocked += "blocked" in info
        if terminated or truncated:
            break
    assert blocked > 0
    assert env.result().trades == []


def test_the_agent_cannot_choose_its_own_position_size(dataset, scaler):
    env = TradingEnv(dataset, scaler=scaler, start=100, stop=1_500,
                     backtest_config=BacktestConfig(risk_per_trade=0.02, leverage=1.0),
                     env_config=EnvConfig(episode_bars=None, random_start=False))
    env.reset(seed=0)
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(PolicyAction.LONG)
        if terminated or truncated:
            break
    result = env.result()
    assert result.trades
    assert all(trade.notional <= 10_000.0 * 1.05 for trade in result.trades)


# ---- determinism and episode sampling -------------------------------------


def test_the_same_seed_and_actions_reproduce_the_episode(dataset, scaler):
    def roll(seed):
        env = make_training_env(dataset, train_start=0, train_stop=TRAIN_STOP,
                                episode_bars=400)
        env.reset(seed=seed)
        rewards = []
        for k in range(200):
            _, reward, terminated, truncated, _ = env.step(k % 4)
            rewards.append(reward)
            if terminated or truncated:
                break
        return env._episode_start, rewards

    assert roll(7) == roll(7)
    assert roll(7)[0] != roll(8)[0] or roll(7)[1] != roll(8)[1]


def test_random_start_samples_different_windows(dataset, scaler):
    env = make_training_env(dataset, train_start=0, train_stop=TRAIN_STOP, episode_bars=300)
    starts = set()
    for seed in range(12):
        env.reset(seed=seed)
        starts.add(env._episode_start)
    assert len(starts) > 3


def test_evaluation_always_starts_at_the_same_bar(dataset, scaler):
    env = make_eval_env(dataset, start=0, stop=TRAIN_STOP, scaler=scaler)
    starts = {env.reset(seed=seed)[1]["episode_start"] for seed in range(5)}
    assert len(starts) == 1


def test_a_forced_start_is_honoured(dataset, scaler):
    env = make_eval_env(dataset, start=0, stop=TRAIN_STOP, scaler=scaler)
    _, info = env.reset(seed=0, options={"start": 1_234, "episode_bars": 200})
    assert info["episode_start"] == 1_234
    assert info["episode_stop"] == 1_434


# ---- the scaler safeguard --------------------------------------------------


def test_the_training_helper_fits_the_scaler_on_training_rows_only(dataset):
    env = make_training_env(dataset, train_start=0, train_stop=TRAIN_STOP)
    matrix = build_features(dataset)
    expected = Scaler.fit(matrix, np.arange(matrix.warmup, TRAIN_STOP))
    assert np.allclose(env.scaler.mean, expected.mean)
    assert np.allclose(env.scaler.std, expected.std)
    # and those statistics are genuinely different from whole-dataset ones
    whole = Scaler.fit(matrix, np.arange(matrix.warmup, len(dataset)))
    assert not np.allclose(env.scaler.mean, whole.mean)


def test_the_environment_will_not_fit_a_scaler_itself(dataset):
    """There is no code path where TradingEnv computes normalisation statistics."""
    with pytest.raises(TypeError):
        TradingEnv(dataset, start=0, stop=TRAIN_STOP)


# ---- results feed the same metric path ------------------------------------


def test_an_episode_produces_a_standard_backtest_result(dataset, scaler):
    env = make_eval_env(dataset, start=0, stop=1_500, scaler=scaler)
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    while True:
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 4)))
        if terminated or truncated:
            break
    result = env.result(seed=0, window=3)
    assert result.dataset_source == dataset.manifest.source
    assert result.window == 3
    assert result.metrics["trades"] == len(result.trades)
    assert result.equity[-1] - 10_000.0 == pytest.approx(
        sum(trade.pnl for trade in result.trades), abs=1e-6)


def test_the_reward_return_term_telescopes_to_realised_growth(dataset, scaler):
    """Σ Δlog_equity must equal log(final / initial).

    This is what keeps the reward tethered to reality. If the two ever diverge,
    the agent is being paid for something the equity curve did not do — which is
    the definition of a reward the backtest cannot cash.
    """
    env = make_eval_env(dataset, start=0, stop=2_000, scaler=scaler)
    env.reset(seed=1)
    rng = np.random.default_rng(3)
    total_log_return = 0.0
    while True:
        _, _, terminated, truncated, info = env.step(int(rng.integers(0, 4)))
        total_log_return += info["reward_log_return"]
        if terminated or truncated:
            break
    result = env.result()
    assert result.trades, "the random agent should have traded"
    assert total_log_return == pytest.approx(
        float(np.log(result.equity[-1] / result.equity[0])), abs=1e-9)


def test_a_hold_only_episode_scores_exactly_zero(dataset, scaler):
    """The bar every learned policy has to clear, and it is not approximate."""
    env = make_eval_env(dataset, start=0, stop=2_000, scaler=scaler)
    env.reset(seed=1)
    total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(PolicyAction.HOLD)
        total += reward
        if terminated or truncated:
            break
    assert total == 0.0
    assert env.result().trades == []


def test_the_environment_passes_the_gymnasium_conformance_checker(dataset, scaler):
    from gymnasium.utils.env_checker import check_env

    check_env(make_eval_env(dataset, start=0, stop=4_000, scaler=scaler),
              skip_render_check=True)
