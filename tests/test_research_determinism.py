"""Determinism and the metric definitions themselves.

Reproducibility is the precondition for every other claim: an experiment that
cannot be re-run bit-for-bit cannot be audited, compared, or trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.collect import generate_synthetic
from research.datasets import Dataset
from research.features import build_features
from research.metrics import (
    ANNUALISED_LOG_CAP, PROFIT_FACTOR_CAP, ExitReason, TradeRecord, aggregate,
    annualised_return, compute_metrics, daily_returns, longest_losing_streak,
    max_drawdown, sharpe, sortino,
)
from research.policy import AccountState, PolicyContext, RandomPolicy, make_policy
from research.regimes import label_regimes

BARS_PER_YEAR = 288 * 365


def ts_for(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * 300_000


def trade(pnl: float, i: int = 0, j: int = 10, fees: float = 0.0,
          notional: float = 1000.0, reason: ExitReason = ExitReason.SIGNAL) -> TradeRecord:
    quantity = notional / 100.0
    return TradeRecord(entry_index=i, exit_index=j, side="LONG", entry_price=100.0,
                       exit_price=100.0 + pnl / quantity, quantity=quantity,
                       pnl=pnl, fees=fees, reason=reason.value)


# ---- determinism ----------------------------------------------------------


def test_the_same_seed_produces_the_same_bytes():
    a = generate_synthetic(bars=1200, seed=7, start_ts=0)
    b = generate_synthetic(bars=1200, seed=7, start_ts=0)
    assert a.manifest.sha256 == b.manifest.sha256
    assert generate_synthetic(bars=1200, seed=8, start_ts=0).manifest.sha256 != a.manifest.sha256


def test_a_dataset_id_is_a_function_of_its_bytes():
    a = generate_synthetic(bars=1200, seed=7, start_ts=0)
    b = generate_synthetic(bars=1200, seed=7, start_ts=0)
    assert a.manifest.dataset_id == b.manifest.dataset_id


def test_features_and_regimes_are_deterministic():
    dataset = generate_synthetic(bars=2500, seed=11)
    first = build_features(dataset)
    second = build_features(dataset)
    assert np.array_equal(first.values.view(np.uint8), second.values.view(np.uint8))
    assert np.array_equal(label_regimes(dataset).regime, label_regimes(dataset).regime)


def test_a_dataset_round_trips_through_disk_unchanged(tmp_path):
    dataset = generate_synthetic(bars=900, seed=5)
    dataset.save(tmp_path)
    reloaded = Dataset.load(tmp_path, dataset.manifest.dataset_id)
    assert reloaded.manifest == dataset.manifest
    assert np.array_equal(reloaded.close, dataset.close)


def test_random_policy_is_reproducible_across_reset():
    dataset = generate_synthetic(bars=600, seed=2)
    account = AccountState(equity=10_000.0, starting_equity=10_000.0)

    def roll(policy):
        return [policy.decide(PolicyContext(dataset=dataset, index=i, account=account)).action
                for i in range(300)]

    policy = RandomPolicy(seed=7, trade_probability=0.3)
    first = roll(policy)
    policy.reset()
    assert roll(policy) == first
    assert roll(RandomPolicy(seed=7, trade_probability=0.3)) == first
    assert roll(RandomPolicy(seed=8, trade_probability=0.3)) != first


def test_strategy_policies_match_the_live_evaluate_path(settings):
    """The adapter must be the same decision the live bot would make."""
    from backend.models import Candle
    from backend.strategies import create_strategy
    from research.policy import LIVE_WINDOW_MARGIN, StrategyPolicy

    dataset = generate_synthetic(bars=900, seed=42)
    candles = [Candle(ts=int(dataset.ts[k]), open=float(dataset.open[k]),
                      high=float(dataset.high[k]), low=float(dataset.low[k]),
                      close=float(dataset.close[k]), volume=float(dataset.volume[k]))
               for k in range(len(dataset))]
    account = AccountState(equity=10_000.0, starting_equity=10_000.0)

    for name in ("scalping", "momentum", "mean_reversion", "volatility_breakout", "hybrid"):
        strategy = create_strategy(name, settings.strategy_parameters)
        policy = StrategyPolicy.from_name(name, settings.strategy_parameters)
        assert policy.lookback == strategy.min_bars + LIVE_WINDOW_MARGIN
        for i in range(0, len(dataset), 23):
            live = strategy.evaluate("research", dataset.manifest.symbol,
                                     candles[max(0, i + 1 - policy.lookback): i + 1], None)
            got = policy.decide(PolicyContext(dataset=dataset, index=i, account=account))
            assert got.action.name == live.action.value
            assert got.confidence == pytest.approx(live.confidence)


# ---- metric definitions ---------------------------------------------------


def test_returns_compound_rather_than_summing():
    """-20% then +20% is -4%, which is what the legacy report got wrong."""
    metrics = compute_metrics(np.array([10_000.0, 8_000.0, 9_600.0]), ts_for(3), [],
                              bars_per_year=BARS_PER_YEAR)
    assert metrics["total_return_pct"] == pytest.approx(-4.0)


def test_max_drawdown_finds_the_deepest_peak_to_trough():
    depth, peak, trough = max_drawdown(np.array([100.0, 120.0, 60.0, 90.0, 80.0, 150.0]))
    assert (depth, peak, trough) == (pytest.approx(0.5), 1, 2)
    assert max_drawdown(np.array([100.0, 110.0, 120.0]))[0] == 0.0


def test_annualised_return_compounds():
    n = BARS_PER_YEAR + 1
    assert annualised_return(np.geomspace(10_000.0, 20_000.0, n), n,
                             BARS_PER_YEAR) == pytest.approx(100.0, abs=0.5)


def test_a_short_window_clamps_instead_of_overflowing_to_infinity():
    """1.02 ** 52560 is inf, and one inf nan-poisons every median it enters."""
    metrics = compute_metrics(np.array([10_000.0, 10_200.0]), ts_for(2), [],
                              bars_per_year=BARS_PER_YEAR)
    assert np.isfinite(metrics["annualised_return_pct"])
    assert metrics["annualised_return_pct"] == pytest.approx(
        math.expm1(ANNUALISED_LOG_CAP) * 100)


def test_ruin_reports_a_number_rather_than_nan():
    metrics = compute_metrics(np.array([10_000.0, 5_000.0, 0.0]), ts_for(3), [],
                              bars_per_year=BARS_PER_YEAR)
    assert metrics["annualised_return_pct"] == -100.0
    assert metrics["max_drawdown_pct"] == 100.0
    assert all(np.isfinite(v) for v in metrics.values.values())


def test_sortino_uses_the_total_period_denominator():
    returns = np.concatenate([np.full(999, 0.001), [-0.9]])
    negatives_only = float(returns.mean() / 0.9 * math.sqrt(BARS_PER_YEAR))
    assert sortino(returns, BARS_PER_YEAR) > negatives_only


def test_zero_variance_gains_are_not_infinite_sharpe():
    assert sharpe(np.full(500, 0.0001), BARS_PER_YEAR) == 0.0
    assert sharpe(np.zeros(500), BARS_PER_YEAR) == 0.0


def test_worst_day_is_a_calendar_day_not_a_bar():
    bars = 288 * 5
    equity = np.full(bars, 10_000.0)
    equity[288 * 2:] *= 0.80
    equity[288 * 3:] *= 1.05
    metrics = compute_metrics(equity, ts_for(bars), [], bars_per_year=BARS_PER_YEAR)
    assert metrics["worst_day_pct"] == pytest.approx(-20.0)
    assert metrics["best_day_pct"] == pytest.approx(5.0)
    returns, _ = daily_returns(equity, ts_for(bars))
    assert returns.size == 5 and int(np.argmin(returns)) == 2


def test_trade_statistics_are_hand_checkable():
    trades = [trade(100), trade(100), trade(100), trade(-50), trade(-50)]
    metrics = compute_metrics(np.array([10_000.0, 10_200.0]), ts_for(2), trades,
                              bars_per_year=BARS_PER_YEAR)
    assert metrics["win_rate_pct"] == pytest.approx(60.0)
    assert metrics["realised_rr"] == pytest.approx(2.0)
    assert metrics["profit_factor"] == pytest.approx(3.0)
    assert metrics["expectancy"] == pytest.approx(40.0)


def test_a_run_without_losses_caps_instead_of_returning_infinity():
    metrics = compute_metrics(np.array([10_000.0, 10_200.0]), ts_for(2),
                              [trade(100), trade(50)], bars_per_year=BARS_PER_YEAR)
    assert metrics["profit_factor"] == PROFIT_FACTOR_CAP
    assert all(np.isfinite(v) for v in metrics.values.values())


def test_longest_losing_streak_counts_consecutive_losses():
    assert longest_losing_streak(
        [trade(1), trade(-1), trade(-1), trade(-1), trade(1), trade(-1)]) == 3
    assert longest_losing_streak([]) == 0


def test_exit_reasons_expose_a_target_that_never_fires():
    trades = ([trade(-50, reason=ExitReason.STOP_LOSS)] * 90
              + [trade(700, reason=ExitReason.TAKE_PROFIT)]
              + [trade(10, reason=ExitReason.SIGNAL)] * 9)
    metrics = compute_metrics(np.array([10_000.0, 10_200.0]), ts_for(2), trades,
                              bars_per_year=BARS_PER_YEAR)
    assert metrics["exit_take_profit_pct"] == pytest.approx(1.0)
    assert metrics["exit_stop_loss_pct"] == pytest.approx(90.0)


def test_aggregation_reports_the_median_not_the_best_run():
    runs = [compute_metrics(np.array([10_000.0, 10_000.0 * (1 + r / 100)]), ts_for(2),
                            [trade(1)], bars_per_year=BARS_PER_YEAR)
            for r in (-5.0, 1.0, 2.0, 3.0, 40.0)]
    summary = aggregate(runs)["total_return_pct"]
    assert summary.median == pytest.approx(2.0)
    assert summary.iqr == pytest.approx(2.0)
    assert summary.maximum == pytest.approx(40.0)
    assert summary.n == 5


def test_the_metrics_module_offers_no_best_of_n_helper():
    """Best-of-N is how a research project fools itself; it is not a convenience."""
    import research.metrics as metrics_module

    assert not any(name in dir(metrics_module) for name in ("best", "best_of", "best_run"))


def test_wide_spread_across_seeds_is_flagged_as_unstable():
    runs = [compute_metrics(np.array([10_000.0, 10_000.0 * (1 + r / 100)]), ts_for(2),
                            [trade(1)], bars_per_year=BARS_PER_YEAR)
            for r in (-30.0, -10.0, 1.0, 12.0, 35.0)]
    assert not aggregate(runs)["total_return_pct"].is_stable


def test_long_format_rows_carry_dataset_identity():
    metrics = compute_metrics(np.array([10_000.0, 10_200.0]), ts_for(2), [trade(1)],
                              bars_per_year=BARS_PER_YEAR)
    rows = metrics.to_rows(policy_id="p1", dataset_id="d1", split="w0", seed=7)
    assert len(rows) == len(metrics.values)
    assert all({"metric", "value", "policy_id", "dataset_id"} <= set(row) for row in rows)


@pytest.mark.parametrize("equity,ts", [(np.array([]), np.array([])),
                                       (np.array([1.0, 2.0]), ts_for(5))])
def test_degenerate_metric_inputs_are_rejected(equity, ts):
    with pytest.raises(ValueError):
        compute_metrics(equity, ts, [], bars_per_year=BARS_PER_YEAR)
