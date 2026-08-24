"""Leakage tests — the ones that decide whether any other number here is real.

Every check follows the same shape: compute something at bar ``i``, then change
the future and recompute. If the answer moves, the backtest is reading data it
would not have had, and everything downstream of it is fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.backtest import BacktestConfig, run_backtest
from research.collect import generate_synthetic
from research.features import build_features
from research.policy import AccountState, PolicyContext, make_policy
from research.regimes import Regime, label_regimes


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=2500, seed=42)


def shock(dataset, from_index: int, factor: float = 4.0):
    """A copy with every bar after ``from_index`` violently changed."""
    mutated = generate_synthetic(
        bars=len(dataset), seed=dataset.manifest.seed, timeframe=dataset.manifest.timeframe
    )
    for name in ("open", "high", "low", "close", "volume"):
        getattr(mutated, name)[from_index + 1:] *= factor
    return mutated


# ---- features -------------------------------------------------------------


@pytest.mark.parametrize("index", [60, 500, 1200, 2498])
def test_feature_row_is_byte_identical_under_future_mutation(dataset, index):
    reference = build_features(dataset).values[index]
    mutated = build_features(shock(dataset, index)).values[index]
    assert np.array_equal(reference.view(np.uint8), mutated.view(np.uint8))


def test_no_feature_is_formed_before_its_indicator(dataset):
    """A feature must be NaN until the indicator behind it has warmed up.

    Filling warm-up NaN with a neutral value looks harmless and is not: the
    matrix then reports a warm-up of zero and a policy trains on fabricated
    observations that no live bar would ever produce.
    """
    matrix = build_features(dataset)
    earliest = {
        "ret_60": 60, "vol_60": 59, "vol_ratio": 59, "atr_pct": 13, "rsi_14": 14,
        "macd_hist": 33, "bb_position": 19, "ema_spread": 25, "trend_slope": 29,
        "dist_from_sma": 49,
    }
    for name, floor in earliest.items():
        column = matrix.values[:, matrix.names.index(name)]
        first_finite = int(np.argmax(np.isfinite(column)))
        assert first_finite >= floor, f"{name} formed at bar {first_finite}, before {floor}"


def test_feature_subset_reports_its_own_warmup(dataset):
    subset = build_features(dataset, ["atr_pct", "bb_width"])
    assert subset.warmup >= 19


def test_scaler_cannot_be_fitted_on_everything_by_default(dataset):
    """Fitting normalisation over the whole series leaks test statistics."""
    from research.features import Scaler

    matrix = build_features(dataset)
    with pytest.raises(TypeError):
        Scaler.fit(matrix)          # rows is required, not optional
    with pytest.raises(ValueError):
        Scaler.fit(matrix, [])


def test_scaler_fitted_on_train_does_not_normalise_test_to_zero_mean(dataset):
    from research.features import Scaler

    matrix = build_features(dataset)
    train = np.arange(matrix.warmup, 1800)
    scaled = Scaler.fit(matrix, train).transform(matrix)
    assert np.abs(scaled.values[train].mean(axis=0)).max() < 1e-9
    # If the test rows also came out centred, the scaler had seen them.
    assert np.abs(scaled.values[1800:].mean(axis=0)).max() > 1e-6


# ---- regimes --------------------------------------------------------------


def test_regime_labels_do_not_change_when_the_future_arrives(dataset):
    """The subtle leak: thresholds fitted on the whole dataset.

    Labelling a prefix on its own must give the same answer as labelling it as
    part of a longer series. If it does not, every label knew the future
    distribution of volatility.
    """
    full = label_regimes(dataset)
    for cut in (1200, 2000):
        prefix = generate_synthetic(bars=cut, seed=dataset.manifest.seed)
        assert np.array_equal(prefix.close, dataset.close[:cut])
        assert np.array_equal(label_regimes(prefix).regime, full.regime[:cut])


def test_regime_labels_survive_a_future_shock(dataset):
    index = 1800
    reference = label_regimes(dataset).regime[:index + 1]
    assert np.array_equal(label_regimes(shock(dataset, index)).regime[:index + 1], reference)


def test_uncalibrated_head_is_unknown_rather_than_guessed(dataset):
    labels = label_regimes(dataset)
    assert np.all(labels.regime[:labels.warmup] == int(Regime.UNKNOWN))
    assert not np.any(labels.regime[labels.warmup:] == int(Regime.UNKNOWN))


def test_short_dataset_refuses_to_label_at_all():
    labels = label_regimes(generate_synthetic(bars=400, seed=3))
    assert np.all(labels.regime == int(Regime.UNKNOWN))


# ---- policy context and backtest ------------------------------------------


def test_policy_context_series_never_reaches_past_the_current_bar(dataset):
    account = AccountState(equity=10_000.0, starting_equity=10_000.0)
    ctx = PolicyContext(dataset=dataset, index=900, account=account)
    series = ctx.series(400)
    assert series.close[-1] == dataset.close[900]
    assert series.ts[-1] == dataset.ts[900]
    assert series.close.size == 400
    assert PolicyContext(dataset=dataset, index=5, account=account).series(400).close.size == 6


@pytest.mark.parametrize("policy_name", ["mean_reversion", "momentum", "volatility_breakout"])
def test_backtest_is_unchanged_by_bars_after_the_window(dataset, settings, policy_name):
    config = BacktestConfig(starting_equity=10_000.0, risk_per_trade=0.02)
    kwargs = dict(start=200, stop=1500, config=config)
    baseline = run_backtest(make_policy(policy_name, settings.strategy_parameters),
                            dataset, **kwargs)
    mutated = run_backtest(make_policy(policy_name, settings.strategy_parameters),
                           shock(dataset, 1500, factor=3.0), **kwargs)
    assert np.array_equal(baseline.equity, mutated.equity)
    assert baseline.trades == mutated.trades
