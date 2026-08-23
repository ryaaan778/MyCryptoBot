"""Indicators checked against values computed by hand from the definitions."""

from __future__ import annotations

import numpy as np
import pytest

from backend.strategies.indicators import (
    atr,
    bollinger,
    ema,
    last_valid,
    macd,
    rsi,
    sma,
    true_range,
    wilder_smooth,
)

RAMP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def test_sma_matches_definition():
    result = sma(RAMP, 3)
    assert np.isnan(result[:2]).all(), "warm-up region must be nan"
    assert result[2] == pytest.approx(2.0)      # (1+2+3)/3
    assert result[-1] == pytest.approx(9.0)     # (8+9+10)/3


def test_ema_is_sma_seeded():
    result = ema(RAMP, 3)
    assert np.isnan(result[:2]).all()
    assert result[2] == pytest.approx(2.0)      # seed = SMA of first 3
    # alpha = 2/(3+1) = 0.5 -> next = 0.5*4 + 0.5*2 = 3.0
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_wilder_smoothing_recurrence():
    values = [1.0, 1.5, 1.5, 1.5, 1.5]
    result = wilder_smooth(values, 2)
    assert result[1] == pytest.approx(1.25)                     # seed = (1+1.5)/2
    assert result[2] == pytest.approx((1.25 * 1 + 1.5) / 2)     # 1.375
    assert result[3] == pytest.approx(1.4375)


def test_rsi_saturates_on_unbroken_gains():
    assert rsi(RAMP, 3)[-1] == pytest.approx(100.0)


def test_rsi_bottoms_on_unbroken_losses():
    assert rsi(list(reversed(RAMP)), 3)[-1] == pytest.approx(0.0)


def test_rsi_flat_series_is_neutral():
    assert rsi([5.0] * 20, 5)[-1] == pytest.approx(50.0)


def test_true_range_uses_previous_close():
    highs = np.array([2.0, 3.0])
    lows = np.array([1.0, 2.0])
    closes = np.array([1.5, 2.5])
    tr = true_range(highs, lows, closes)
    assert tr[0] == pytest.approx(1.0)                  # first bar: high - low
    assert tr[1] == pytest.approx(1.5)                  # |high - prev_close|


def test_atr_applies_wilder_smoothing():
    highs = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    lows = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    closes = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    # TR = [1.0, 1.5, 1.5, 1.5, 1.5]; Wilder(2) converges toward 1.5 from 1.25.
    assert atr(highs, lows, closes, 2)[-1] == pytest.approx(1.46875)


def test_bollinger_uses_population_std():
    upper, middle, lower = bollinger([1.0, 2.0, 3.0, 4.0, 5.0], 3, 2.0)
    expected_std = float(np.std([3.0, 4.0, 5.0]))
    assert middle[-1] == pytest.approx(4.0)
    assert upper[-1] == pytest.approx(4.0 + 2.0 * expected_std)
    assert lower[-1] == pytest.approx(4.0 - 2.0 * expected_std)


def test_macd_histogram_is_line_minus_signal():
    closes = list(np.linspace(100, 140, 80))
    line, signal, hist = macd(closes, 12, 26, 9)
    assert np.isfinite(line[-1]) and np.isfinite(signal[-1])
    assert hist[-1] == pytest.approx(line[-1] - signal[-1])


def test_indicators_return_input_length():
    for fn in (lambda v: sma(v, 3), lambda v: ema(v, 3), lambda v: rsi(v, 3)):
        assert fn(RAMP).size == len(RAMP)


def test_last_valid_skips_nan_tail():
    values = np.array([1.0, 2.0, np.nan])
    assert last_valid(values) == pytest.approx(2.0)
    assert last_valid(np.array([np.nan]), default=-1.0) == -1.0


def test_short_series_is_all_nan_not_an_error():
    assert np.isnan(sma([1.0, 2.0], 5)).all()
    assert np.isnan(ema([1.0, 2.0], 5)).all()
    assert np.isnan(rsi([1.0, 2.0], 5)).all()
