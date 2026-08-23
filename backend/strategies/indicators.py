"""Technical indicators in plain numpy.

Deliberately no TA-Lib: it needs a C toolchain and would make the project
painful to install. These follow the standard definitions — Wilder's smoothing
for RSI and ATR, SMA-seeded EMA — so the numbers agree with what a charting
package shows.

Every function returns an array the same length as its input, with ``nan`` in
the warm-up region so callers can't accidentally read an unformed value.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_array(values: Array | list[float]) -> Array:
    return np.asarray(values, dtype=np.float64)


def sma(values: Array | list[float], period: int) -> Array:
    v = _as_array(values)
    out = np.full(v.size, np.nan)
    if period <= 0 or v.size < period:
        return out
    cumsum = np.cumsum(np.insert(v, 0, 0.0))
    out[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def ema(values: Array | list[float], period: int) -> Array:
    """Exponential MA seeded with the SMA of the first ``period`` samples."""
    v = _as_array(values)
    out = np.full(v.size, np.nan)
    if period <= 0 or v.size < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = v[:period].mean()
    for i in range(period, v.size):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_smooth(values: Array | list[float], period: int) -> Array:
    """Wilder's smoothing (an EMA with alpha = 1/period), SMA-seeded."""
    v = _as_array(values)
    out = np.full(v.size, np.nan)
    if period <= 0 or v.size < period:
        return out
    out[period - 1] = v[:period].mean()
    for i in range(period, v.size):
        out[i] = (out[i - 1] * (period - 1) + v[i]) / period
    return out


def rsi(closes: Array | list[float], period: int = 14) -> Array:
    c = _as_array(closes)
    out = np.full(c.size, np.nan)
    if c.size <= period:
        return out
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss)
        values = 100.0 - (100.0 / (1.0 + rs))
    # A zero average loss means an unbroken run of gains: RSI is 100 by definition.
    values = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, values)
    values = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, values)
    out[1:] = values
    return out


def true_range(highs: Array, lows: Array, closes: Array) -> Array:
    h, l, c = _as_array(highs), _as_array(lows), _as_array(closes)
    tr = np.full(h.size, np.nan)
    if h.size == 0:
        return tr
    tr[0] = h[0] - l[0]
    prev_close = c[:-1]
    tr[1:] = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - prev_close),
        np.abs(l[1:] - prev_close),
    ])
    return tr


def atr(highs: Array, lows: Array, closes: Array, period: int = 14) -> Array:
    return wilder_smooth(true_range(highs, lows, closes), period)


def bollinger(
    closes: Array | list[float], period: int = 20, num_std: float = 2.0
) -> tuple[Array, Array, Array]:
    """Returns ``(upper, middle, lower)``. Population std, as is conventional."""
    c = _as_array(closes)
    middle = sma(c, period)
    upper = np.full(c.size, np.nan)
    lower = np.full(c.size, np.nan)
    if c.size < period or period <= 0:
        return upper, middle, lower
    # Rolling std via stride tricks — O(n) windows without a Python loop.
    windows = np.lib.stride_tricks.sliding_window_view(c, period)
    std = windows.std(axis=1)
    upper[period - 1:] = middle[period - 1:] + num_std * std
    lower[period - 1:] = middle[period - 1:] - num_std * std
    return upper, middle, lower


def macd(
    closes: Array | list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Array, Array, Array]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    c = _as_array(closes)
    fast_ema = ema(c, fast)
    slow_ema = ema(c, slow)
    macd_line = fast_ema - slow_ema
    valid = ~np.isnan(macd_line)
    signal_line = np.full(c.size, np.nan)
    if valid.any():
        start = int(np.argmax(valid))
        sig = ema(macd_line[start:], signal)
        signal_line[start:] = sig
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bandwidth(upper: Array, middle: Array, lower: Array) -> Array:
    """Normalised band width — the standard squeeze detector."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(upper - lower, middle)


def last_valid(values: Array, default: float = float("nan")) -> float:
    """Most recent non-nan value, or ``default`` if the series never formed."""
    v = _as_array(values)
    mask = ~np.isnan(v)
    if not mask.any():
        return default
    return float(v[mask][-1])
