"""Feature construction — market features only, and strictly causal.

Two rules govern this module.

**Causality.** Every feature at row ``i`` is a function of bars ``0..i`` only.
Each builder below uses trailing windows, and ``tests/test_research_leakage.py``
enforces it mechanically: it mutates all bars after ``i`` and asserts the
feature row at ``i`` is byte-identical. A feature that fails that test is a
backtest that lies.

**Relative, not absolute.** The brief asks for normalised/relative inputs rather
than raw prices, and for good reason: a model fed absolute price learns the
price range of its training window and breaks the moment the market leaves it.
Everything here is a ratio, a return, a z-score, or a bounded oscillator.

Account state (position, unrealised P&L, drawdown, exposure) is deliberately
*not* here — it depends on the policy's own actions, so the RL environment
appends it at runtime in Phase 2.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from backend.strategies.indicators import atr, bollinger, ema, macd, rsi, sma

from .datasets import Dataset

logger = logging.getLogger("research.features")

FEATURE_SET_VERSION = "v1"

Builder = Callable[[Dataset], np.ndarray]
_REGISTRY: dict[str, Builder] = {}


def feature(name: str) -> Callable[[Builder], Builder]:
    def register(fn: Builder) -> Builder:
        _REGISTRY[name] = fn
        return fn
    return register


def _safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Divide, substituting ``fill`` for a zero denominator — but never for NaN.

    The distinction matters more than it looks. A zero denominator (a doji with
    ``high == low``, a flat volume window) is a real bar whose ratio is simply
    undefined, and ``fill`` is the right answer. A NaN input is an indicator that
    has not warmed up yet, and filling it would present a fabricated zero as an
    observation — which then makes :func:`build_features` compute a warm-up of 0
    and hand a policy invented data to learn from.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    unformed = ~np.isfinite(a) | ~np.isfinite(b)
    out = np.where(np.isfinite(out), out, fill)
    return np.where(unformed, np.nan, out)


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing windows as a 2-D view; rows before warm-up are NaN-filled."""
    if values.size < window:
        return np.full((values.size, window), np.nan)
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    pad = np.full((window - 1, window), np.nan)
    return np.vstack([pad, view])


def log_returns(dataset: Dataset) -> np.ndarray:
    """Bar-to-bar log returns. Row 0 is NaN — there is no prior bar to compare."""
    close = dataset.close
    out = np.full(close.size, np.nan)
    out[1:] = np.log(close[1:] / close[:-1])
    return out


# ---- returns over several horizons ---------------------------------------


def _return_over(dataset: Dataset, horizon: int) -> np.ndarray:
    close = dataset.close
    out = np.full(close.size, np.nan)
    if close.size > horizon:
        out[horizon:] = np.log(close[horizon:] / close[:-horizon])
    return out


@feature("ret_1")
def _ret_1(d: Dataset) -> np.ndarray:
    return _return_over(d, 1)


@feature("ret_5")
def _ret_5(d: Dataset) -> np.ndarray:
    return _return_over(d, 5)


@feature("ret_15")
def _ret_15(d: Dataset) -> np.ndarray:
    return _return_over(d, 15)


@feature("ret_60")
def _ret_60(d: Dataset) -> np.ndarray:
    return _return_over(d, 60)


# ---- volatility -----------------------------------------------------------


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation; NaN wherever the window is not yet full."""
    # All-NaN warm-up rows make nanstd complain; the NaN it returns is what we want.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(_rolling(values, window), axis=1)


@feature("vol_20")
def _vol_20(d: Dataset) -> np.ndarray:
    return _rolling_std(log_returns(d), 20)


@feature("vol_60")
def _vol_60(d: Dataset) -> np.ndarray:
    return _rolling_std(log_returns(d), 60)


@feature("vol_ratio")
def _vol_ratio(d: Dataset) -> np.ndarray:
    """Short vol over long vol — volatility expansion vs contraction."""
    return _safe_divide(_vol_20(d), _vol_60(d), fill=1.0) - 1.0


@feature("atr_pct")
def _atr_pct(d: Dataset) -> np.ndarray:
    return _safe_divide(atr(d.high, d.low, d.close, 14), d.close)


# ---- oscillators ----------------------------------------------------------


@feature("rsi_14")
def _rsi_14(d: Dataset) -> np.ndarray:
    # Rescaled to [-1, 1] so it sits on the same footing as the other inputs.
    return rsi(d.close, 14) / 50.0 - 1.0


@feature("rsi_7")
def _rsi_7(d: Dataset) -> np.ndarray:
    return rsi(d.close, 7) / 50.0 - 1.0


@feature("macd_hist")
def _macd_hist(d: Dataset) -> np.ndarray:
    _, _, hist = macd(d.close, 12, 26, 9)
    return _safe_divide(hist, d.close) * 100.0


# ---- bands and trend ------------------------------------------------------


@feature("bb_position")
def _bb_position(d: Dataset) -> np.ndarray:
    """Where price sits inside the bands: 0 at the mean, ±1 at the edges."""
    upper, middle, lower = bollinger(d.close, 20, 2.0)
    half_width = (upper - lower) / 2.0
    return _safe_divide(d.close - middle, half_width)


@feature("bb_width")
def _bb_width(d: Dataset) -> np.ndarray:
    upper, middle, lower = bollinger(d.close, 20, 2.0)
    return _safe_divide(upper - lower, middle)


@feature("ema_spread")
def _ema_spread(d: Dataset) -> np.ndarray:
    return _safe_divide(ema(d.close, 12) - ema(d.close, 26), d.close) * 100.0


@feature("trend_slope")
def _trend_slope(d: Dataset) -> np.ndarray:
    """Slope of a trailing least-squares fit on log price, per bar."""
    window = 30
    log_price = np.log(d.close)
    windows = _rolling(log_price, window)
    x = np.arange(window, dtype=np.float64)
    x_centred = x - x.mean()
    denominator = float((x_centred ** 2).sum())
    means = windows.mean(axis=1, keepdims=True)
    return ((windows - means) @ x_centred) / denominator * 1000.0


@feature("dist_from_sma")
def _dist_from_sma(d: Dataset) -> np.ndarray:
    return _safe_divide(d.close - sma(d.close, 50), d.close) * 100.0


# ---- volume and candle shape ---------------------------------------------


@feature("volume_ratio")
def _volume_ratio(d: Dataset) -> np.ndarray:
    return _safe_divide(d.volume, sma(d.volume, 20), fill=1.0) - 1.0


@feature("body_ratio")
def _body_ratio(d: Dataset) -> np.ndarray:
    """Signed body size relative to the bar's range — conviction of the bar."""
    return _safe_divide(d.close - d.open, d.high - d.low)


@feature("upper_wick")
def _upper_wick(d: Dataset) -> np.ndarray:
    return _safe_divide(d.high - np.maximum(d.open, d.close), d.high - d.low)


@feature("lower_wick")
def _lower_wick(d: Dataset) -> np.ndarray:
    return _safe_divide(np.minimum(d.open, d.close) - d.low, d.high - d.low)


# ---- time -----------------------------------------------------------------
# Encoded as sine/cosine pairs so midnight is adjacent to 23:00 rather than
# maximally distant, which a raw hour index would imply.


@feature("hour_sin")
def _hour_sin(d: Dataset) -> np.ndarray:
    hours = (d.ts / 3_600_000) % 24
    return np.sin(2 * np.pi * hours / 24)


@feature("hour_cos")
def _hour_cos(d: Dataset) -> np.ndarray:
    hours = (d.ts / 3_600_000) % 24
    return np.cos(2 * np.pi * hours / 24)


@feature("dow_sin")
def _dow_sin(d: Dataset) -> np.ndarray:
    days = (d.ts / 86_400_000 + 4) % 7   # epoch day 0 was a Thursday
    return np.sin(2 * np.pi * days / 7)


@feature("dow_cos")
def _dow_cos(d: Dataset) -> np.ndarray:
    days = (d.ts / 86_400_000 + 4) % 7
    return np.cos(2 * np.pi * days / 7)


DEFAULT_FEATURES: tuple[str, ...] = tuple(_REGISTRY)


def available_features() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Matrix + scaling
# --------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    values: np.ndarray          # (rows, n_features)
    names: tuple[str, ...]
    warmup: int                 # first row where every feature is finite
    version: str = FEATURE_SET_VERSION

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def row(self, index: int) -> np.ndarray:
        return self.values[index]

    def usable_range(self) -> tuple[int, int]:
        return self.warmup, len(self)


def build_features(
    dataset: Dataset, names: Sequence[str] | None = None
) -> FeatureMatrix:
    selected = tuple(names) if names else DEFAULT_FEATURES
    unknown = [n for n in selected if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"unknown features {unknown}; available: {available_features()}")

    columns = [np.asarray(_REGISTRY[name](dataset), dtype=np.float64) for name in selected]
    for name, column in zip(selected, columns):
        if column.shape != dataset.close.shape:
            raise ValueError(
                f"feature {name!r} returned {column.shape}, expected {dataset.close.shape}"
            )

    values = np.column_stack(columns)
    finite_rows = np.all(np.isfinite(values), axis=1)
    warmup = int(np.argmax(finite_rows)) if finite_rows.any() else len(dataset)
    if not finite_rows.any():
        logger.warning("no fully-formed feature rows for %s", dataset.manifest.dataset_id)

    return FeatureMatrix(values=values, names=selected, warmup=warmup)


@dataclass
class Scaler:
    """Z-score scaler fitted on an explicit set of rows.

    ``fit`` demands the row indices rather than defaulting to "all rows",
    because fitting normalisation statistics over the whole dataset is one of
    the quietest ways to leak test-period information into training. The
    awkwardness is the point.
    """

    mean: np.ndarray
    std: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def fit(cls, matrix: FeatureMatrix, rows: np.ndarray | Sequence[int]) -> "Scaler":
        idx = np.asarray(rows, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("cannot fit a scaler on zero rows")
        subset = matrix.values[idx]
        subset = subset[np.all(np.isfinite(subset), axis=1)]
        if subset.size == 0:
            raise ValueError("no finite rows in the fitting range")
        mean = subset.mean(axis=0)
        std = subset.std(axis=0)
        std[std < 1e-12] = 1.0     # constant column -> leave it centred at zero
        return cls(mean=mean, std=std, names=matrix.names)

    def transform(self, matrix: FeatureMatrix) -> FeatureMatrix:
        if matrix.names != self.names:
            raise ValueError("feature names differ from the ones this scaler was fitted on")
        scaled = (matrix.values - self.mean) / self.std
        return FeatureMatrix(
            values=scaled, names=matrix.names, warmup=matrix.warmup, version=matrix.version
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Scaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            names=tuple(payload["names"]),
        )
