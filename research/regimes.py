"""Causal regime labelling.

Per-regime metrics are one of the most useful things the validation harness
produces — a policy that makes all its money in one market state and gives it
back in another is a policy you have not actually validated, and the promotion
gate vetoes exactly that. But regime labels are also one of the easiest places
to leak the future without noticing, in two distinct ways:

1. **The statistic.** Labelling bar ``i`` from a window centred on ``i``, or from
   a smoothed series that peeks forward, tells you today's regime using
   tomorrow's bars. Avoided here by deriving both axes from
   :mod:`research.features`, which is mechanically tested for causality.
2. **The thresholds.** Subtler and more common: computing "high volatility" as
   the top tercile *of the whole dataset* means every label knows the full
   distribution of volatility, including the test period's. A backtest that
   labels 2021 using 2024's volatility range is quietly cheating.

So thresholds here are estimated from a **trailing calibration window** only,
refreshed periodically, and bars before the first calibration are ``UNKNOWN``
rather than guessed. That leaves a genuinely unlabelled head on every dataset,
which is correct: at bar 500 you really do not know whether this is a
high-volatility market by historical standards.

Two axes are labelled — realised volatility and trend slope — and a compact
four-way composite is derived from them for reporting. Both raw axes stay
available, because collapsing to four buckets loses information that
regime-conditional analysis sometimes wants back.

**Hysteresis, not smoothing.** Raw thresholding flickers: a statistic sitting on
a boundary flips label every other bar and produces hundreds of one-bar
"regimes" that mean nothing. The usual fix — requiring a label to persist for N
bars before accepting it — needs N future bars and is therefore a leak. Instead
each bucket has a wider entry threshold than exit threshold, so a regime is
harder to enter than to leave behind. That is stateful but strictly backward
-looking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .datasets import Dataset
from .features import build_features

logger = logging.getLogger("research.regimes")

REGIME_SET_VERSION = "v1"


class Regime(IntEnum):
    """Composite market state. ``UNKNOWN`` is a real answer, not a failure."""

    UNKNOWN = 0
    QUIET_RANGE = 1      # no directional slope, subdued volatility
    TRENDING_UP = 2
    TRENDING_DOWN = 3
    CHOPPY = 4           # no directional slope, elevated volatility

    @property
    def label(self) -> str:
        return self.name.lower()


class VolBucket(IntEnum):
    UNKNOWN = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3


class TrendBucket(IntEnum):
    UNKNOWN = 0
    DOWN = 1
    FLAT = 2
    UP = 3


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds and window sizes.

    ``calibration_window`` is trailing rather than expanding on purpose. Regime
    is a relative judgement — "volatile compared to what?" — and comparing
    against the last few weeks is both more useful and more honest than
    comparing against a decade the market no longer resembles. It also keeps
    labelling O(n) instead of O(n²).
    """

    vol_feature: str = "vol_60"
    trend_feature: str = "trend_slope"
    calibration_window: int = 2_000       # ~7 days of 5m bars
    recalibrate_every: int = 200
    min_calibration: int = 500            # below this, refuse to label

    # Entry quantiles: wide, so a bucket must be clearly earned.
    vol_low_enter: float = 0.30
    vol_high_enter: float = 0.70
    trend_down_enter: float = 0.25
    trend_up_enter: float = 0.75

    # Exit quantiles: narrower, so an established bucket persists through noise.
    vol_low_exit: float = 0.42
    vol_high_exit: float = 0.58
    trend_down_exit: float = 0.40
    trend_up_exit: float = 0.60

    def validate(self) -> None:
        if not 0.0 < self.vol_low_enter < self.vol_low_exit < self.vol_high_exit < self.vol_high_enter < 1.0:
            raise ValueError("vol quantiles must satisfy low_enter < low_exit < high_exit < high_enter")
        if not 0.0 < self.trend_down_enter < self.trend_down_exit < self.trend_up_exit < self.trend_up_enter < 1.0:
            raise ValueError("trend quantiles must satisfy down_enter < down_exit < up_exit < up_enter")
        if self.min_calibration < 50:
            raise ValueError("min_calibration below 50 bars gives meaningless quantiles")
        if self.calibration_window < self.min_calibration:
            raise ValueError("calibration_window must be at least min_calibration")
        if self.recalibrate_every < 1:
            raise ValueError("recalibrate_every must be positive")


@dataclass
class RegimeLabels:
    """Per-bar labels, aligned one-to-one with the dataset's rows."""

    regime: np.ndarray          # int8, Regime
    vol_bucket: np.ndarray      # int8, VolBucket
    trend_bucket: np.ndarray    # int8, TrendBucket
    warmup: int                 # first bar carrying a real label
    config: RegimeConfig
    version: str = REGIME_SET_VERSION

    def __len__(self) -> int:
        return int(self.regime.size)

    def mask(self, regime: Regime) -> np.ndarray:
        return self.regime == int(regime)

    def counts(self) -> dict[str, int]:
        return {
            r.label: int(np.count_nonzero(self.regime == int(r)))
            for r in Regime
            if np.count_nonzero(self.regime == int(r))
        }

    def shares(self) -> dict[str, float]:
        labelled = int(np.count_nonzero(self.regime != int(Regime.UNKNOWN)))
        if labelled == 0:
            return {}
        return {
            r.label: int(np.count_nonzero(self.regime == int(r))) / labelled
            for r in Regime
            if r is not Regime.UNKNOWN and np.count_nonzero(self.regime == int(r))
        }

    def segments(self) -> list[tuple[int, int, Regime]]:
        """Contiguous runs as ``(start, stop_exclusive, regime)``."""
        if len(self) == 0:
            return []
        change = np.flatnonzero(np.diff(self.regime)) + 1
        bounds = np.concatenate([[0], change, [len(self)]])
        return [
            (int(bounds[k]), int(bounds[k + 1]), Regime(int(self.regime[bounds[k]])))
            for k in range(bounds.size - 1)
        ]

    def summary(self) -> str:
        segs = [s for s in self.segments() if s[2] is not Regime.UNKNOWN]
        lengths = [stop - start for start, stop, _ in segs]
        median_len = int(np.median(lengths)) if lengths else 0
        parts = [f"{k}={v} ({self.shares().get(k, 0):.0%})" for k, v in self.counts().items()]
        return (
            f"regimes {self.version}: " + ", ".join(parts)
            + f" | {len(segs)} segments, median {median_len} bars, warmup {self.warmup}"
        )


def _quantiles(sample: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    return np.quantile(sample, qs)


def _bucket_with_hysteresis(
    values: np.ndarray,
    *,
    low_enter_q: float,
    low_exit_q: float,
    high_exit_q: float,
    high_enter_q: float,
    config: RegimeConfig,
    low_code: int,
    mid_code: int,
    high_code: int,
) -> tuple[np.ndarray, int]:
    """Three-way bucketing against trailing quantiles, with entry/exit deadbands.

    Returns the code array and the first bar that could be labelled. Every
    threshold used at bar ``i`` is computed from bars strictly before ``i``.
    """
    n = values.size
    out = np.full(n, 0, dtype=np.int8)
    if n == 0:
        return out, 0

    finite = np.isfinite(values)
    # The calibration sample only ever contains formed values, so the warm-up
    # NaNs of the underlying feature cannot skew a threshold.
    formed = np.flatnonzero(finite)
    if formed.size == 0:
        logger.warning("no formed values for regime axis; leaving everything UNKNOWN")
        return out, n

    first_formed = int(formed[0])
    first_labelled = first_formed + config.min_calibration
    if first_labelled >= n:
        logger.warning(
            "dataset too short to calibrate regimes (%d bars, need %d)",
            n, first_labelled + 1,
        )
        return out, n

    thresholds: np.ndarray | None = None
    next_recalibration = first_labelled
    state = mid_code

    for i in range(first_labelled, n):
        if i >= next_recalibration:
            window_start = max(first_formed, i - config.calibration_window)
            sample = values[window_start:i]
            sample = sample[np.isfinite(sample)]
            if sample.size >= config.min_calibration:
                thresholds = _quantiles(
                    sample, (low_enter_q, low_exit_q, high_exit_q, high_enter_q)
                )
                next_recalibration = i + config.recalibrate_every
            else:
                # Not enough history yet; try again next bar rather than
                # calibrating on a sample too small to mean anything.
                next_recalibration = i + 1

        value = values[i]
        if thresholds is None or not np.isfinite(value):
            out[i] = 0
            continue

        low_enter, low_exit, high_exit, high_enter = thresholds
        if state == high_code:
            state = high_code if value >= high_exit else (
                low_code if value <= low_enter else mid_code
            )
        elif state == low_code:
            state = low_code if value <= low_exit else (
                high_code if value >= high_enter else mid_code
            )
        else:
            state = (
                high_code if value >= high_enter
                else low_code if value <= low_enter
                else mid_code
            )
        out[i] = state

    return out, first_labelled


def label_regimes(
    dataset: Dataset, config: RegimeConfig | None = None
) -> RegimeLabels:
    """Label every bar, using only information available at that bar."""
    cfg = config or RegimeConfig()
    cfg.validate()

    matrix = build_features(dataset, [cfg.vol_feature, cfg.trend_feature])
    vol = matrix.values[:, 0]
    trend = matrix.values[:, 1]

    vol_bucket, vol_start = _bucket_with_hysteresis(
        vol,
        low_enter_q=cfg.vol_low_enter, low_exit_q=cfg.vol_low_exit,
        high_exit_q=cfg.vol_high_exit, high_enter_q=cfg.vol_high_enter,
        config=cfg,
        low_code=int(VolBucket.LOW), mid_code=int(VolBucket.NORMAL),
        high_code=int(VolBucket.HIGH),
    )
    trend_bucket, trend_start = _bucket_with_hysteresis(
        trend,
        low_enter_q=cfg.trend_down_enter, low_exit_q=cfg.trend_down_exit,
        high_exit_q=cfg.trend_up_exit, high_enter_q=cfg.trend_up_enter,
        config=cfg,
        low_code=int(TrendBucket.DOWN), mid_code=int(TrendBucket.FLAT),
        high_code=int(TrendBucket.UP),
    )

    regime = np.full(len(dataset), int(Regime.UNKNOWN), dtype=np.int8)
    known = (vol_bucket != int(VolBucket.UNKNOWN)) & (trend_bucket != int(TrendBucket.UNKNOWN))

    up = known & (trend_bucket == int(TrendBucket.UP))
    down = known & (trend_bucket == int(TrendBucket.DOWN))
    flat = known & (trend_bucket == int(TrendBucket.FLAT))

    regime[up] = int(Regime.TRENDING_UP)
    regime[down] = int(Regime.TRENDING_DOWN)
    regime[flat & (vol_bucket == int(VolBucket.HIGH))] = int(Regime.CHOPPY)
    regime[flat & (vol_bucket != int(VolBucket.HIGH))] = int(Regime.QUIET_RANGE)

    warmup = max(vol_start, trend_start)
    if warmup >= len(dataset):
        logger.warning(
            "dataset %s produced no regime labels", dataset.manifest.dataset_id
        )

    return RegimeLabels(
        regime=regime,
        vol_bucket=vol_bucket,
        trend_bucket=trend_bucket,
        warmup=int(warmup),
        config=cfg,
    )


def regime_of(labels: RegimeLabels, index: int) -> Regime:
    return Regime(int(labels.regime[index]))
