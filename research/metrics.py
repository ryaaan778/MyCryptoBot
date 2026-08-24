"""Performance metrics.

Deliberately opinionated in three places, because the defaults elsewhere are
how backtests flatter themselves.

**Compounding, not summing.** Returns compound. The legacy report this system
replaces summed daily percentages, which turns a −20% day followed by a +20% day
into breakeven when it is really −4%. Every return here is computed from the
equity curve.

**Median and IQR across seeds, never the best run.** ``aggregate`` refuses to
report a maximum as a headline. Picking the best of five seeds and calling it
the result is the most common way an RL project fools itself, so the aggregation
API does not offer it as a convenience.

**Worst day is a first-class metric.** In the forensics on the old strategy, a
single day accounted for 285% of the total loss while the other seventeen days
summed positive. An average hides that; the promotion gate needs to see it.

A note on Sharpe: it is reported because it is the lingua franca, but Sortino
leads because upside volatility is not a risk anyone is trying to avoid, and a
trading policy's return distribution is not symmetric. Both are annualised from
per-bar log returns, and both are noise below a few dozen trades — which is why
``trades`` travels alongside them in every ``MetricSet`` rather than being
something you have to remember to look up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger("research.metrics")

METRICS_VERSION = "v1"

#: Profit factor is unbounded when a run has no losing trades. Reporting `inf`
#: poisons every downstream average, so it is capped and the cap is documented.
PROFIT_FACTOR_CAP = 100.0

EPSILON = 1e-12


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL = "SIGNAL"
    END_OF_DATA = "END_OF_DATA"
    RISK = "RISK"


@dataclass(frozen=True)
class TradeRecord:
    """One completed round trip.

    ``fees`` is already subtracted from ``pnl`` — it is carried separately only
    so cost drag can be reported. Double-subtracting it is the classic way to
    make a backtest look worse than the engine that produced it.
    """

    entry_index: int
    exit_index: int
    side: str                 # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float                # net, in quote currency
    fees: float
    reason: str = ExitReason.SIGNAL.value

    @property
    def bars_held(self) -> int:
        return max(0, self.exit_index - self.entry_index)

    @property
    def notional(self) -> float:
        return abs(self.entry_price * self.quantity)

    @property
    def return_pct(self) -> float:
        if self.notional < EPSILON:
            return 0.0
        return self.pnl / self.notional * 100.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True)
class MetricSet:
    """A run's metrics in long format, ready for the ``policy_metrics`` table."""

    values: dict[str, float]
    trades: int
    bars: int
    version: str = METRICS_VERSION

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.values.get(key, default)

    def to_rows(self, **tags) -> list[dict]:
        """Long-format rows. ``tags`` carry dataset/split/regime/seed identity."""
        return [
            {"metric": name, "value": float(value), **tags}
            for name, value in sorted(self.values.items())
        ]

    def describe(self) -> str:
        v = self.values
        return (
            f"ret {v['total_return_pct']:+7.2f}%  ann {v['annualised_return_pct']:+8.2f}%  "
            f"DD {v['max_drawdown_pct']:6.2f}%  Sortino {v['sortino']:6.2f}  "
            f"Sharpe {v['sharpe']:6.2f}  PF {v['profit_factor']:5.2f}  "
            f"win {v['win_rate_pct']:5.1f}%  R:R {v['realised_rr']:5.2f}  "
            f"n={self.trades:>5}  worst day {v['worst_day_pct']:+7.2f}%"
        )


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def equity_returns(equity: np.ndarray) -> np.ndarray:
    """Per-bar log returns of the equity curve.

    Log returns because they add across time, which is what makes the
    annualisation below correct rather than approximately correct.
    """
    e = np.asarray(equity, dtype=np.float64)
    if e.size < 2:
        return np.zeros(0)
    # A blown account (equity <= 0) has no defined log return; clamp so the rest
    # of the metric set still computes and max_drawdown reports the ruin.
    safe = np.maximum(e, EPSILON)
    return np.log(safe[1:] / safe[:-1])


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Deepest peak-to-trough fall, as a positive fraction, plus its bounds."""
    e = np.asarray(equity, dtype=np.float64)
    if e.size == 0:
        return 0.0, 0, 0
    peaks = np.maximum.accumulate(e)
    drawdowns = np.where(peaks > EPSILON, (peaks - e) / peaks, 0.0)
    trough = int(np.argmax(drawdowns))
    depth = float(drawdowns[trough])
    peak = int(np.argmax(e[:trough + 1])) if trough else 0
    return depth, peak, trough


def sharpe(returns: np.ndarray, bars_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd < EPSILON:
        return 0.0
    return float(r.mean() / sd * math.sqrt(bars_per_year))


def sortino(returns: np.ndarray, bars_per_year: float) -> float:
    """Downside deviation over *all* periods, not only the losing ones.

    This is the standard target-downside-deviation definition: the sum of
    squared negative excursions divided by the total period count. The common
    variant that divides by the count of negative periods is strictly harsher
    (a smaller denominator inside the root gives a larger deviation), but it is
    also not comparable between policies — a policy that trades rarely and one
    that trades constantly get denominators built from different sample sizes,
    so their ratios cannot be ranked against each other. Since the whole purpose
    here is ranking challengers against a champion, comparability wins.

    That does make Sortino the more generous of the two, so it is not asked to
    carry tail risk on its own: ``max_drawdown_pct``, ``worst_day_pct`` and the
    per-regime breakdown are the metrics the promotion gate leans on for that.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = float(math.sqrt(float((downside ** 2).sum()) / r.size))
    if dd < EPSILON:
        # No losing bar at all. Real over a short window, meaningless as a ratio.
        return 0.0 if r.mean() <= 0 else PROFIT_FACTOR_CAP
    return float(r.mean() / dd * math.sqrt(bars_per_year))


#: Annualised return is clamped to this multiple of capital per year. A short
#: window extrapolates absurdly — a 2-bar 2% gain annualises to 1.02**52560 —
#: and an `inf` from one run silently poisons every median it is aggregated into.
ANNUALISED_LOG_CAP = math.log(1_000.0)


def annualised_return(equity: np.ndarray, bars: int, bars_per_year: float) -> float:
    """CAGR in percent, computed in log space and clamped.

    The clamp is not cosmetic. Annualising anything shorter than a few weeks is
    extrapolation, not measurement, and left unclamped it overflows to infinity
    on short windows — which then propagates through aggregation as a nan and
    takes an entire experiment's reporting with it. Read this metric only
    alongside the window length it came from.
    """
    e = np.asarray(equity, dtype=np.float64)
    if e.size < 2 or bars <= 0 or e[0] <= EPSILON:
        return 0.0
    if e[-1] <= EPSILON:
        return -100.0        # ruin; a CAGR would be undefined
    years = bars / bars_per_year
    if years < EPSILON:
        return 0.0

    annual_log_growth = math.log(e[-1] / e[0]) / years
    if abs(annual_log_growth) > ANNUALISED_LOG_CAP:
        logger.debug(
            "annualised return clamped: %.1f bars (%.4f years) extrapolates past the cap",
            bars, years,
        )
        annual_log_growth = math.copysign(ANNUALISED_LOG_CAP, annual_log_growth)
    return float(math.expm1(annual_log_growth) * 100.0)


def daily_returns(equity: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calendar-day returns from the equity curve, using the last bar of each day."""
    e = np.asarray(equity, dtype=np.float64)
    t = np.asarray(ts, dtype=np.int64)
    if e.size == 0 or t.size != e.size:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    days = t // 86_400_000
    # Index of the final bar in each day, in chronological order.
    boundaries = np.flatnonzero(np.diff(days)) + 1
    closes = np.concatenate([boundaries - 1, [e.size - 1]]) if boundaries.size else np.array([e.size - 1])
    day_equity = e[closes]
    day_ids = days[closes]
    if day_equity.size < 2:
        return np.zeros(0), day_ids
    prev = np.concatenate([[e[0]], day_equity[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > EPSILON, day_equity / prev - 1.0, 0.0)
    return rets, day_ids


def longest_losing_streak(trades: Sequence[TradeRecord]) -> int:
    longest = current = 0
    for trade in trades:
        current = 0 if trade.is_win else current + 1
        longest = max(longest, current)
    return longest


# --------------------------------------------------------------------------
# The full set
# --------------------------------------------------------------------------


def compute_metrics(
    equity: np.ndarray,
    ts: np.ndarray,
    trades: Sequence[TradeRecord],
    *,
    bars_per_year: float,
    exposure_bars: int | None = None,
    slippage_cost: float = 0.0,
) -> MetricSet:
    """Every metric the promotion gate and the reports need, from one run.

    ``exposure_bars`` is the number of bars holding a position; when omitted it
    is inferred from the trades, which is right for a single-position backtest
    and an undercount for overlapping ones.
    """
    e = np.asarray(equity, dtype=np.float64)
    t = np.asarray(ts, dtype=np.int64)
    if e.size == 0:
        raise ValueError("cannot compute metrics on an empty equity curve")
    if t.size != e.size:
        raise ValueError(f"ts has {t.size} rows, equity has {e.size}")

    bars = e.size
    returns = equity_returns(e)
    depth, peak_i, trough_i = max_drawdown(e)

    wins = [x for x in trades if x.is_win]
    losses = [x for x in trades if not x.is_win]
    gross_win = float(sum(x.pnl for x in wins))
    gross_loss = float(abs(sum(x.pnl for x in losses)))
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    if gross_loss > EPSILON:
        profit_factor = min(gross_win / gross_loss, PROFIT_FACTOR_CAP)
    else:
        profit_factor = PROFIT_FACTOR_CAP if gross_win > 0 else 0.0

    realised_rr = min(avg_win / avg_loss, PROFIT_FACTOR_CAP) if avg_loss > EPSILON else (
        PROFIT_FACTOR_CAP if avg_win > 0 else 0.0
    )

    if exposure_bars is None:
        exposure_bars = int(sum(x.bars_held for x in trades))

    day_rets, _ = daily_returns(e, t)
    total_fees = float(sum(x.fees for x in trades))
    total_notional = float(sum(x.notional for x in trades))
    mean_equity = float(np.mean(e)) if e.size else 0.0
    years = bars / bars_per_year if bars_per_year > 0 else 0.0

    values: dict[str, float] = {
        # returns
        "total_return_pct": float((e[-1] / e[0] - 1.0) * 100.0) if e[0] > EPSILON else 0.0,
        "annualised_return_pct": annualised_return(e, bars, bars_per_year),
        "final_equity": float(e[-1]),
        # risk
        "max_drawdown_pct": depth * 100.0,
        "max_drawdown_bars": float(trough_i - peak_i),
        "volatility_pct": float(returns.std(ddof=1) * math.sqrt(bars_per_year) * 100.0)
                          if returns.size > 1 else 0.0,
        "sharpe": sharpe(returns, bars_per_year),
        "sortino": sortino(returns, bars_per_year),
        "worst_day_pct": float(day_rets.min() * 100.0) if day_rets.size else 0.0,
        "best_day_pct": float(day_rets.max() * 100.0) if day_rets.size else 0.0,
        "worst_bar_pct": float(np.expm1(returns.min()) * 100.0) if returns.size else 0.0,
        # trade quality
        "trades": float(len(trades)),
        "win_rate_pct": float(len(wins) / len(trades) * 100.0) if trades else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "realised_rr": realised_rr,
        "profit_factor": profit_factor,
        "expectancy": float(sum(x.pnl for x in trades) / len(trades)) if trades else 0.0,
        "expectancy_pct": float(np.mean([x.return_pct for x in trades])) if trades else 0.0,
        "longest_losing_streak": float(longest_losing_streak(trades)),
        "avg_bars_held": float(np.mean([x.bars_held for x in trades])) if trades else 0.0,
        # cost and activity
        "total_fees": total_fees,
        "slippage_cost": float(slippage_cost),
        "cost_drag_pct": float((total_fees + slippage_cost) / e[0] * 100.0)
                         if e[0] > EPSILON else 0.0,
        "turnover_annualised": float(total_notional / mean_equity / years)
                               if mean_equity > EPSILON and years > EPSILON else 0.0,
        "exposure_pct": float(min(exposure_bars, bars) / bars * 100.0) if bars else 0.0,
        "trades_per_year": float(len(trades) / years) if years > EPSILON else 0.0,
        # exits — the diagnostic that exposed the unreachable 1:7 take-profit
        **{
            f"exit_{reason.value.lower()}_pct": float(
                sum(1 for x in trades if x.reason == reason.value) / len(trades) * 100.0
            ) if trades else 0.0
            for reason in ExitReason
        },
    }
    return MetricSet(values=values, trades=len(trades), bars=bars)


# --------------------------------------------------------------------------
# Per-regime and cross-seed aggregation
# --------------------------------------------------------------------------


def metrics_by_regime(
    equity: np.ndarray,
    ts: np.ndarray,
    trades: Sequence[TradeRecord],
    regime: np.ndarray,
    *,
    bars_per_year: float,
    min_bars: int = 200,
) -> dict[int, MetricSet]:
    """Metrics restricted to the bars of each regime.

    A regime's equity curve is stitched from its own bars, so the returns are
    the policy's returns *while that regime held* — not a contiguous curve.
    Drawdown within a stitched curve is therefore a within-regime figure and
    should not be compared to the overall drawdown as if it were a subset of it.
    A regime with fewer than ``min_bars`` bars is skipped rather than reported,
    because a Sortino from forty bars is a random number with a decimal point.
    """
    e = np.asarray(equity, dtype=np.float64)
    r = np.asarray(regime)
    if r.size != e.size:
        raise ValueError(f"regime has {r.size} labels, equity has {e.size} bars")

    out: dict[int, MetricSet] = {}
    for code in sorted(set(int(x) for x in np.unique(r))):
        mask = r == code
        count = int(mask.sum())
        if count < min_bars:
            continue
        idx = np.flatnonzero(mask)
        # Rebase so each regime's curve starts at the same notional capital and
        # the returns are comparable across regimes.
        sub_equity = e[idx]
        sub_equity = sub_equity / sub_equity[0] * e[0] if sub_equity[0] > EPSILON else sub_equity
        sub_trades = [x for x in trades if 0 <= x.exit_index < r.size and r[x.exit_index] == code]
        out[code] = compute_metrics(
            sub_equity, np.asarray(ts)[idx], sub_trades, bars_per_year=bars_per_year
        )
    return out


@dataclass(frozen=True)
class Aggregate:
    """Distribution of one metric across seeds or windows."""

    metric: str
    median: float
    iqr: float
    q25: float
    q75: float
    minimum: float
    maximum: float
    n: int

    @property
    def is_stable(self) -> bool:
        """Whether the spread is small relative to the level.

        A median Sortino of 1.4 with an IQR of 2.8 is not a policy that works;
        it is a policy whose result depends on the seed.
        """
        scale = max(abs(self.median), EPSILON)
        return self.iqr <= scale

    def describe(self) -> str:
        return (
            f"{self.metric:<26} median {self.median:>9.3f}  IQR {self.iqr:>8.3f}  "
            f"[{self.minimum:>9.3f}, {self.maximum:>9.3f}]  n={self.n}"
            + ("" if self.is_stable else "   UNSTABLE")
        )


def aggregate(runs: Sequence[MetricSet], metrics: Iterable[str] | None = None) -> dict[str, Aggregate]:
    """Median and IQR across runs.

    There is deliberately no ``best()`` counterpart. Reporting the best of N
    seeds is not a summary of N runs, it is a summary of one run chosen after
    seeing the answers, and offering it as a convenience is how it ends up in a
    report.
    """
    if not runs:
        raise ValueError("nothing to aggregate")
    names = list(metrics) if metrics else sorted(
        set().union(*(set(run.values) for run in runs))
    )
    out: dict[str, Aggregate] = {}
    for name in names:
        series = np.array(
            [run.values[name] for run in runs if name in run.values], dtype=np.float64
        )
        if series.size == 0:
            continue
        q25, median, q75 = (float(x) for x in np.quantile(series, [0.25, 0.5, 0.75]))
        out[name] = Aggregate(
            metric=name, median=median, iqr=q75 - q25, q25=q25, q75=q75,
            minimum=float(series.min()), maximum=float(series.max()), n=int(series.size),
        )
    return out


def summarise(
    aggregates: Mapping[str, Aggregate],
    headline: Sequence[str] = (
        "total_return_pct", "annualised_return_pct", "max_drawdown_pct",
        "sortino", "sharpe", "profit_factor", "win_rate_pct", "realised_rr",
        "trades", "worst_day_pct", "exposure_pct", "turnover_annualised",
    ),
) -> str:
    lines = [a.describe() for name in headline if (a := aggregates.get(name))]
    unstable = [a.metric for a in aggregates.values() if not a.is_stable and a.n > 2]
    if unstable:
        lines.append(
            f"  {len(unstable)} metric(s) vary more than their own level across runs: "
            + ", ".join(sorted(unstable)[:6])
        )
    return "\n".join(lines)
