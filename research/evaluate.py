"""The promotion gate.

A challenger becomes a champion only if **every** condition below holds. They
are deliberately conjunctive rather than a weighted score: a score lets a
spectacular return buy its way past a catastrophic drawdown, and the whole
purpose of this gate is that it cannot be bought.

| condition | why it is there |
|---|---|
| Sortino beats the champion | otherwise there is no reason to switch |
| Sortino beats the best baseline | "made money" is not a result if buy-and-hold made more |
| max drawdown ≤ champion × 1.1 | a challenger may not buy return with risk |
| ≥ 30 validation trades | a Sortino from four trades is a random number |
| positive median across ≥ 5 seeds | one lucky seed is not a policy |
| no regime with catastrophic loss | an edge that only exists in one market state is not an edge |
| ≥ 2 of the 3 most recent windows at or above baseline | a decayed edge should not promote on old windows |

Reported as a full breakdown rather than a boolean, because "rejected" is not
useful feedback to a research agent — "rejected: 22 trades, needed 30" is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dataclasses import replace as dc_replace

from .backtest import BacktestResult
from .metrics import MetricSet, aggregate

logger = logging.getLogger("research.evaluate")

GATE_VERSION = "v1"


@dataclass(frozen=True)
class GateConfig:
    min_trades: int = 30
    min_seeds: int = 5
    max_drawdown_multiple: float = 1.1
    regime_drawdown_multiple: float = 2.0
    recent_windows: int = 3
    recent_windows_required: int = 2
    primary_metric: str = "sortino"
    version: str = GATE_VERSION

    def validate(self) -> None:
        if self.min_trades < 1:
            raise ValueError("min_trades must be at least 1")
        if self.min_seeds < 1:
            raise ValueError("min_seeds must be at least 1")
        if self.max_drawdown_multiple <= 0 or self.regime_drawdown_multiple <= 0:
            raise ValueError("drawdown multiples must be positive")
        if self.recent_windows_required > self.recent_windows:
            raise ValueError("cannot require more recent windows than are examined")


@dataclass(frozen=True)
class Condition:
    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    detail: str

    def describe(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name:<34} {self.detail}"


@dataclass
class GateResult:
    policy_id: str
    passed: bool
    conditions: list[Condition] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    version: str = GATE_VERSION

    @property
    def failures(self) -> list[Condition]:
        return [c for c in self.conditions if not c.passed]

    def reason(self) -> str:
        if self.passed:
            return "all promotion conditions met"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "version": self.version,
            "conditions": {
                c.name: {"passed": c.passed, "actual": c.actual,
                         "threshold": c.threshold, "detail": c.detail}
                for c in self.conditions
            },
            "summary": self.summary,
        }

    def describe(self) -> str:
        head = f"{self.policy_id}: {'PROMOTE' if self.passed else 'REJECT'}"
        return "\n".join([head] + [c.describe() for c in self.conditions])


def _median(results: Sequence[BacktestResult], metric: str) -> float:
    values = [r.metrics.get(metric) for r in results]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def evaluate_gate(
    policy_id: str,
    challenger: Sequence[BacktestResult],
    *,
    baselines: Mapping[str, Sequence[BacktestResult]],
    champion: Sequence[BacktestResult] | None = None,
    regime_metrics: Mapping[str, MetricSet] | None = None,
    overall_drawdown_pct: float | None = None,
    config: GateConfig | None = None,
) -> GateResult:
    """Run every condition and report all of them, passed or failed."""
    cfg = config or GateConfig()
    cfg.validate()
    if not challenger:
        raise ValueError("no challenger results to evaluate")

    metric = cfg.primary_metric
    challenger_score = _median(challenger, metric)
    challenger_dd = _median(challenger, "max_drawdown_pct")
    total_trades = int(sum(r.metrics["trades"] for r in challenger))
    seeds = {r.seed for r in challenger if r.seed is not None}
    conditions: list[Condition] = []

    # --- beats the best baseline ---
    baseline_scores = {name: _median(runs, metric) for name, runs in baselines.items()}
    if baseline_scores:
        best_name = max(baseline_scores, key=lambda k: baseline_scores[k])
        best_score = baseline_scores[best_name]
    else:
        best_name, best_score = "none", float("-inf")
    conditions.append(Condition(
        name="beats_best_baseline",
        passed=bool(challenger_score > best_score),
        actual=challenger_score, threshold=best_score,
        detail=(f"{metric} {challenger_score:.3f} vs best baseline "
                f"{best_name} {best_score:.3f}"),
    ))

    # --- beats the incumbent ---
    if champion:
        champion_score = _median(champion, metric)
        champion_dd = _median(champion, "max_drawdown_pct")
        conditions.append(Condition(
            name="beats_champion",
            passed=bool(challenger_score > champion_score),
            actual=challenger_score, threshold=champion_score,
            detail=f"{metric} {challenger_score:.3f} vs champion {champion_score:.3f}",
        ))
        allowed_dd = champion_dd * cfg.max_drawdown_multiple
        conditions.append(Condition(
            name="drawdown_within_champion_tolerance",
            passed=bool(challenger_dd <= allowed_dd),
            actual=challenger_dd, threshold=allowed_dd,
            detail=(f"max DD {challenger_dd:.2f}% vs allowed {allowed_dd:.2f}% "
                    f"(champion {champion_dd:.2f}% x {cfg.max_drawdown_multiple})"),
        ))
    else:
        # No incumbent is not a free pass: the baselines still have to be beaten.
        conditions.append(Condition(
            name="beats_champion", passed=True, actual=challenger_score, threshold=None,
            detail="no incumbent champion; baseline condition governs",
        ))

    # --- enough trades to mean anything ---
    conditions.append(Condition(
        name="minimum_trades",
        passed=total_trades >= cfg.min_trades,
        actual=float(total_trades), threshold=float(cfg.min_trades),
        detail=f"{total_trades} trades, need {cfg.min_trades}",
    ))

    # --- enough seeds, and a positive median across them ---
    conditions.append(Condition(
        name="minimum_seeds",
        passed=len(seeds) >= cfg.min_seeds,
        actual=float(len(seeds)), threshold=float(cfg.min_seeds),
        detail=f"{len(seeds)} distinct seed(s), need {cfg.min_seeds}",
    ))
    median_return = _median(challenger, "total_return_pct")
    conditions.append(Condition(
        name="positive_median_return",
        passed=bool(median_return > 0),
        actual=median_return, threshold=0.0,
        detail=f"median return {median_return:+.2f}% across {len(challenger)} runs",
    ))

    # --- no regime it falls apart in ---
    if regime_metrics:
        worst_regime, worst_dd = max(
            ((name, m.get("max_drawdown_pct", 0.0)) for name, m in regime_metrics.items()),
            key=lambda pair: pair[1],
        )
        # The basis has to be measured the same way the regime figure was.
        # Per-regime drawdown comes from a curve stitched across every window and
        # compounded; `challenger_dd` is the median drawdown *within* one window
        # and never sees that compounding. Comparing the two directly makes the
        # threshold far too tight — it would reject a policy for a drawdown the
        # basis is structurally incapable of reaching. The caller passes the
        # stitched overall drawdown so both sides are on the same footing.
        basis = overall_drawdown_pct if overall_drawdown_pct is not None else challenger_dd
        allowed = basis * cfg.regime_drawdown_multiple
        conditions.append(Condition(
            name="no_catastrophic_regime",
            passed=bool(worst_dd <= allowed),
            actual=worst_dd, threshold=allowed,
            detail=(f"worst regime {worst_regime} DD {worst_dd:.2f}% vs allowed "
                    f"{allowed:.2f}% (overall {basis:.2f}% x "
                    f"{cfg.regime_drawdown_multiple})"),
        ))
    else:
        conditions.append(Condition(
            name="no_catastrophic_regime", passed=False, actual=None, threshold=None,
            detail="no per-regime metrics supplied — cannot verify, so cannot promote",
        ))

    # --- still working recently, not just on average ---
    windows = sorted({r.window for r in challenger if r.window is not None})
    if windows:
        recent = windows[-cfg.recent_windows:]
        clear = 0
        for window in recent:
            challenger_window = _median(
                [r for r in challenger if r.window == window], metric)
            baseline_window = max(
                (_median([r for r in runs if r.window == window], metric)
                 for runs in baselines.values()), default=float("-inf"))
            if np.isfinite(challenger_window) and challenger_window >= baseline_window:
                clear += 1
        conditions.append(Condition(
            name="recent_windows_hold_up",
            passed=clear >= cfg.recent_windows_required,
            actual=float(clear), threshold=float(cfg.recent_windows_required),
            detail=(f"{clear} of the last {len(recent)} window(s) at or above baseline, "
                    f"need {cfg.recent_windows_required}"),
        ))
    else:
        conditions.append(Condition(
            name="recent_windows_hold_up", passed=False, actual=None, threshold=None,
            detail="results carry no window index — walk-forward evaluation is required",
        ))

    summary = {
        f"median_{metric}": challenger_score,
        "median_total_return_pct": median_return,
        "median_max_drawdown_pct": challenger_dd,
        "trades": float(total_trades),
        "seeds": float(len(seeds)),
        "runs": float(len(challenger)),
    }
    result = GateResult(policy_id=policy_id,
                        passed=all(c.passed for c in conditions),
                        conditions=conditions, summary=summary)
    logger.info("gate %s: %s", policy_id, "PROMOTE" if result.passed else result.reason())
    return result


def stitch_windows(results: Sequence[BacktestResult], labels):
    """Chain per-window results into one continuous equity curve.

    Each walk-forward window restarts the account at the same capital, so simply
    concatenating the curves injects a fabricated jump at every boundary — a
    window that ended at 9,200 followed by one that starts at 10,000 reads as an
    +8.7% bar that nobody traded. Compounding each window's *relative* returns
    onto a running curve removes those artefacts, which matters here because the
    per-regime figures are computed from exactly these returns.
    """
    equity_parts, ts_parts, regime_parts, trades = [], [], [], []
    level = None
    cursor = 0
    for result in results:
        relative = result.equity / result.equity[0]
        level = relative if level is None else relative * level[-1]
        equity_parts.append(level)
        ts_parts.append(result.ts)
        regime_parts.append(labels.regime[result.start_index:result.stop_index])
        # Trade indices are absolute dataset positions; the stitched arrays are
        # not. Left unmapped, every trade either falls outside the array and is
        # dropped, or lands on some unrelated bar and is attributed to the wrong
        # regime — silently, because neither is an error.
        shift = cursor - result.start_index
        trades.extend(
            dc_replace(trade,
                       entry_index=trade.entry_index + shift,
                       exit_index=trade.exit_index + shift)
            for trade in result.trades
        )
        cursor += result.equity.size
    scale = results[0].equity[0]
    return (np.concatenate(equity_parts) * scale, np.concatenate(ts_parts),
            np.concatenate(regime_parts), trades)


def regime_metrics_across_seeds(
    results: Sequence[BacktestResult], labels, *, bars_per_year: float,
    min_bars: int = 200, regime_name=None,
) -> tuple[dict[str, MetricSet], float]:
    """Per-regime metrics pooled across seeds, keeping the **worst** drawdown.

    Reporting one arbitrary seed is how the catastrophic-regime veto gets
    fooled: if that seed happened to learn to sit out — a genuine local optimum
    under this reward, and one that actually occurred — every regime reports a
    0.00% drawdown and the veto passes without checking anything. Taking the
    worst drawdown any seed suffered in a regime matches what the veto is for,
    which is finding the market state where the policy falls apart.

    Each seed is stitched separately, because seeds are parallel runs of the
    same policy rather than consecutive periods; chaining them would invent a
    continuity that does not exist.

    Returns the pooled per-regime metrics and the worst *overall* drawdown on
    the same stitched curves — the gate needs both measured the same way, or the
    regime threshold is derived from a number that cannot reach it.
    """
    from .metrics import compute_metrics, max_drawdown, metrics_by_regime

    per_seed: dict[str, list[MetricSet]] = {}
    worst_overall = 0.0
    for seed in sorted({r.seed for r in results if r.seed is not None}):
        windows = sorted((r for r in results if r.seed == seed),
                         key=lambda r: r.start_index)
        if not windows:
            continue
        equity, ts, regime, trades = stitch_windows(windows, labels)
        worst_overall = max(worst_overall, float(max_drawdown(equity)[0] * 100.0))
        for code, metrics in metrics_by_regime(
            equity, ts, trades, regime, bars_per_year=bars_per_year, min_bars=min_bars
        ).items():
            name = regime_name(code) if regime_name else str(code)
            per_seed.setdefault(name, []).append(metrics)

    pooled: dict[str, MetricSet] = {}
    for name, runs in per_seed.items():
        values = {
            "max_drawdown_pct": max(m.get("max_drawdown_pct", 0.0) for m in runs),
            "total_return_pct": float(np.median(
                [m.get("total_return_pct", 0.0) for m in runs])),
            "sortino": float(np.median([m.get("sortino", 0.0) for m in runs])),
            "win_rate_pct": float(np.median([m.get("win_rate_pct", 0.0) for m in runs])),
            "trades": float(sum(m.get("trades", 0.0) for m in runs)),
            "seeds": float(len(runs)),
        }
        pooled[name] = MetricSet(values=values,
                                 trades=int(values["trades"]),
                                 bars=max(m.bars for m in runs))
    return pooled, worst_overall


def compare_to_baselines(
    challenger: Sequence[BacktestResult],
    baselines: Mapping[str, Sequence[BacktestResult]],
    metrics: Sequence[str] = ("total_return_pct", "sortino", "max_drawdown_pct",
                              "win_rate_pct", "trades"),
) -> str:
    """A side-by-side table. Medians only — best-of-N is not on offer."""
    rows = [("challenger", challenger)] + sorted(baselines.items())
    header = f"{'policy':<22}" + "".join(f"{m:>16}" for m in metrics)
    lines = [header, "-" * len(header)]
    for name, runs in rows:
        summary = aggregate([r.metrics for r in runs], metrics)
        lines.append(f"{name:<22}" + "".join(
            f"{summary[m].median:>16.3f}" if m in summary else f"{'-':>16}" for m in metrics))
    return "\n".join(lines)
