"""The promotion gate.

Every condition is tested in isolation by constructing results that fail exactly
one of them. A gate that can be passed by being spectacular on one axis is not a
gate, so the conjunction itself is the thing under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.backtest import BacktestResult
from research.evaluate import GateConfig, evaluate_gate
from research.metrics import MetricSet, TradeRecord, compute_metrics

BARS_PER_YEAR = 288 * 365


def result(*, sortino=1.0, total_return=5.0, drawdown=10.0, trades=40,
           seed=0, window=0) -> BacktestResult:
    """A BacktestResult with the metrics the gate reads, set directly."""
    equity = np.array([10_000.0, 10_000.0 * (1 + total_return / 100)])
    ts = np.array([0, 300_000], dtype=np.int64)
    base = compute_metrics(equity, ts, [], bars_per_year=BARS_PER_YEAR)
    values = dict(base.values)
    values.update({"sortino": sortino, "total_return_pct": total_return,
                   "max_drawdown_pct": drawdown, "trades": float(trades)})
    return BacktestResult(
        policy="challenger", dataset_id="d", dataset_source="SYNTHETIC",
        symbol="BTC/USDT", timeframe="5m", start_index=0, stop_index=2,
        ts=ts, equity=equity, trades=[],
        metrics=MetricSet(values=values, trades=trades, bars=2),
        actions=np.zeros(2, dtype=np.int8), seed=seed, window=window,
    )


def good_challenger(**overrides):
    """Five seeds across three windows, comfortably clearing every condition."""
    settings = dict(sortino=2.0, total_return=6.0, drawdown=8.0, trades=10)
    settings.update(overrides)
    return [result(seed=s, window=w, **settings) for w in range(3) for s in range(5)]


def weak_baselines(metric=0.5):
    return {"flat": [result(sortino=metric, total_return=0.0, drawdown=0.0, trades=0,
                            seed=s, window=w) for w in range(3) for s in range(5)],
            "buy_and_hold": [result(sortino=metric, total_return=1.0, drawdown=5.0,
                                    trades=1, seed=s, window=w)
                             for w in range(3) for s in range(5)]}


def regimes(worst_dd=12.0):
    return {"trending_up": MetricSet({"max_drawdown_pct": 6.0}, 20, 500),
            "choppy": MetricSet({"max_drawdown_pct": worst_dd}, 20, 500)}


def test_a_strong_challenger_with_no_incumbent_promotes():
    gate = evaluate_gate("JOLYAN_v1", good_challenger(), baselines=weak_baselines(),
                         regime_metrics=regimes())
    assert gate.passed, gate.reason()
    assert all(condition.passed for condition in gate.conditions)


def test_losing_to_a_baseline_blocks_promotion():
    gate = evaluate_gate("JOLYAN_v1", good_challenger(sortino=0.3),
                         baselines=weak_baselines(metric=1.5), regime_metrics=regimes())
    assert not gate.passed
    assert "beats_best_baseline" in gate.reason()


def test_flat_is_a_real_baseline_to_clear():
    """A negative-Sortino policy loses to doing nothing, which scores zero."""
    baselines = {"flat": [result(sortino=0.0, total_return=0.0, drawdown=0.0, trades=0,
                                 seed=s, window=w) for w in range(3) for s in range(5)]}
    gate = evaluate_gate("JOLYAN_v1", good_challenger(sortino=-0.5, total_return=-2.0),
                         baselines=baselines, regime_metrics=regimes())
    assert not gate.passed
    failed = {c.name for c in gate.failures}
    assert {"beats_best_baseline", "positive_median_return"} <= failed


def test_losing_to_the_champion_blocks_promotion():
    champion = [result(sortino=3.0, drawdown=8.0, seed=s, window=w)
                for w in range(3) for s in range(5)]
    gate = evaluate_gate("JOLYAN_v2", good_challenger(sortino=2.0),
                         baselines=weak_baselines(), champion=champion,
                         regime_metrics=regimes())
    assert not gate.passed
    assert "beats_champion" in gate.reason()


def test_a_challenger_may_not_buy_return_with_drawdown():
    champion = [result(sortino=1.0, drawdown=10.0, seed=s, window=w)
                for w in range(3) for s in range(5)]
    gate = evaluate_gate("JOLYAN_v2", good_challenger(sortino=5.0, drawdown=30.0),
                         baselines=weak_baselines(), champion=champion,
                         regime_metrics=regimes(worst_dd=30.0))
    assert not gate.passed
    assert "drawdown_within_champion_tolerance" in gate.reason()


def test_a_slightly_deeper_drawdown_is_tolerated():
    champion = [result(sortino=1.0, drawdown=10.0, seed=s, window=w)
                for w in range(3) for s in range(5)]
    gate = evaluate_gate("JOLYAN_v2", good_challenger(sortino=5.0, drawdown=10.5),
                         baselines=weak_baselines(), champion=champion,
                         regime_metrics=regimes(worst_dd=15.0))
    assert gate.passed, gate.reason()


def test_too_few_trades_blocks_promotion():
    """A Sortino computed from four trades is a random number with a decimal point."""
    thin = [result(sortino=9.0, trades=1, seed=s, window=w)
            for w in range(3) for s in range(5)]
    gate = evaluate_gate("JOLYAN_v1", thin, baselines=weak_baselines(),
                         regime_metrics=regimes())
    assert not gate.passed
    assert "minimum_trades" in gate.reason()


def test_too_few_seeds_blocks_promotion():
    """One lucky seed is not a policy."""
    single = [result(sortino=9.0, trades=40, seed=0, window=w) for w in range(3)]
    gate = evaluate_gate("JOLYAN_v1", single, baselines=weak_baselines(),
                         regime_metrics=regimes())
    assert not gate.passed
    assert "minimum_seeds" in gate.reason()


def test_a_negative_median_blocks_promotion_even_with_a_good_ratio():
    runs = [result(sortino=2.0, total_return=r, trades=10, seed=s, window=w)
            for w in range(3) for s, r in enumerate([-4.0, -3.0, -1.0, 2.0, 40.0])]
    gate = evaluate_gate("JOLYAN_v1", runs, baselines=weak_baselines(),
                         regime_metrics=regimes())
    assert not gate.passed
    assert "positive_median_return" in gate.reason()


def test_a_catastrophic_regime_blocks_promotion():
    gate = evaluate_gate("JOLYAN_v1", good_challenger(drawdown=8.0),
                         baselines=weak_baselines(), regime_metrics=regimes(worst_dd=40.0))
    assert not gate.passed
    assert "no_catastrophic_regime" in gate.reason()


def test_missing_regime_metrics_cannot_promote():
    """Unverifiable is not the same as satisfied."""
    gate = evaluate_gate("JOLYAN_v1", good_challenger(), baselines=weak_baselines())
    assert not gate.passed
    assert "no_catastrophic_regime" in gate.reason()


def test_a_decayed_edge_blocks_promotion():
    """Strong early windows must not carry a policy that has stopped working."""
    runs = ([result(sortino=8.0, trades=10, seed=s, window=w)
             for w in (0, 1) for s in range(5)]
            + [result(sortino=-2.0, total_return=6.0, trades=10, seed=s, window=w)
               for w in (2, 3, 4) for s in range(5)])
    gate = evaluate_gate("JOLYAN_v1", runs, baselines=weak_baselines(metric=0.5),
                         regime_metrics=regimes())
    assert not gate.passed
    assert "recent_windows_hold_up" in gate.reason()


def test_results_without_window_indices_cannot_promote():
    runs = [result(seed=s, window=None) for s in range(5)]
    for run in runs:
        run.window = None
    gate = evaluate_gate("JOLYAN_v1", runs, baselines=weak_baselines(),
                         regime_metrics=regimes())
    assert not gate.passed
    assert "recent_windows_hold_up" in gate.reason()


def test_every_condition_is_reported_not_just_the_first_failure():
    """"Rejected" teaches an agent nothing; "rejected: 22 trades, needed 30" does."""
    bad = [result(sortino=-1.0, total_return=-5.0, drawdown=50.0, trades=1, seed=0, window=0)]
    gate = evaluate_gate("JOLYAN_v1", bad, baselines=weak_baselines(metric=2.0))
    assert not gate.passed
    assert len(gate.conditions) == 7
    assert len(gate.failures) >= 5
    for condition in gate.conditions:
        assert condition.detail
    assert "PASS" in gate.describe() or "FAIL" in gate.describe()


def test_the_gate_result_serialises_for_the_evaluations_table():
    gate = evaluate_gate("JOLYAN_v1", good_challenger(), baselines=weak_baselines(),
                         regime_metrics=regimes())
    payload = gate.as_dict()
    assert payload["passed"] is True
    assert set(payload["conditions"]) == {c.name for c in gate.conditions}
    assert payload["summary"]["trades"] > 0


def test_an_empty_challenger_is_an_error():
    with pytest.raises(ValueError):
        evaluate_gate("JOLYAN_v1", [], baselines=weak_baselines())


@pytest.mark.parametrize("override", [{"min_trades": 0}, {"min_seeds": 0},
                                      {"max_drawdown_multiple": 0},
                                      {"recent_windows_required": 5, "recent_windows": 3}])
def test_malformed_gate_configs_are_rejected(override):
    with pytest.raises(ValueError):
        GateConfig(**override).validate()


def test_the_comparison_table_reports_medians():
    from research.evaluate import compare_to_baselines

    table = compare_to_baselines(good_challenger(), weak_baselines())
    assert "challenger" in table and "buy_and_hold" in table and "flat" in table


# ---- stitching walk-forward windows into one curve -------------------------


class FakeLabels:
    def __init__(self, regime):
        self.regime = regime


def windowed(equity, start, trades=()):
    from research.metrics import TradeRecord

    array = np.asarray(equity, dtype=np.float64)
    return BacktestResult(
        policy="p", dataset_id="d", dataset_source="SYNTHETIC", symbol="BTC/USDT",
        timeframe="5m", start_index=start, stop_index=start + array.size,
        ts=np.arange(array.size, dtype=np.int64) + start,
        equity=array, trades=list(trades),
        metrics=MetricSet({}, len(trades), array.size),
        actions=np.zeros(array.size, dtype=np.int8), seed=0, window=0,
    )


def test_stitching_chains_returns_instead_of_concatenating_curves():
    """Each window restarts at the same capital; concatenating fabricates a jump."""
    from research.evaluate import stitch_windows

    labels = FakeLabels(np.zeros(100, dtype=np.int8))
    equity, _, _, _ = stitch_windows(
        [windowed([10_000.0, 9_200.0], 0), windowed([10_000.0, 10_500.0], 10)], labels)
    assert equity[2] == pytest.approx(equity[1]), "a fabricated jump at the boundary"
    assert equity[-1] == pytest.approx(10_000.0 * 0.92 * 1.05)


def test_stitching_remaps_trade_indices_into_the_stitched_frame():
    """Trades carry absolute dataset positions; the stitched arrays do not.

    Left unmapped, a trade either falls outside the array and is silently
    dropped, or lands on an unrelated bar and is attributed to the wrong regime.
    Neither raises.
    """
    from research.evaluate import stitch_windows
    from research.metrics import TradeRecord

    def trade_at(entry, exit_):
        return TradeRecord(entry_index=entry, exit_index=exit_, side="LONG",
                           entry_price=100.0, exit_price=101.0, quantity=1.0,
                           pnl=1.0, fees=0.0)

    labels = FakeLabels(np.zeros(100, dtype=np.int8))
    windows = [
        windowed([10_000.0] * 5, 20, [trade_at(21, 23)]),
        windowed([10_000.0] * 5, 60, [trade_at(61, 64)]),
    ]
    equity, _, regime, trades = stitch_windows(windows, labels)
    assert [(t.entry_index, t.exit_index) for t in trades] == [(1, 3), (6, 9)]
    assert all(0 <= t.exit_index < equity.size for t in trades)
    assert all(0 <= t.exit_index < regime.size for t in trades)


def test_stitched_trades_are_attributed_to_a_regime():
    from research.evaluate import stitch_windows
    from research.metrics import TradeRecord, metrics_by_regime

    regime = np.zeros(100, dtype=np.int8)
    regime[60:] = 1
    labels = FakeLabels(regime)
    trades_a = [TradeRecord(21, 24, "LONG", 100.0, 99.0, 1.0, -1.0, 0.0)]
    trades_b = [TradeRecord(61, 64, "LONG", 100.0, 102.0, 1.0, 2.0, 0.0)]
    windows = [windowed(np.linspace(10_000, 9_900, 20), 20, trades_a),
               windowed(np.linspace(10_000, 10_200, 20), 60, trades_b)]
    equity, ts, stitched_regime, trades = stitch_windows(windows, labels)
    by_regime = metrics_by_regime(equity, ts, trades, stitched_regime,
                                  bars_per_year=288 * 365, min_bars=5)
    attributed = sum(int(m["trades"]) for m in by_regime.values())
    assert attributed == len(trades) == 2


def test_regime_veto_is_not_fooled_by_one_idle_seed():
    """A seed that learned to sit out must not make the veto vacuous.

    Doing nothing scores exactly zero under this reward, so an idle policy is a
    genuine local optimum — and it really did occur. Reporting one arbitrary
    seed would then show 0.00% drawdown in every regime and the
    catastrophic-regime condition would pass without checking anything.
    """
    from research.evaluate import regime_metrics_across_seeds
    from research.metrics import TradeRecord

    regime = np.zeros(400, dtype=np.int8)
    regime[200:] = 1
    labels = FakeLabels(regime)

    idle = [windowed(np.full(200, 10_000.0), 0), windowed(np.full(200, 10_000.0), 200)]
    for run in idle:
        run.seed = 0
    losing = [
        windowed(np.linspace(10_000, 9_000, 200), 0,
                 [TradeRecord(10, 20, "LONG", 100.0, 90.0, 1.0, -10.0, 0.0)]),
        windowed(np.linspace(10_000, 4_000, 200), 200,
                 [TradeRecord(210, 220, "LONG", 100.0, 60.0, 1.0, -40.0, 0.0)]),
    ]
    for run in losing:
        run.seed = 1

    pooled, worst_overall = regime_metrics_across_seeds(
        idle + losing, labels, bars_per_year=288 * 365, min_bars=50,
        regime_name=lambda code: {0: "quiet_range", 1: "choppy"}[code])

    assert set(pooled) == {"quiet_range", "choppy"}
    assert all(m["seeds"] == 2 for m in pooled.values())
    worst = max(m["max_drawdown_pct"] for m in pooled.values())
    assert worst > 30.0, f"the losing seed's drawdown was hidden: {worst:.2f}%"
    assert sum(int(m["trades"]) for m in pooled.values()) == 2
    assert worst_overall > 30.0, "the stitched overall drawdown must be reported too"


def test_the_regime_threshold_uses_a_comparable_basis():
    """Per-regime drawdown is compounded across windows; per-window median is not.

    Comparing the two directly makes the threshold structurally unreachable, so
    the gate would reject a policy for a drawdown its own basis could never
    match. The caller supplies the stitched overall figure instead.
    """
    challenger = good_challenger(drawdown=5.0)          # per-window median
    high_regimes = regimes(worst_dd=40.0)               # stitched, compounded

    tight = evaluate_gate("P", challenger, baselines=weak_baselines(),
                          regime_metrics=high_regimes)
    assert not tight.passed        # 40% vs 5% x 2 — rejected on the wrong basis

    fair = evaluate_gate("P", challenger, baselines=weak_baselines(),
                         regime_metrics=high_regimes, overall_drawdown_pct=35.0)
    condition = next(c for c in fair.conditions if c.name == "no_catastrophic_regime")
    assert condition.passed, condition.detail
    assert "35.00" in condition.detail

    still_bad = evaluate_gate("P", challenger, baselines=weak_baselines(),
                              regime_metrics=regimes(worst_dd=90.0),
                              overall_drawdown_pct=35.0)
    assert not still_bad.passed
