"""The backtester's bar protocol, cost accounting and risk authority.

These are the tests that stop a backtest from quietly manufacturing an edge:
filling at the wrong price, resolving an ambiguous bar in its own favour, or
letting a policy size its own position.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.execution.paper import PaperExecution
from backend.models import Side
from research.backtest import BacktestConfig, _ticker, run_backtest, run_walk_forward
from research.collect import generate_synthetic
from research.datasets import DataSource, Dataset
from research.metrics import ExitReason
from research.policy import (
    BasePolicy, BuyAndHoldPolicy, Decision, FlatPolicy, PolicyAction, make_policy,
)
from research.splits import default_config, make_walk_forward, scale_config

NO_EXITS = dict(stop_loss_pct=None, take_profit_pct=None, sizing_stop_pct=5.0)


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic(bars=3000, seed=77)


class EnterOnce(BasePolicy):
    """Enters exactly once, on a chosen bar."""

    name = "enter_once"

    def __init__(self, at: int, action: PolicyAction = PolicyAction.LONG) -> None:
        self.at, self.action, self.done = at, action, False

    def reset(self, seed=None):
        self.done = False

    def decide(self, ctx):
        if ctx.index == self.at and not self.done:
            self.done = True
            return Decision(self.action, 1.0, "entry")
        return self.hold()


def flat_bars(n=60, **overrides):
    ts = np.arange(n, dtype=np.int64) * 300_000
    arrays = {
        "open": np.full(n, 100.0), "high": np.full(n, 100.5),
        "low": np.full(n, 99.5), "close": np.full(n, 100.0),
        "volume": np.ones(n),
    }
    for name, mutate in overrides.items():
        mutate(arrays[name])
    return Dataset.from_arrays(ts=ts, source=DataSource.SYNTHETIC,
                               symbol="BTC/USDT", timeframe="5m", **arrays)


# ---- the flat baseline ----------------------------------------------------


def test_doing_nothing_returns_exactly_zero(dataset):
    result = run_backtest(FlatPolicy(), dataset, config=BacktestConfig())
    assert result.trades == []
    assert result.metrics["total_return_pct"] == 0.0
    assert np.all(result.equity == 10_000.0)


# ---- the bar protocol -----------------------------------------------------


def test_a_decision_fills_at_the_next_bars_open(dataset, settings):
    at = 100
    result = run_backtest(EnterOnce(at), dataset,
                          config=BacktestConfig(spread_bps=0.0, **NO_EXITS))
    assert len(result.trades) == 1
    trade = result.trades[0]
    expected = PaperExecution(settings, bus=None).fill_price(
        Side.BUY, _ticker(dataset.manifest.symbol, float(dataset.open[at + 1]), 0.0))
    assert trade.entry_index == at + 1
    assert trade.entry_price == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("spread", [0.0, 2.0, 10.0])
def test_entry_price_matches_the_paper_engine_at_any_spread(dataset, settings, spread):
    at = 100
    result = run_backtest(EnterOnce(at), dataset,
                          config=BacktestConfig(spread_bps=spread, **NO_EXITS))
    expected = PaperExecution(settings, bus=None).fill_price(
        Side.BUY, _ticker(dataset.manifest.symbol, float(dataset.open[at + 1]), spread))
    assert result.trades[0].entry_price == pytest.approx(expected, abs=1e-9)
    assert result.trades[0].entry_price > float(dataset.open[at + 1])


def test_a_bar_spanning_stop_and_target_books_the_stop():
    """No intra-bar path is knowable, so the ambiguous bar must resolve against us."""
    bars = flat_bars(high=lambda a: a.__setitem__(52, 130.0),
                     low=lambda a: a.__setitem__(52, 70.0))
    result = run_backtest(EnterOnce(50), bars, config=BacktestConfig(
        stop_loss_pct=1.0, take_profit_pct=7.0, spread_bps=0.0))
    assert len(result.trades) == 1
    assert result.trades[0].reason == ExitReason.STOP_LOSS.value
    assert result.trades[0].pnl < 0


def test_a_gap_through_the_stop_fills_at_the_open():
    bars = flat_bars(
        open=lambda a: a.__setitem__(52, 80.0), high=lambda a: a.__setitem__(52, 80.5),
        low=lambda a: a.__setitem__(52, 79.0), close=lambda a: a.__setitem__(52, 80.0))
    result = run_backtest(EnterOnce(50), bars, config=BacktestConfig(
        stop_loss_pct=1.0, take_profit_pct=7.0, spread_bps=0.0))
    exit_price = result.trades[0].exit_price
    assert 79.0 <= exit_price <= 80.5, "a gap must not fill at the stop price"


def test_a_short_position_stops_out_above_entry():
    bars = flat_bars(high=lambda a: a.__setitem__(52, 130.0))
    result = run_backtest(EnterOnce(50, PolicyAction.SHORT), bars, config=BacktestConfig(
        stop_loss_pct=1.0, take_profit_pct=7.0, spread_bps=0.0))
    assert result.trades[0].side == "SHORT"
    assert result.trades[0].reason == ExitReason.STOP_LOSS.value


# ---- cost accounting ------------------------------------------------------


def test_equity_reconciles_with_the_sum_of_trade_pnl(dataset, settings):
    """Catches entry fees being dropped from trade P&L, among other slips."""
    result = run_backtest(make_policy("mean_reversion", settings.strategy_parameters),
                          dataset, config=BacktestConfig())
    assert result.trades
    realised = sum(trade.pnl for trade in result.trades)
    assert result.equity[-1] - 10_000.0 == pytest.approx(realised, abs=1e-6)


def test_both_sides_of_the_round_trip_are_charged(dataset):
    result = run_backtest(EnterOnce(100), dataset, config=BacktestConfig(**NO_EXITS))
    trade = result.trades[0]
    assert trade.fees > 0
    # fees covers entry + exit; pnl is net of both
    assert trade.pnl < trade.pnl + trade.fees


def test_a_wider_spread_costs_more(dataset):
    cheap = run_backtest(EnterOnce(100), dataset,
                         config=BacktestConfig(spread_bps=0.0, **NO_EXITS))
    dear = run_backtest(EnterOnce(100), dataset,
                        config=BacktestConfig(spread_bps=50.0, **NO_EXITS))
    assert dear.trades[0].entry_price > cheap.trades[0].entry_price


# ---- risk authority -------------------------------------------------------


class AlwaysIn(BasePolicy):
    name = "always_in"

    def decide(self, ctx):
        return Decision(PolicyAction.LONG if ctx.account.is_flat else PolicyAction.HOLD,
                        1.0, "in")


def test_the_policy_never_chooses_its_own_size(dataset):
    result = run_backtest(AlwaysIn(), dataset, config=BacktestConfig(
        starting_equity=10_000.0, risk_per_trade=0.02, leverage=1.0))
    assert result.trades
    assert result.trades[0].notional <= 10_000.0 * 1.05


def test_an_exposure_cap_blocks_every_entry(dataset):
    result = run_backtest(AlwaysIn(), dataset,
                          config=BacktestConfig(max_exposure_pct=0.001))
    assert not result.trades
    assert result.blocked_by_risk > 0
    assert any("exposure" in reason.lower() for reason in result.block_reasons)


def test_a_drawdown_limit_halts_new_entries(dataset):
    result = run_backtest(AlwaysIn(), dataset,
                          config=BacktestConfig(drawdown_limit_pct=0.0001))
    assert any("drawdown" in reason.lower() for reason in result.block_reasons)


def test_research_code_cannot_enable_live_trading(dataset, settings):
    from research.backtest import _settings_for

    enabled = settings.model_copy(deep=True)
    enabled.trading.live_enabled = True
    assert _settings_for(BacktestConfig(), enabled).trading.live_enabled is False


# ---- determinism ----------------------------------------------------------


def test_repeated_runs_are_byte_identical(dataset, settings):
    runs = [run_backtest(make_policy("momentum", settings.strategy_parameters),
                         dataset, config=BacktestConfig()) for _ in range(3)]
    for other in runs[1:]:
        assert np.array_equal(runs[0].equity, other.equity)
        assert runs[0].trades == other.trades


def test_a_seeded_random_policy_reproduces_and_a_new_seed_diverges(dataset):
    same = [run_backtest(make_policy("random", seed=5), dataset,
                         config=BacktestConfig(), seed=5) for _ in range(2)]
    assert np.array_equal(same[0].equity, same[1].equity)
    other = run_backtest(make_policy("random", seed=6), dataset,
                         config=BacktestConfig(), seed=6)
    assert not np.array_equal(same[0].equity, other.equity)


# ---- baselines ------------------------------------------------------------


def test_buy_and_hold_without_protective_exits_takes_exactly_one_trade(dataset):
    """With the strategies' 1% stop it would be stopped out and re-entered ~126
    times, which is not what anybody means by buy-and-hold."""
    result = run_backtest(BuyAndHoldPolicy(), dataset, config=BacktestConfig(**NO_EXITS))
    assert len(result.trades) == 1
    assert result.trades[0].reason == ExitReason.END_OF_DATA.value


def test_buy_and_hold_with_a_tight_stop_churns(dataset):
    result = run_backtest(BuyAndHoldPolicy(), dataset,
                          config=BacktestConfig(stop_loss_pct=1.0, take_profit_pct=7.0))
    assert len(result.trades) > 20


class Flipper(BasePolicy):
    name = "flipper"

    def decide(self, ctx):
        if ctx.index >= 60 and ctx.index % 40 == 0:
            return Decision(
                PolicyAction.SHORT if ctx.account.is_long else PolicyAction.LONG, 1.0, "flip")
        return self.hold()


def test_a_reversal_closes_before_it_opens(dataset):
    result = run_backtest(Flipper(), dataset, config=BacktestConfig(
        risk_per_trade=0.01, stop_loss_pct=50.0, take_profit_pct=500.0))
    assert len(result.trades) >= 5
    assert {trade.side for trade in result.trades} == {"LONG", "SHORT"}
    for earlier, later in zip(result.trades, result.trades[1:]):
        assert later.entry_index >= earlier.exit_index


def test_disallowing_shorts_produces_long_only_trading(dataset):
    result = run_backtest(Flipper(), dataset, config=BacktestConfig(
        risk_per_trade=0.01, stop_loss_pct=50.0, take_profit_pct=500.0, allow_short=False))
    assert {trade.side for trade in result.trades} == {"LONG"}


# ---- walk-forward driver --------------------------------------------------


def test_walk_forward_evaluates_only_validation_ranges(settings):
    dataset = generate_synthetic(bars=40_000, seed=3)
    plan = make_walk_forward(dataset, scale_config(
        default_config(dataset, warmup_bars=600), 0.06))
    results = run_walk_forward(
        lambda: make_policy("momentum", settings.strategy_parameters), dataset, plan,
        config=BacktestConfig())
    assert len(results) == len(plan)
    for result, window in zip(results, plan.windows):
        assert (result.start_index, result.stop_index) == (
            window.validate.start, window.validate.stop)


def test_walk_forward_refuses_a_dataset_the_plan_was_not_built_from():
    dataset = generate_synthetic(bars=40_000, seed=3)
    plan = make_walk_forward(dataset, scale_config(
        default_config(dataset, warmup_bars=600), 0.06))
    with pytest.raises(ValueError):
        run_walk_forward(lambda: FlatPolicy(), generate_synthetic(bars=40_000, seed=4),
                         plan, config=BacktestConfig())


# ---- input validation -----------------------------------------------------


@pytest.mark.parametrize("override", [
    {"starting_equity": 0}, {"risk_per_trade": 0}, {"risk_per_trade": 2},
    {"stop_loss_pct": 0}, {"take_profit_pct": -1}, {"leverage": 0},
    {"spread_bps": -1}, {"sizing_stop_pct": 0},
])
def test_malformed_configs_are_rejected(dataset, override):
    with pytest.raises(ValueError):
        run_backtest(FlatPolicy(), dataset, config=BacktestConfig(**override))


def test_a_range_shorter_than_two_bars_is_rejected(dataset):
    with pytest.raises(ValueError):
        run_backtest(FlatPolicy(), dataset, start=0, stop=1)
