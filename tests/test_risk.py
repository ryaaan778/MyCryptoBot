"""Risk engine: sizing, gating, escalation and the kill switch."""

from __future__ import annotations

import pytest

from backend.eventbus import EventBus
from backend.execution.base import Fill
from backend.execution.paper import PaperExecution
from backend.models import PositionSide, RiskLevel
from backend.portfolio import Portfolio
from backend.risk import RiskEngine


@pytest.fixture
def engine(settings):
    portfolio = Portfolio(settings.initial_balance)
    return RiskEngine(settings, portfolio), portfolio


def bot(settings, bot_id: str = "jotaro"):
    found = settings.bot(bot_id)
    assert found is not None
    return found


def open_position(portfolio, cfg, price=100.0, quantity=1.0, side=PositionSide.LONG):
    return portfolio.open_position(
        bot_id=cfg.id, symbol=cfg.symbol, side=side,
        fill=Fill(order_id="o", price=price, quantity=quantity, fee=0.0),
        leverage=cfg.leverage,
    )


# --------------------------------------------------------------------- sizing


def test_size_matches_the_per_trade_risk_budget(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings, "jonathan")
    decision = risk.size_position(cfg, price=100.0, stop_distance_pct=2.0)
    assert decision.allowed

    # Being stopped out must cost exactly the configured per-trade budget.
    allocated = portfolio.equity * cfg.allocation
    expected_loss = allocated * min(cfg.risk_per_trade, settings.risk.max_risk_per_trade)
    actual_loss = decision.quantity * 100.0 * 0.02
    assert actual_loss == pytest.approx(expected_loss, rel=1e-6)


def test_a_tighter_stop_permits_a_larger_position(engine, settings):
    risk, _ = engine
    cfg = bot(settings, "jonathan")
    tight = risk.size_position(cfg, 100.0, 0.5).quantity
    wide = risk.size_position(cfg, 100.0, 4.0).quantity
    assert tight > wide


def test_sizing_is_capped_by_the_bots_allocation(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings, "kira")
    decision = risk.size_position(cfg, price=100.0, stop_distance_pct=0.01)
    margin = decision.quantity * 100.0 / cfg.leverage
    allocated = portfolio.equity * cfg.allocation
    assert margin <= allocated * 1.000001, "margin must not exceed the allocation"


def test_sizing_rejects_nonsense_inputs(engine, settings):
    risk, _ = engine
    cfg = bot(settings)
    assert not risk.size_position(cfg, price=0.0, stop_distance_pct=1.0).allowed
    assert not risk.size_position(cfg, price=100.0, stop_distance_pct=0.0).allowed
    assert not risk.size_position(cfg, price=100.0, stop_distance_pct=-1.0).allowed


def test_sizing_fails_when_equity_is_gone(engine, settings):
    risk, portfolio = engine
    portfolio.cash = 0.0
    portfolio.starting_equity = 0.0
    decision = risk.size_position(bot(settings), 100.0, 1.0)
    assert not decision.allowed
    assert "equity" in decision.reason


# --------------------------------------------------------------------- gating


def test_desk_position_cap_blocks_further_entries(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings)
    settings.risk.max_portfolio_positions = 2
    open_position(portfolio, cfg)
    open_position(portfolio, cfg)
    decision = risk.can_open(cfg, notional=10.0)
    assert not decision.allowed
    assert "desk already holds 2/2" in decision.reason


def test_per_bot_cap_blocks_only_that_bot(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings)
    cfg.max_positions = 1
    open_position(portfolio, cfg)
    assert not risk.can_open(cfg, notional=10.0).allowed

    other = bot(settings, "jonathan")
    assert risk.can_open(other, notional=10.0).allowed, "other bots must be unaffected"


def test_exposure_limit_blocks_an_oversized_order(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings)
    huge = portfolio.equity * (settings.risk.max_exposure_pct / 100.0) + 1_000.0
    decision = risk.can_open(cfg, notional=huge)
    assert not decision.allowed
    assert "exposure would reach" in decision.reason


def test_drawdown_limit_stops_new_entries(engine, settings):
    risk, portfolio = engine
    portfolio.peak_equity = 20_000.0          # equity is 10k -> 50% drawdown
    decision = risk.can_open(bot(settings), notional=10.0)
    assert not decision.allowed
    assert "drawdown" in decision.reason


def test_halted_bot_cannot_open(engine, settings):
    risk, _ = engine
    cfg = bot(settings)
    risk.halt(cfg.id)
    assert not risk.can_open(cfg, 10.0).allowed
    risk.resume(cfg.id)
    assert risk.can_open(cfg, 10.0).allowed


def test_emergency_stop_blocks_every_bot(engine, settings):
    risk, _ = engine
    risk.trigger_emergency_stop()
    for cfg in settings.bots:
        decision = risk.can_open(cfg, 10.0)
        assert not decision.allowed
        assert "EMERGENCY STOP" in decision.reason


# ------------------------------------------------------------------ escalation


def test_a_full_book_alone_does_not_read_critical(engine, settings):
    """Being fully deployed is not the same as being in danger.

    Slot pressure is capped below the danger bands on purpose — otherwise the
    risk meter pins to CRITICAL the moment the desk is simply busy, and stops
    carrying information.
    """
    risk, portfolio = engine
    settings.risk.max_portfolio_positions = 3
    cfg = bot(settings)
    for _ in range(3):
        open_position(portfolio, cfg, price=1.0, quantity=0.001)
    assert risk.utilization() <= RiskEngine.SLOT_PRESSURE_CEILING
    assert risk.level() is not RiskLevel.CRITICAL


def test_drawdown_escalates_to_critical(engine, settings):
    risk, portfolio = engine
    portfolio.peak_equity = portfolio.equity / (
        1 - settings.risk.drawdown_limit_pct / 100.0
    )
    assert risk.utilization() == pytest.approx(1.0, rel=0.05)
    assert risk.level() is RiskLevel.CRITICAL


def test_evaluate_reports_breaches(engine, settings):
    risk, portfolio = engine
    portfolio.peak_equity = 100_000.0         # a catastrophic drawdown
    state = risk.evaluate()
    assert state.breaches
    assert state.level is RiskLevel.CRITICAL
    assert state.max_open_positions == settings.risk.max_portfolio_positions


def test_leverage_raises_the_per_bot_risk_reading(engine, settings):
    """KIRA at 8x carrying the same margin as JONATHAN at 3x is riskier."""
    risk, portfolio = engine
    low, high = bot(settings, "jonathan"), bot(settings, "kira")
    low.allocation = high.allocation = 0.2

    open_position(portfolio, low, price=100.0, quantity=low.leverage * 4.0)
    open_position(portfolio, high, price=100.0, quantity=high.leverage * 4.0)

    low_margin = sum(p.margin for p in portfolio.positions.values() if p.bot_id == low.id)
    high_margin = sum(p.margin for p in portfolio.positions.values() if p.bot_id == high.id)
    assert low_margin == pytest.approx(high_margin), "same margin, different leverage"

    assert risk.bot_risk(high)[1] > risk.bot_risk(low)[1]


def test_open_losses_raise_the_per_bot_risk_reading(engine, settings):
    risk, portfolio = engine
    cfg = bot(settings, "jonathan")
    open_position(portfolio, cfg, price=100.0, quantity=5.0)
    calm = risk.bot_risk(cfg)[1]
    portfolio.mark(cfg.symbol, 80.0)
    assert risk.bot_risk(cfg)[1] > calm


def test_halted_bot_reads_critical(engine, settings):
    risk, _ = engine
    cfg = bot(settings)
    risk.halt(cfg.id)
    level, utilization = risk.bot_risk(cfg)
    assert level is RiskLevel.CRITICAL
    assert utilization == pytest.approx(1.0)


# ----------------------------------------------------------------- stop levels


@pytest.mark.parametrize(
    "side,expected_stop_below",
    [(PositionSide.LONG, True), (PositionSide.SHORT, False)],
)
def test_stop_levels_sit_on_the_correct_side(side, expected_stop_below):
    stop, target = RiskEngine.stop_levels(side, entry=100.0, stop_loss_pct=2.0, take_profit_pct=5.0)
    if expected_stop_below:
        assert stop == pytest.approx(98.0) and target == pytest.approx(105.0)
    else:
        assert stop == pytest.approx(102.0) and target == pytest.approx(95.0)


def test_clearing_the_emergency_stop_releases_every_halt(engine, settings):
    risk, _ = engine
    risk.trigger_emergency_stop()
    risk.halt("kira")
    risk.clear_emergency_stop()
    assert not risk.emergency_stop
    assert risk.halted_bots == set()
    assert risk.can_open(bot(settings, "kira"), 10.0).allowed
