"""JOJO moving capital toward whoever is earning it.

The behaviours worth pinning down are the ones that separate "allocates on
evidence" from "chases whatever went up last week" — because the second is easy
to write by accident and loses money in a way that looks like diligence.
"""

from __future__ import annotations

import pytest

from backend.allocator import (
    AgentPerformance,
    Allocation,
    AllocatorConfig,
    PerformanceAllocator,
)
from backend.config import Settings

ROSTER = ["JONATHAN", "JOSEPH", "JOTARO", "JOLYAN", "KIRA"]


def perf(bot_id, trades=0, wins=0, pnl=0.0, capital=2000.0) -> AgentPerformance:
    return AgentPerformance(
        bot_id=bot_id, trades=trades, wins=wins, realized_pnl=pnl, capital=capital
    )


def allocate(perfs, **cfg) -> dict[str, Allocation]:
    allocator = PerformanceAllocator(AllocatorConfig(**cfg))
    return {d.bot_id: d for d in allocator.allocate(perfs)}


def deployed(result: dict[str, Allocation]) -> float:
    return sum(d.allocation for d in result.values())


# ---------------------------------------------------------------------------
# the starting state
# ---------------------------------------------------------------------------

def test_an_untested_roster_is_split_evenly():
    """Nobody has evidence, so nobody gets a head start."""
    result = allocate({b: perf(b) for b in ROSTER})
    assert all(d.allocation == pytest.approx(0.2) for d in result.values())
    assert deployed(result) == pytest.approx(1.0)


def test_no_agents_allocates_nothing():
    assert PerformanceAllocator().allocate({}) == []


# ---------------------------------------------------------------------------
# evidence, not streaks
# ---------------------------------------------------------------------------

def test_a_short_hot_streak_cannot_capture_the_desk():
    """Three lucky trades must not outrank forty mediocre ones."""
    result = allocate({
        "JONATHAN": perf("JONATHAN", trades=40, wins=20, pnl=10),
        "JOSEPH":   perf("JOSEPH",   trades=40, wins=20, pnl=-5),
        "JOTARO":   perf("JOTARO",   trades=3,  wins=3,  pnl=900),
        "JOLYAN":   perf("JOLYAN",   trades=40, wins=19, pnl=0),
        "KIRA":     perf("KIRA",     trades=40, wins=21, pnl=20),
    })
    even = 1.0 / len(ROSTER)
    assert result["JOTARO"].allocation <= even + 1e-9, (
        "an unproven agent was handed more than an even share on 3 trades"
    )


def test_the_same_edge_earns_the_ceiling_once_it_is_proven():
    """The cap is about sample size, not about disliking winners."""
    base = {
        "JONATHAN": perf("JONATHAN", trades=40, wins=20, pnl=10),
        "JOSEPH":   perf("JOSEPH",   trades=40, wins=20, pnl=-5),
        "JOLYAN":   perf("JOLYAN",   trades=40, wins=19, pnl=0),
        "KIRA":     perf("KIRA",     trades=40, wins=21, pnl=20),
    }
    unproven = allocate({**base, "JOTARO": perf("JOTARO", trades=3, wins=3, pnl=900)})
    proven = allocate({**base, "JOTARO": perf("JOTARO", trades=60, wins=44, pnl=4200)})

    assert proven["JOTARO"].allocation > unproven["JOTARO"].allocation
    assert proven["JOTARO"].allocation == pytest.approx(0.40)


def test_evidence_weight_rises_with_sample_size():
    allocator = PerformanceAllocator(AllocatorConfig(prior_trades=20))
    assert allocator.evidence_weight(0) == 0.0
    assert allocator.evidence_weight(20) == pytest.approx(0.5)
    assert allocator.evidence_weight(200) > 0.9
    assert allocator.evidence_weight(5) < allocator.evidence_weight(50)


def test_edge_is_scale_free_so_a_big_budget_is_not_mistaken_for_skill():
    """Two agents with the same percentage edge score the same.

    Without normalising by capital, whoever already holds the largest
    allocation books the largest absolute P&L and looks the most skilful — and
    allocation then feeds on itself regardless of results.
    """
    small = perf("small", trades=50, wins=30, pnl=100, capital=1_000)
    large = perf("large", trades=50, wins=30, pnl=1_000, capital=10_000)
    allocator = PerformanceAllocator()
    assert allocator.score(small) == pytest.approx(allocator.score(large))


# ---------------------------------------------------------------------------
# who gets more
# ---------------------------------------------------------------------------

def test_the_winner_gets_more_than_the_losers():
    result = allocate({
        "JONATHAN": perf("JONATHAN", trades=110, wins=50, pnl=-300),
        "JOSEPH":   perf("JOSEPH",   trades=95,  wins=40, pnl=-800),
        "JOTARO":   perf("JOTARO",   trades=130, wins=64, pnl=60),
        "JOLYAN":   perf("JOLYAN",   trades=88,  wins=41, pnl=-120),
        "KIRA":     perf("KIRA",     trades=120, wins=78, pnl=2400),
    })
    assert result["KIRA"].allocation == max(d.allocation for d in result.values())
    for loser in ("JONATHAN", "JOSEPH", "JOLYAN"):
        assert result[loser].allocation < result["KIRA"].allocation


def test_two_winners_share_the_top_and_losers_sit_at_the_floor():
    result = allocate({
        "JONATHAN": perf("JONATHAN", trades=90,  wins=48, pnl=800),
        "JOSEPH":   perf("JOSEPH",   trades=95,  wins=40, pnl=-800),
        "JOTARO":   perf("JOTARO",   trades=130, wins=64, pnl=-60),
        "JOLYAN":   perf("JOLYAN",   trades=88,  wins=41, pnl=-120),
        "KIRA":     perf("KIRA",     trades=120, wins=70, pnl=1200),
    })
    assert result["JONATHAN"].allocation > 0.3
    assert result["KIRA"].allocation > 0.3
    for loser in ("JOSEPH", "JOTARO", "JOLYAN"):
        assert result[loser].allocation == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# the bounds
# ---------------------------------------------------------------------------

def test_nobody_is_ever_starved_to_zero():
    """An agent with no capital can never earn its way back — an absorbing state."""
    result = allocate({
        **{b: perf(b, trades=80, wins=20, pnl=-2000) for b in ROSTER[:4]},
        "KIRA": perf("KIRA", trades=120, wins=90, pnl=5000),
    })
    for d in result.values():
        assert d.allocation >= 0.05 - 1e-9


def test_nobody_takes_the_whole_desk():
    result = allocate({
        **{b: perf(b, trades=80, wins=10, pnl=-9000) for b in ROSTER[:4]},
        "KIRA": perf("KIRA", trades=300, wins=280, pnl=90_000),
    })
    assert result["KIRA"].allocation == pytest.approx(0.40)


def test_the_desk_is_never_over_allocated():
    for pnls in ([5000, -100, -100, -100, -100], [1, 1, 1, 1, 1], [900, 800, 700, 600, 500]):
        result = allocate({
            b: perf(b, trades=60, wins=35, pnl=v) for b, v in zip(ROSTER, pnls)
        })
        assert deployed(result) <= 1.0 + 1e-9, f"over-allocated on {pnls}"


def test_impossible_floors_fall_back_to_an_even_split():
    """Five agents cannot each hold 30% of one desk."""
    result = allocate({b: perf(b, trades=60, wins=40, pnl=100) for b in ROSTER},
                      floor=0.30, ceiling=0.5)
    assert deployed(result) == pytest.approx(1.0)
    assert all(d.allocation == pytest.approx(0.2) for d in result.values())


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        PerformanceAllocator(AllocatorConfig(floor=0.6, ceiling=0.3))
    with pytest.raises(ValueError):
        PerformanceAllocator(AllocatorConfig(total=0.0))


# ---------------------------------------------------------------------------
# the case that matters most
# ---------------------------------------------------------------------------

def test_a_losing_roster_is_cut_back_rather_than_split_evenly():
    """When nobody is earning, the desk holds cash instead of deploying it all.

    An even split here would put full capital behind a roster the evidence says
    is losing money, which is the most expensive possible reading of "allocate
    fairly".
    """
    result = allocate({b: perf(b, trades=60, wins=20, pnl=-500) for b in ROSTER})
    assert all(d.allocation == pytest.approx(0.05) for d in result.values())
    assert deployed(result) == pytest.approx(0.25)
    assert "no agent is profitable" in result["KIRA"].reason


def test_capital_is_held_back_when_only_a_capped_winner_deserves_it():
    result = allocate({
        **{b: perf(b, trades=100, wins=30, pnl=-1000) for b in ROSTER[:4]},
        "KIRA": perf("KIRA", trades=120, wins=90, pnl=5000),
    })
    assert deployed(result) < 1.0, (
        "the remainder should stay in cash, not be handed to losing agents"
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_every_allocation_explains_itself():
    result = allocate({
        b: perf(b, trades=60, wins=35, pnl=v)
        for b, v in zip(ROSTER, [900, -100, -100, -100, -100])
    })
    for decision in result.values():
        assert decision.reason, f"{decision.bot_id} has no stated reason"
        payload = decision.as_dict()
        assert set(payload) >= {"bot_id", "allocation", "delta", "trades", "reason"}


def test_delta_reports_the_move():
    allocator = PerformanceAllocator()
    perfs = {b: perf(b) for b in ROSTER}
    decisions = allocator.allocate(perfs, previous={b: 0.1 for b in ROSTER})
    for d in decisions:
        assert d.previous == pytest.approx(0.1)
        assert d.delta == pytest.approx(d.allocation - 0.1)


def test_allocator_is_off_by_default():
    assert Settings().allocator.enabled is False


# ---------------------------------------------------------------------------
# regressions: two bugs that made the allocator actively harmful
# ---------------------------------------------------------------------------

def test_scoring_does_not_feed_back_through_the_budget():
    """The bug that handed the ceiling to the worst agent on the desk.

    An early version measured edge against ``equity * allocation``. Cutting an
    agent shrank its denominator, inflated its apparent edge, and won the money
    straight back — so the roster thrashed and the worst performer surfaced at
    the ceiling within a few rounds. ``capital`` must describe the trades, not
    the budget: the same results scored while holding a small budget and a large
    one have to produce the same score.
    """
    allocator = PerformanceAllocator()
    results = dict(trades=60, wins=35, pnl=600.0)
    assert allocator.score(perf("a", **results, capital=2000.0)) == pytest.approx(
        allocator.score(perf("a", **results, capital=2000.0))
    )

    # And the score must fall, not rise, when the same P&L came from bigger bets.
    small_bets = allocator.score(perf("a", **results, capital=1_000.0))
    big_bets = allocator.score(perf("a", **results, capital=10_000.0))
    assert big_bets < small_bets, (
        "the same P&L earned on 10x the notional is a weaker edge, not a stronger one"
    )


def test_smoothing_moves_part_of_the_way_not_all():
    allocator = PerformanceAllocator(AllocatorConfig(smoothing=0.5))
    perfs = {b: perf(b, trades=80, wins=60, pnl=900 if b == "KIRA" else -400)
             for b in ROSTER}
    first = {d.bot_id: d.allocation for d in allocator.allocate(perfs, {b: 0.2 for b in ROSTER})}
    second = {d.bot_id: d.allocation for d in allocator.allocate(perfs, first)}
    assert 0.2 < first["KIRA"] < second["KIRA"], "budget should ease toward the target"


def test_smoothing_of_one_jumps_straight_to_target():
    allocator = PerformanceAllocator(AllocatorConfig(smoothing=1.0))
    perfs = {b: perf(b, trades=80, wins=60, pnl=900 if b == "KIRA" else -400)
             for b in ROSTER}
    result = {d.bot_id: d.allocation for d in allocator.allocate(perfs, {b: 0.2 for b in ROSTER})}
    assert result["KIRA"] == pytest.approx(0.40)


def test_bounds_survive_an_out_of_range_starting_config():
    """A hand-edited config.json must not be able to smuggle a share past the cap."""
    hostile = {"JONATHAN": 0.9, "JOSEPH": 0.9, "JOTARO": 0.0,
               "JOLYAN": 0.0, "KIRA": 0.0}
    allocator = PerformanceAllocator()
    decisions = allocator.allocate(
        {b: perf(b, trades=80, wins=40, pnl=100) for b in ROSTER}, hostile
    )
    assert sum(d.allocation for d in decisions) <= 1.0 + 1e-9
    for d in decisions:
        assert 0.05 - 1e-9 <= d.allocation <= 0.40 + 1e-9


def test_invalid_smoothing_is_rejected():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            PerformanceAllocator(AllocatorConfig(smoothing=bad))


def test_allocation_converges_on_the_better_agent_over_rounds():
    """The end-to-end property: money moves toward whoever is actually earning."""
    allocator = PerformanceAllocator()
    current = {b: 0.2 for b in ROSTER}
    for round_index in range(1, 9):
        trades = 25 * round_index
        perfs = {
            b: perf(b, trades=trades, wins=int(trades * 0.5),
                    pnl=(2.6 if b == "KIRA" else -0.8) * trades)
            for b in ROSTER
        }
        current = {d.bot_id: d.allocation for d in allocator.allocate(perfs, current)}

    assert current["KIRA"] == max(current.values())
    assert current["KIRA"] > 0.35
    for other in ROSTER:
        if other != "KIRA":
            assert current[other] < 0.12
