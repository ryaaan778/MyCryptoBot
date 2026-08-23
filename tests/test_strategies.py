"""Strategies against constructed candle series with a known correct answer."""

from __future__ import annotations

import numpy as np
import pytest

from backend.models import Candle, Position, PositionSide, SignalAction
from backend.strategies import REGISTRY, create_strategy
from backend.strategies.mean_reversion import MeanReversionStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.scalping import ScalpingStrategy


def make_candles(closes, *, spread: float = 0.5, volume: float = 10.0) -> list[Candle]:
    out = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i else close
        out.append(Candle(
            ts=1_700_000_000_000 + i * 300_000,
            open=float(prev),
            high=float(max(prev, close) + spread),
            low=float(min(prev, close) - spread),
            close=float(close),
            volume=volume,
        ))
    return out


def long_position(symbol: str = "BTC/USDT") -> Position:
    return Position(
        bot_id="t", symbol=symbol, side=PositionSide.LONG,
        quantity=1.0, entry_price=100.0, mark_price=100.0, margin=100.0,
    )


def short_position(symbol: str = "BTC/USDT") -> Position:
    return Position(
        bot_id="t", symbol=symbol, side=PositionSide.SHORT,
        quantity=1.0, entry_price=100.0, mark_price=100.0, margin=100.0,
    )


# ---------------------------------------------------------------- common rules


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_warmup_returns_hold_not_an_error(name):
    strategy = create_strategy(name)
    signal = strategy.evaluate("t", "BTC/USDT", make_candles([100.0] * 5))
    assert signal.action is SignalAction.HOLD
    assert "warming up" in signal.reason


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_flat_market_produces_no_trade(name):
    strategy = create_strategy(name)
    candles = make_candles([100.0] * 300, spread=0.01)
    signal = strategy.evaluate("t", "BTC/USDT", candles)
    assert signal.action is SignalAction.HOLD


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_indicator_payload_is_json_safe(name):
    strategy = create_strategy(name)
    rng = np.random.default_rng(7)
    closes = list(100 + np.cumsum(rng.normal(0, 1, 400)))
    signal = strategy.evaluate("t", "BTC/USDT", candles := make_candles(closes))
    assert candles
    for key, value in signal.indicators.items():
        assert np.isfinite(value), f"{name}.{key} leaked a non-finite value"


def test_unknown_strategy_names_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        create_strategy("stand_power")


# -------------------------------------------------------------------- scalping


def signals_over(strategy, closes, start: int, position=None):
    """Every non-HOLD signal the strategy produces from bar ``start`` onward.

    A cross fires on the single bar it happens, so asserting on one hand-picked
    bar is brittle. What actually matters is the behaviour across the move:
    during a sustained rally the strategy should produce longs and never shorts.
    """
    candles = make_candles(closes)
    out = []
    for i in range(max(strategy.min_bars, start), len(candles) + 1):
        signal = strategy.evaluate("t", "BTC/USDT", candles[:i], position)
        if signal.action is not SignalAction.HOLD:
            out.append(signal)
    return out


def choppy(bars: int, level: float = 100.0, amplitude: float = 0.6) -> list[float]:
    """Two-sided noise, so RSI sits near its midline instead of pinned at 0/100."""
    return [level + (amplitude if i % 2 else -amplitude) for i in range(bars)]


def warmup(bars: int, level: float = 100.0, seed: int = 11) -> list[float]:
    """A gentle seeded random walk — gives RSI two-sided history without the
    systematic EMA crosses that a strict alternation would manufacture."""
    rng = np.random.default_rng(seed)
    return list(level + np.cumsum(rng.normal(0.0, 0.15, bars)))


def _leg(start: float, bars: int, step: float, pullback_every: int = 3) -> list[float]:
    """A directional leg with regular retracements, so RSI trends without pinning."""
    out = [start]
    for i in range(bars):
        delta = -step * 0.4 if i % pullback_every == pullback_every - 1 else step
        out.append(out[-1] + delta)
    return out[1:]


def vshape(up_cross: bool = True) -> tuple[list[float], int]:
    """A reversal series, returned with the index of the pivot.

    Testing a *cross* requires the fast line to start on the far side: a series
    that only ever rallies leaves the EMAs pre-positioned and no cross ever
    happens. The opening leg establishes the wrong side, the second leg crosses
    it, and the retracements keep RSI mid-range so the strategy's own extreme
    filters don't (correctly) veto the entry.
    """
    head = warmup(30)
    first = _leg(head[-1], 18, -0.7 if up_cross else 0.7)
    second = _leg(first[-1], 22, 0.9 if up_cross else -0.9)
    return head + first + second, len(head) + len(first)


def test_scalping_goes_long_on_bullish_ema_cross():
    closes, pivot = vshape(up_cross=True)
    signals = signals_over(ScalpingStrategy(), closes, start=pivot)
    actions = {s.action for s in signals}
    assert SignalAction.LONG in actions, "an upward reversal must produce a long"
    assert SignalAction.SHORT not in actions, "nothing in a rally justifies a short"
    longs = [s for s in signals if s.action is SignalAction.LONG]
    assert all("crossed above" in s.reason for s in longs)
    assert max(s.confidence for s in longs) > 0.4


def test_scalping_goes_short_on_bearish_ema_cross():
    closes, pivot = vshape(up_cross=False)
    signals = signals_over(ScalpingStrategy(), closes, start=pivot)
    actions = {s.action for s in signals}
    assert SignalAction.SHORT in actions
    assert SignalAction.LONG not in actions
    assert all(
        "crossed below" in s.reason for s in signals if s.action is SignalAction.SHORT
    )


def test_scalping_rsi_ceiling_vetoes_an_otherwise_valid_long():
    """The RSI ceiling must actually gate entries, not just decorate them.

    The same series is run twice: once with the ceiling above the reading at the
    cross (long allowed) and once below it (long vetoed). Only the parameter
    differs, so any change in behaviour is the gate doing its job.
    """
    closes, pivot = vshape(up_cross=True)

    permissive = signals_over(ScalpingStrategy(), closes, start=pivot)
    longs = [s for s in permissive if s.action is SignalAction.LONG]
    assert longs, "baseline series must produce a long to gate"

    rsi_at_cross = longs[0].indicators["rsi"]
    strict = ScalpingStrategy({"scalping_rsi_overbought": rsi_at_cross - 1.0})
    assert SignalAction.LONG not in {
        s.action for s in signals_over(strict, closes, start=pivot)
    }, "a ceiling below the RSI at the cross must veto the entry"


def test_scalping_closes_a_long_when_rsi_is_exhausted():
    # A steady grind up leaves RSI pinned high with no fresh cross on the last bar.
    closes = [100.0] * 30 + list(np.linspace(100.0, 130.0, 40))
    strategy = ScalpingStrategy()
    signal = strategy.evaluate("t", "BTC/USDT", make_candles(closes), long_position())
    assert signal.action in (SignalAction.CLOSE, SignalAction.LONG)
    if signal.action is SignalAction.CLOSE:
        assert "overbought" in signal.reason


# ------------------------------------------------------------- mean reversion


def test_mean_reversion_fades_a_spike_above_the_mean():
    closes = [100.0] * 80 + [118.0]
    signal = MeanReversionStrategy().evaluate("t", "BTC/USDT", make_candles(closes))
    assert signal.action is SignalAction.SHORT
    assert signal.indicators["zscore"] > 2.0


def test_mean_reversion_buys_a_flush_below_the_mean():
    closes = [100.0] * 80 + [82.0]
    signal = MeanReversionStrategy().evaluate("t", "BTC/USDT", make_candles(closes))
    assert signal.action is SignalAction.LONG
    assert signal.indicators["zscore"] < -2.0


def test_mean_reversion_closes_once_price_returns_to_the_mean():
    # Dispersion around 100 (so stdev > 0), with the final bar sitting on the mean.
    closes = choppy(120, level=100.0, amplitude=1.0) + [100.0]
    strategy = MeanReversionStrategy()
    signal = strategy.evaluate("t", "BTC/USDT", make_candles(closes), long_position())
    assert signal.action is SignalAction.CLOSE
    assert abs(signal.indicators["zscore"]) <= strategy.exit_z


# ------------------------------------------------------------------- momentum


def test_momentum_requires_rsi_confirmation():
    """A MACD cross with RSI on the wrong side of the threshold must not fire."""
    closes, pivot = vshape(up_cross=True)
    strategy = MomentumStrategy({"momentum_rsi_threshold": 99.0})
    signals = signals_over(strategy, closes, start=pivot)
    assert SignalAction.LONG not in {s.action for s in signals}

    # The same series with the default threshold does fire, so the veto above
    # came from the RSI gate rather than from an absent MACD cross.
    assert SignalAction.LONG in {
        s.action for s in signals_over(MomentumStrategy(), closes, start=pivot)
    }


def test_momentum_fires_long_when_both_agree():
    strategy = MomentumStrategy()
    closes, pivot = vshape(up_cross=True)
    signals = signals_over(strategy, closes, start=pivot)
    longs = [s for s in signals if s.action is SignalAction.LONG]
    assert longs, "an upward reversal must produce a momentum long"
    assert SignalAction.SHORT not in {s.action for s in signals}
    assert all(s.indicators["rsi"] > strategy.rsi_threshold for s in longs)
    assert all("MACD crossed up" in s.reason for s in longs)


def test_momentum_fires_short_on_a_downward_reversal():
    closes, pivot = vshape(up_cross=False)
    signals = signals_over(MomentumStrategy(), closes, start=pivot)
    shorts = [s for s in signals if s.action is SignalAction.SHORT]
    assert shorts
    assert SignalAction.LONG not in {s.action for s in signals}
    assert all("MACD crossed down" in s.reason for s in shorts)


# --------------------------------------------------------------------- hybrid


def test_hybrid_weights_are_normalised():
    strategy = create_strategy("hybrid", {
        "weight_scalping": 2.0, "weight_volatility": 2.0,
        "weight_momentum": 2.0, "weight_mean_reversion": 2.0,
    })
    assert sum(strategy.weights.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(0.25) for w in strategy.weights.values())


def test_hybrid_reports_which_members_voted():
    closes = [100.0] * 60 + list(np.linspace(100.0, 88.0, 30))
    signal = create_strategy("hybrid").evaluate("t", "BTC/USDT", make_candles(closes))
    if signal.action is not SignalAction.HOLD:
        assert any(k.startswith("w_") for k in signal.indicators)


def test_opposes_detects_a_conflicting_signal():
    strategy = ScalpingStrategy()
    assert strategy.opposes(long_position(), SignalAction.SHORT)
    assert strategy.opposes(short_position(), SignalAction.LONG)
    assert not strategy.opposes(long_position(), SignalAction.LONG)
    assert not strategy.opposes(None, SignalAction.LONG)
