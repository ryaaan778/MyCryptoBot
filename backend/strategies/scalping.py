"""Scalping — JOTARO.

Fast EMA cross confirmed by RSI. Short holds, tight stops, many attempts.
Parameters come straight from ``config.json``'s ``scalping_*`` block.
"""

from __future__ import annotations

from ..models import Position, Signal, SignalAction
from .base import Series, Strategy
from .indicators import ema, last_valid, rsi


class ScalpingStrategy(Strategy):
    name = "scalping"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.fast = self.int_param("scalping_ema_short", 3)
        self.slow = self.int_param("scalping_ema_long", 15)
        self.rsi_period = self.int_param("scalping_rsi_period", 7)
        self.oversold = self.param("scalping_rsi_oversold", 20.0)
        self.overbought = self.param("scalping_rsi_overbought", 80.0)
        self.min_bars = max(self.slow + 5, self.rsi_period + 5, 25)

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        fast_line = ema(series.close, self.fast)
        slow_line = ema(series.close, self.slow)
        rsi_line = rsi(series.close, self.rsi_period)

        fast_now, fast_prev = fast_line[-1], fast_line[-2]
        slow_now, slow_prev = slow_line[-1], slow_line[-2]
        r = last_valid(rsi_line, 50.0)
        readings = dict(ema_fast=fast_now, ema_slow=slow_now, rsi=r, price=series.close[-1])

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        # Separation between the EMAs scales confidence: a wide, decisive cross
        # is worth more than the two lines grazing each other.
        spread = abs(fast_now - slow_now) / series.close[-1] if series.close[-1] else 0.0
        strength = min(1.0, spread * 400.0)

        if crossed_up and r < self.overbought:
            confidence = 0.45 + 0.35 * strength + 0.2 * (1.0 - r / 100.0)
            return self.signal(
                bot_id, symbol, SignalAction.LONG, confidence,
                f"EMA{self.fast} crossed above EMA{self.slow} with RSI {r:.1f}", **readings,
            )
        if crossed_down and r > self.oversold:
            confidence = 0.45 + 0.35 * strength + 0.2 * (r / 100.0)
            return self.signal(
                bot_id, symbol, SignalAction.SHORT, confidence,
                f"EMA{self.fast} crossed below EMA{self.slow} with RSI {r:.1f}", **readings,
            )

        # Exhaustion exit: RSI has run to an extreme while we are still in.
        if position is not None:
            if position.side.value == "LONG" and r >= self.overbought:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.7,
                    f"RSI {r:.1f} overbought — banking the scalp", **readings,
                )
            if position.side.value == "SHORT" and r <= self.oversold:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.7,
                    f"RSI {r:.1f} oversold — banking the scalp", **readings,
                )

        return self.hold(bot_id, symbol, f"no cross (RSI {r:.1f})", **readings)
