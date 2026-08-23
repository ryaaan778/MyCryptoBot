"""Momentum — JONATHAN.

MACD cross gated by RSI's position relative to its midline. Steady, slower,
fewer trades: the strategy that holds the line.
"""

from __future__ import annotations

import numpy as np

from ..models import Position, Signal, SignalAction
from .base import Series, Strategy
from .indicators import last_valid, macd, rsi


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.fast = self.int_param("momentum_fast_period", 8)
        self.slow = self.int_param("momentum_slow_period", 21)
        self.signal_period = self.int_param("momentum_signal_period", 7)
        self.rsi_period = self.int_param("momentum_rsi_period", 7)
        self.rsi_threshold = self.param("momentum_rsi_threshold", 50.0)
        self.min_bars = max(self.slow + self.signal_period + 10, 45)

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        macd_line, signal_line, hist = macd(
            series.close, self.fast, self.slow, self.signal_period
        )
        rsi_line = rsi(series.close, self.rsi_period)

        if np.isnan(hist[-1]) or np.isnan(hist[-2]):
            return self.hold(bot_id, symbol, "MACD not formed yet")

        r = last_valid(rsi_line, 50.0)
        price = float(series.close[-1])
        readings = dict(
            macd=float(macd_line[-1]), macd_signal=float(signal_line[-1]),
            histogram=float(hist[-1]), rsi=r, price=price,
        )

        crossed_up = hist[-2] <= 0 < hist[-1]
        crossed_down = hist[-2] >= 0 > hist[-1]

        # Histogram magnitude relative to price gives a scale-free strength.
        strength = min(1.0, abs(float(hist[-1])) / price * 2_000.0) if price else 0.0

        if crossed_up and r > self.rsi_threshold:
            confidence = 0.5 + 0.3 * strength + 0.2 * min(1.0, (r - self.rsi_threshold) / 30.0)
            return self.signal(
                bot_id, symbol, SignalAction.LONG, confidence,
                f"MACD crossed up, RSI {r:.1f} above {self.rsi_threshold:.0f}", **readings,
            )
        if crossed_down and r < self.rsi_threshold:
            confidence = 0.5 + 0.3 * strength + 0.2 * min(1.0, (self.rsi_threshold - r) / 30.0)
            return self.signal(
                bot_id, symbol, SignalAction.SHORT, confidence,
                f"MACD crossed down, RSI {r:.1f} below {self.rsi_threshold:.0f}", **readings,
            )

        # Momentum rolling over against the position is the exit trigger.
        if position is not None:
            if position.side.value == "LONG" and crossed_down:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.65,
                    "momentum rolled over", **readings,
                )
            if position.side.value == "SHORT" and crossed_up:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.65,
                    "momentum turned back up", **readings,
                )

        return self.hold(bot_id, symbol, f"no MACD cross (hist {hist[-1]:+.4f})", **readings)
