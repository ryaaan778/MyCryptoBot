"""Volatility breakout — JOSEPH.

Waits for a Bollinger squeeze, then trades the direction the price escapes in,
requiring the move to clear an ATR-scaled threshold so ordinary noise doesn't
count as a breakout.
"""

from __future__ import annotations

import numpy as np

from ..models import Position, Signal, SignalAction
from .base import Series, Strategy
from .indicators import atr, bollinger, bollinger_bandwidth, last_valid


class VolatilityBreakoutStrategy(Strategy):
    name = "volatility_breakout"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.atr_period = self.int_param("volatility_atr_period", 7)
        self.atr_multiplier = self.param("volatility_atr_multiplier", 2.5)
        self.bb_period = self.int_param("volatility_bb_period", 15)
        self.bb_std = self.param("volatility_bb_std", 2.5)
        self.squeeze_lookback = self.int_param("volatility_squeeze_lookback", 40)
        self.min_bars = max(self.bb_period + self.squeeze_lookback + 5, 60)

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        upper, middle, lower = bollinger(series.close, self.bb_period, self.bb_std)
        width = bollinger_bandwidth(upper, middle, lower)
        atr_line = atr(series.high, series.low, series.close, self.atr_period)

        price = float(series.close[-1])
        a = last_valid(atr_line, 0.0)
        w = last_valid(width, 0.0)

        # Is the band unusually tight relative to the recent past? That's the coil.
        recent = width[-self.squeeze_lookback:]
        recent = recent[~np.isnan(recent)]
        if recent.size < 5:
            return self.hold(bot_id, symbol, "insufficient band history", price=price)
        squeeze_floor = float(np.percentile(recent, 35))
        was_squeezed = bool(width[-2] <= squeeze_floor) if not np.isnan(width[-2]) else False

        up_band, low_band = float(upper[-1]), float(lower[-1])
        threshold = a * (self.atr_multiplier / 10.0)
        readings = dict(
            atr=a, bb_upper=up_band, bb_lower=low_band, bb_width=w,
            squeeze_floor=squeeze_floor, price=price,
        )

        broke_up = price > up_band + threshold
        broke_down = price < low_band - threshold

        if broke_up:
            excess = (price - up_band) / a if a else 0.0
            confidence = min(1.0, 0.4 + 0.3 * excess + (0.2 if was_squeezed else 0.0))
            return self.signal(
                bot_id, symbol, SignalAction.LONG, confidence,
                f"broke above upper band by {excess:.2f} ATR"
                + (" out of a squeeze" if was_squeezed else ""),
                **readings,
            )
        if broke_down:
            excess = (low_band - price) / a if a else 0.0
            confidence = min(1.0, 0.4 + 0.3 * excess + (0.2 if was_squeezed else 0.0))
            return self.signal(
                bot_id, symbol, SignalAction.SHORT, confidence,
                f"broke below lower band by {excess:.2f} ATR"
                + (" out of a squeeze" if was_squeezed else ""),
                **readings,
            )

        # Breakouts that fail back inside the bands are the classic trap — leave.
        if position is not None and not np.isnan(middle[-1]):
            mid = float(middle[-1])
            if position.side.value == "LONG" and price < mid:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.6,
                    "breakout failed back below the mean", **readings,
                )
            if position.side.value == "SHORT" and price > mid:
                return self.signal(
                    bot_id, symbol, SignalAction.CLOSE, 0.6,
                    "breakdown failed back above the mean", **readings,
                )

        return self.hold(bot_id, symbol, "inside the bands", **readings)
