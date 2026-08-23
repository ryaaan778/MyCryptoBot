"""Mean reversion — KIRA.

Fades statistically stretched moves: a z-score extreme against the rolling mean,
confirmed by RSI. Patient, quiet, and — because it buys falling knives with the
highest leverage on the roster — the bot most likely to light up the risk meter.
"""

from __future__ import annotations

import numpy as np

from ..models import Position, Signal, SignalAction
from .base import Series, Strategy
from .indicators import last_valid, rsi, sma


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.period = self.int_param("mean_reversion_period", 20)
        self.entry_z = self.param("mean_reversion_entry_z", 2.0)
        self.exit_z = self.param("mean_reversion_exit_z", 0.4)
        self.rsi_period = self.int_param("mean_reversion_rsi_period", 14)
        self.rsi_low = self.param("mean_reversion_rsi_low", 32.0)
        self.rsi_high = self.param("mean_reversion_rsi_high", 68.0)
        self.min_bars = max(self.period + self.rsi_period + 10, 45)

    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal:
        mean_line = sma(series.close, self.period)
        window = series.close[-self.period:]
        std = float(window.std())
        mean = float(mean_line[-1]) if not np.isnan(mean_line[-1]) else float(window.mean())
        price = float(series.close[-1])

        if std <= 0:
            return self.hold(bot_id, symbol, "no dispersion to fade", price=price)

        z = (price - mean) / std
        r = last_valid(rsi(series.close, self.rsi_period), 50.0)
        readings = dict(zscore=z, mean=mean, stdev=std, rsi=r, price=price)

        # Exit first: once price has returned to the mean the trade is done.
        if position is not None and abs(z) <= self.exit_z:
            return self.signal(
                bot_id, symbol, SignalAction.CLOSE, 0.75,
                f"reverted to the mean (z {z:+.2f})", **readings,
            )

        stretch = (abs(z) - self.entry_z) / self.entry_z if self.entry_z else 0.0

        if z <= -self.entry_z and r <= self.rsi_low:
            confidence = min(1.0, 0.5 + 0.3 * stretch + 0.2 * (1.0 - r / 100.0))
            return self.signal(
                bot_id, symbol, SignalAction.LONG, confidence,
                f"stretched {z:+.2f}σ below the mean, RSI {r:.1f}", **readings,
            )
        if z >= self.entry_z and r >= self.rsi_high:
            confidence = min(1.0, 0.5 + 0.3 * stretch + 0.2 * (r / 100.0))
            return self.signal(
                bot_id, symbol, SignalAction.SHORT, confidence,
                f"stretched {z:+.2f}σ above the mean, RSI {r:.1f}", **readings,
            )

        return self.hold(bot_id, symbol, f"within range (z {z:+.2f})", **readings)
