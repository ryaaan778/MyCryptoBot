"""Strategy interface.

A strategy is a pure function of recent candles (plus the bot's current
position) to a :class:`Signal`. It never touches the exchange, the risk engine
or the event bus — that keeps every strategy trivially testable against a fixed
candle series.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..models import Candle, Position, PositionSide, Signal, SignalAction


@dataclass(frozen=True)
class Series:
    """Candle fields as numpy arrays, built once per evaluation."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_candles(cls, candles: list[Candle]) -> "Series":
        return cls(
            ts=np.array([c.ts for c in candles], dtype=np.int64),
            open=np.array([c.open for c in candles], dtype=np.float64),
            high=np.array([c.high for c in candles], dtype=np.float64),
            low=np.array([c.low for c in candles], dtype=np.float64),
            close=np.array([c.close for c in candles], dtype=np.float64),
            volume=np.array([c.volume for c in candles], dtype=np.float64),
        )

    def __len__(self) -> int:
        return int(self.close.size)


class Strategy(abc.ABC):
    name: str = "base"
    min_bars: int = 50

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params = dict(params or {})

    def param(self, key: str, default: float) -> float:
        value = self.params.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def int_param(self, key: str, default: int) -> int:
        return max(1, int(round(self.param(key, float(default)))))

    # ---- API ---------------------------------------------------------------

    def evaluate(
        self,
        bot_id: str,
        symbol: str,
        candles: list[Candle],
        position: Position | None = None,
    ) -> Signal:
        if len(candles) < self.min_bars:
            return self.hold(bot_id, symbol, f"warming up ({len(candles)}/{self.min_bars} bars)")
        return self.compute(bot_id, symbol, Series.from_candles(candles), position)

    @abc.abstractmethod
    def compute(
        self, bot_id: str, symbol: str, series: Series, position: Position | None
    ) -> Signal: ...

    # ---- helpers -----------------------------------------------------------

    def hold(self, bot_id: str, symbol: str, reason: str = "", **indicators: float) -> Signal:
        return Signal(
            bot_id=bot_id, symbol=symbol, action=SignalAction.HOLD,
            confidence=0.0, reason=reason, strategy=self.name,
            indicators=_clean(indicators),
        )

    def signal(
        self,
        bot_id: str,
        symbol: str,
        action: SignalAction,
        confidence: float,
        reason: str,
        **indicators: float,
    ) -> Signal:
        return Signal(
            bot_id=bot_id, symbol=symbol, action=action,
            confidence=max(0.0, min(1.0, confidence)), reason=reason,
            strategy=self.name, indicators=_clean(indicators),
        )

    @staticmethod
    def opposes(position: Position | None, action: SignalAction) -> bool:
        """True when ``action`` points against an open position."""
        if position is None:
            return False
        return (
            (position.side is PositionSide.LONG and action is SignalAction.SHORT)
            or (position.side is PositionSide.SHORT and action is SignalAction.LONG)
        )


def _clean(values: dict[str, float]) -> dict[str, float]:
    """Drop nan/inf so the payload is always valid JSON."""
    out: dict[str, float] = {}
    for key, value in values.items():
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[key] = round(f, 6)
    return out
