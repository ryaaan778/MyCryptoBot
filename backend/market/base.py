"""Market data provider interface.

Two implementations exist: a real Binance feed and a synthetic one. The engine
is written against this interface only, so which feed is behind it never changes
the trading logic.
"""

from __future__ import annotations

import abc
from typing import Awaitable, Callable

from ..models import Candle, Ticker

TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

TickerCallback = Callable[[Ticker], None | Awaitable[None]]


class MarketDataProvider(abc.ABC):
    """Supplies candles and a live price stream for a set of symbols."""

    name: str = "base"
    is_synthetic: bool = False

    @abc.abstractmethod
    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        """Prepare history and begin streaming. Raises on unrecoverable failure."""

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def candles(self, symbol: str, timeframe: str, limit: int = 200) -> list[Candle]:
        """Most recent closed candles, oldest first."""

    @abc.abstractmethod
    def ticker(self, symbol: str) -> Ticker | None:
        """Latest known ticker, or None if nothing has arrived yet."""

    def on_ticker(self, callback: TickerCallback) -> None:
        self._ticker_callback = callback  # type: ignore[attr-defined]

    def on_candle(self, callback: Callable[[str, str, Candle], None]) -> None:
        self._candle_callback = callback  # type: ignore[attr-defined]

    # ---- helpers for subclasses -------------------------------------------

    def _emit_ticker(self, ticker: Ticker) -> None:
        cb = getattr(self, "_ticker_callback", None)
        if cb is not None:
            cb(ticker)

    def _emit_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        cb = getattr(self, "_candle_callback", None)
        if cb is not None:
            cb(symbol, timeframe, candle)
