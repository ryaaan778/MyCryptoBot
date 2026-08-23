"""Live Binance market data.

REST (via ccxt) for candle history and candle closes, and a raw Binance
combined WebSocket stream for book tickers — that gives real bid/ask rather
than a polled last price.

This provider cannot be exercised from a sandbox with no exchange access; it is
written to fail loudly on connect so ``factory.create_provider`` can fall back
to the synthetic feed and tell the UI it did.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from ..config import to_display_symbol
from ..models import Candle, Ticker, now_ms
from .base import TIMEFRAME_MS, MarketDataProvider

logger = logging.getLogger("jojo.market.binance")

WS_ENDPOINT = "wss://stream.binance.com:9443/stream"
WS_ENDPOINT_TESTNET = "wss://stream.testnet.binance.vision/stream"


class BinanceProvider(MarketDataProvider):
    name = "binance"
    is_synthetic = False

    def __init__(
        self,
        *,
        testnet: bool = True,
        history_bars: int = 240,
        kline_poll_sec: float = 15.0,
        connect_timeout: float = 12.0,
    ) -> None:
        self.testnet = testnet
        self.history_bars = history_bars
        self.kline_poll_sec = kline_poll_sec
        self.connect_timeout = connect_timeout
        self._exchange = None
        self._symbols: list[str] = []
        self._timeframes: list[str] = []
        self._history: dict[tuple[str, str], list[Candle]] = {}
        self._tickers: dict[str, Ticker] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        import ccxt.async_support as ccxt  # imported lazily: heavy module

        self._symbols = symbols
        self._timeframes = [tf for tf in timeframes if tf in TIMEFRAME_MS] or ["5m"]
        self._exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        if self.testnet:
            self._exchange.set_sandbox_mode(True)

        # Fail fast and loudly — the factory needs a clear signal to fall back.
        try:
            await asyncio.wait_for(self._load_history(), timeout=self.connect_timeout)
        except Exception as exc:
            await self._shutdown_exchange()
            raise ConnectionError(f"Binance history fetch failed: {exc}") from exc

        self._running = True
        self._tasks.append(asyncio.create_task(self._ticker_stream(), name="binance-ws"))
        self._tasks.append(asyncio.create_task(self._kline_poller(), name="binance-klines"))
        logger.info("binance feed started: %s", ", ".join(symbols))

    async def _load_history(self) -> None:
        assert self._exchange is not None
        for symbol in self._symbols:
            for tf in self._timeframes:
                raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=self.history_bars)
                self._history[(symbol, tf)] = [
                    Candle(ts=int(r[0]), open=float(r[1]), high=float(r[2]),
                           low=float(r[3]), close=float(r[4]), volume=float(r[5]))
                    for r in raw
                ]
            last = self._history[(symbol, self._timeframes[0])][-1]
            self._tickers[symbol] = Ticker(
                symbol=symbol, price=last.close, bid=last.close, ask=last.close
            )

    async def _ticker_stream(self) -> None:
        import websockets

        streams = "/".join(f"{to_display_symbol(s).lower()}@bookTicker" for s in self._symbols)
        url = f"{WS_ENDPOINT_TESTNET if self.testnet else WS_ENDPOINT}?streams={streams}"
        by_display = {to_display_symbol(s): s for s in self._symbols}
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    backoff = 1.0
                    logger.info("binance ticker stream connected")
                    while self._running:
                        payload = json.loads(await ws.recv())
                        data = payload.get("data", payload)
                        display = data.get("s")
                        symbol = by_display.get(display)
                        if not symbol:
                            continue
                        bid, ask = float(data["b"]), float(data["a"])
                        previous = self._tickers.get(symbol)
                        ticker = Ticker(
                            symbol=symbol,
                            price=round((bid + ask) / 2.0, 8),
                            bid=bid,
                            ask=ask,
                            volume_24h=previous.volume_24h if previous else 0.0,
                            change_24h_pct=previous.change_24h_pct if previous else 0.0,
                            ts=now_ms(),
                        )
                        self._tickers[symbol] = ticker
                        self._emit_ticker(ticker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("binance ticker stream dropped (%s); retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _kline_poller(self) -> None:
        assert self._exchange is not None
        while self._running:
            await asyncio.sleep(self.kline_poll_sec)
            for symbol in self._symbols:
                for tf in self._timeframes:
                    try:
                        raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=3)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug("kline poll failed for %s %s: %s", symbol, tf, exc)
                        continue
                    history = self._history.setdefault((symbol, tf), [])
                    known = history[-1].ts if history else 0
                    for row in raw:
                        ts = int(row[0])
                        if ts <= known:
                            continue
                        candle = Candle(
                            ts=ts, open=float(row[1]), high=float(row[2]),
                            low=float(row[3]), close=float(row[4]), volume=float(row[5]),
                        )
                        history.append(candle)
                        if len(history) > self.history_bars * 2:
                            del history[: len(history) - self.history_bars]
                        self._emit_candle(symbol, tf, candle)

    async def _shutdown_exchange(self) -> None:
        if self._exchange is not None:
            with contextlib.suppress(Exception):
                await self._exchange.close()
            self._exchange = None

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        await self._shutdown_exchange()

    def candles(self, symbol: str, timeframe: str, limit: int = 200) -> list[Candle]:
        return self._history.get((symbol, timeframe), [])[-limit:]

    def ticker(self, symbol: str) -> Ticker | None:
        return self._tickers.get(symbol)
