"""Synthetic market feed.

Geometric Brownian motion with a Markov regime switch, so the tape shows
trends, ranges and volatility bursts rather than uniform noise — the strategies
have something real to disagree about. Seeded, so a given seed always produces
the same tape and tests are deterministic.

The simulated clock runs faster than the wall clock (``time_scale``) so a 5m
candle closes every few real seconds. Without that a demo would look frozen.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..models import Candle, Ticker, now_ms
from .base import TIMEFRAME_MS, MarketDataProvider

logger = logging.getLogger("jojo.market.sim")

# Plausible starting levels so the world's numbers look like a real book.
BASE_PRICES: dict[str, float] = {
    "BTC/USDT": 64_250.0,
    "ETH/USDT": 3_120.0,
    "BNB/USDT": 585.0,
    "SOL/USDT": 146.5,
    "ADA/USDT": 0.452,
    "XRP/USDT": 0.615,
    "DOGE/USDT": 0.158,
}

# regime -> (annualised drift, annualised vol, mean dwell time in sim-seconds)
REGIMES: dict[str, tuple[float, float, float]] = {
    "TREND_UP": (0.85, 0.55, 2_400.0),
    "TREND_DOWN": (-0.80, 0.60, 2_100.0),
    "RANGE": (0.0, 0.30, 3_000.0),
    "VOLATILE": (0.05, 1.35, 1_200.0),
}
REGIME_NAMES = list(REGIMES)
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass
class _Series:
    """The evolving state of one symbol."""

    symbol: str
    price: float
    rng: np.random.Generator
    regime: str = "RANGE"
    regime_elapsed: float = 0.0
    session_open: float = 0.0
    session_volume: float = 0.0
    # timeframe -> list of closed candles (oldest first)
    history: dict[str, list[Candle]] = field(default_factory=dict)
    # timeframe -> the candle currently forming
    forming: dict[str, Candle] = field(default_factory=dict)

    def maybe_switch_regime(self, dt: float) -> bool:
        self.regime_elapsed += dt
        dwell = REGIMES[self.regime][2]
        # Exponential hazard: probability of switching in this slice of time.
        if self.rng.random() < 1.0 - math.exp(-dt / dwell):
            choices = [r for r in REGIME_NAMES if r != self.regime]
            self.regime = str(self.rng.choice(choices))
            self.regime_elapsed = 0.0
            return True
        return False

    def step(self, dt: float) -> float:
        mu, sigma, _ = REGIMES[self.regime]
        dt_years = dt / SECONDS_PER_YEAR
        shock = self.rng.standard_normal()
        drift = (mu - 0.5 * sigma * sigma) * dt_years
        diffusion = sigma * math.sqrt(max(dt_years, 1e-12)) * shock
        self.price = max(self.price * math.exp(drift + diffusion), 1e-8)
        return self.price


class SimulatedProvider(MarketDataProvider):
    name = "simulated"
    is_synthetic = True

    def __init__(
        self,
        seed: int = 20250328,
        *,
        time_scale: float = 60.0,
        tick_hz: float = 8.0,
        history_bars: int = 240,
    ) -> None:
        self.seed = seed
        self.time_scale = time_scale
        self.tick_interval = 1.0 / tick_hz
        self.history_bars = history_bars
        self._series: dict[str, _Series] = {}
        self._tickers: dict[str, Ticker] = {}
        self._timeframes: list[str] = ["5m"]
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ---- lifecycle ---------------------------------------------------------

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        self._timeframes = [tf for tf in timeframes if tf in TIMEFRAME_MS] or ["5m"]
        for index, symbol in enumerate(symbols):
            rng = np.random.default_rng(self.seed + index * 7919)
            base = BASE_PRICES.get(symbol, 100.0)
            series = _Series(symbol=symbol, price=base, rng=rng)
            series.regime = str(rng.choice(REGIME_NAMES))
            self._backfill(series)
            series.session_open = series.history[self._timeframes[0]][0].open
            self._series[symbol] = series
            self._publish_ticker(series)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="sim-market")
        logger.info(
            "simulated feed started: %d symbols, timeframes=%s, seed=%d, %gx clock",
            len(symbols), self._timeframes, self.seed, self.time_scale,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ---- history -----------------------------------------------------------

    def _backfill(self, series: _Series) -> None:
        """Walk the price backwards-in-time forwards so history joins the live price."""
        now = now_ms()
        for tf in self._timeframes:
            step_ms = TIMEFRAME_MS[tf]
            bars = self.history_bars
            start_price = series.price
            # Rewind: generate a path of `bars` candles ending at the current price.
            sub_steps = 12
            dt = step_ms / 1000.0 / sub_steps
            path_rng = np.random.default_rng(self.seed + hash(f"{series.symbol}{tf}") % 100_000)
            temp = _Series(series.symbol, start_price, path_rng, regime=series.regime)
            candles: list[Candle] = []
            open_ts = now - bars * step_ms
            for i in range(bars):
                o = temp.price
                highs, lows = o, o
                vol = 0.0
                for _ in range(sub_steps):
                    temp.maybe_switch_regime(dt)
                    p = temp.step(dt)
                    highs = max(highs, p)
                    lows = min(lows, p)
                    vol += abs(path_rng.normal(1.0, 0.35)) * (o / 1000.0 + 1.0)
                candles.append(
                    Candle(
                        ts=open_ts + i * step_ms,
                        open=round(o, 8),
                        high=round(highs, 8),
                        low=round(lows, 8),
                        close=round(temp.price, 8),
                        volume=round(vol, 4),
                    )
                )
            series.history[tf] = candles
            series.price = temp.price
            series.forming[tf] = Candle(
                ts=now - (now % step_ms),
                open=series.price, high=series.price,
                low=series.price, close=series.price, volume=0.0,
            )

    # ---- streaming ---------------------------------------------------------

    async def _run(self) -> None:
        last = time.monotonic()
        while self._running:
            await asyncio.sleep(self.tick_interval)
            real_dt = time.monotonic() - last
            last = time.monotonic()
            sim_dt = real_dt * self.time_scale
            for series in self._series.values():
                if series.maybe_switch_regime(sim_dt):
                    logger.debug("%s regime -> %s", series.symbol, series.regime)
                price = series.step(sim_dt)
                self._roll_candles(series, price, sim_dt)
                self._publish_ticker(series)

    def _roll_candles(self, series: _Series, price: float, sim_dt: float) -> None:
        ts = now_ms()
        volume_slice = abs(series.rng.normal(1.0, 0.4)) * sim_dt / 60.0 * (price / 1000.0 + 1.0)
        series.session_volume += volume_slice
        for tf in self._timeframes:
            step_ms = TIMEFRAME_MS[tf]
            # The simulated clock advances faster, so candle boundaries are
            # derived from scaled elapsed time rather than the wall clock.
            forming = series.forming[tf]
            forming.close = price
            forming.high = max(forming.high, price)
            forming.low = min(forming.low, price)
            forming.volume += volume_slice
            elapsed_sim = (ts - forming.ts) * self.time_scale
            if elapsed_sim >= step_ms:
                closed = Candle(**forming.model_dump())
                closed.open = round(closed.open, 8)
                closed.close = round(closed.close, 8)
                history = series.history[tf]
                history.append(closed)
                if len(history) > self.history_bars * 2:
                    del history[: len(history) - self.history_bars]
                series.forming[tf] = Candle(
                    ts=ts, open=price, high=price, low=price, close=price, volume=0.0
                )
                self._emit_candle(series.symbol, tf, closed)

    def _publish_ticker(self, series: _Series) -> None:
        price = series.price
        # Spread widens with the regime's volatility, like a real book.
        vol_factor = REGIMES[series.regime][1]
        half_spread = price * (0.00002 + 0.00004 * vol_factor)
        change = ((price / series.session_open) - 1.0) * 100.0 if series.session_open else 0.0
        ticker = Ticker(
            symbol=series.symbol,
            price=round(price, 8),
            bid=round(price - half_spread, 8),
            ask=round(price + half_spread, 8),
            volume_24h=round(series.session_volume, 2),
            change_24h_pct=round(change, 3),
        )
        self._tickers[series.symbol] = ticker
        self._emit_ticker(ticker)

    # ---- reads -------------------------------------------------------------

    def candles(self, symbol: str, timeframe: str, limit: int = 200) -> list[Candle]:
        series = self._series.get(symbol)
        if not series:
            return []
        history = series.history.get(timeframe, [])
        return history[-limit:]

    def ticker(self, symbol: str) -> Ticker | None:
        return self._tickers.get(symbol)

    def regime(self, symbol: str) -> str:
        series = self._series.get(symbol)
        return series.regime if series else "UNKNOWN"
