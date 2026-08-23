"""Provider selection with honest degradation.

``provider: auto`` tries Binance first and falls back to the synthetic feed when
the exchange is unreachable. The fallback is *reported*, not hidden — the caller
gets a note that the HUD surfaces as a DEGRADED badge, because a trader must
never mistake synthetic prices for real ones.
"""

from __future__ import annotations

import logging

from ..config import Settings
from .base import MarketDataProvider
from .simulated import SimulatedProvider

logger = logging.getLogger("jojo.market")


async def create_provider(
    settings: Settings, symbols: list[str], timeframes: list[str]
) -> tuple[MarketDataProvider, bool, str]:
    """Return ``(provider, degraded, note)`` with the provider already started."""
    choice = (settings.provider or "auto").lower()
    history = settings.server.candle_history

    if choice == "simulated":
        provider = SimulatedProvider(seed=settings.seed, history_bars=history)
        await provider.start(symbols, timeframes)
        return provider, False, "synthetic market feed (explicitly selected)"

    if choice in ("auto", "binance"):
        from .binance import BinanceProvider

        provider = BinanceProvider(testnet=settings.exchange.testnet, history_bars=history)
        try:
            await provider.start(symbols, timeframes)
            return provider, False, "live Binance market data"
        except Exception as exc:
            await provider.stop()
            note = f"Binance unreachable ({type(exc).__name__}: {exc})"
            if choice == "binance":
                # Explicitly asked for Binance: still fall back rather than
                # leaving the system blind, but say so loudly.
                logger.error("%s — falling back to synthetic feed", note)
            else:
                logger.warning("%s — using synthetic feed", note)
            fallback = SimulatedProvider(seed=settings.seed, history_bars=history)
            await fallback.start(symbols, timeframes)
            return fallback, True, f"{note}; showing SYNTHETIC prices"

    raise ValueError(f"unknown provider {settings.provider!r} (expected auto|binance|simulated)")
