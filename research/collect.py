"""Dataset collection.

Two kinds of source, deliberately kept far apart:

* :func:`collect_exchange` — real market history from **any** ccxt venue.
  Paginated and resumable, because multi-year 5m history is hundreds of requests
  and losing an hour of downloading to a dropped connection is miserable.
  **This cannot run in a sandbox with no exchange access; it is for your
  machine.**
* :func:`generate_synthetic` — deterministic bars from the same regime-switching
  GBM the live simulator uses, reusing ``SimulatedProvider``'s model rather than
  reimplementing it. Useful for testing the harness itself, and never a
  substitute for real-market evidence.

The collector is venue-generic on purpose. ccxt already gives one uniform
``fetch_ohlcv`` across ~100 exchanges, so adding Bybit or OKX should be an
argument, not a new function — and the same policy evaluated on several venues
is a free robustness check. Page limits and rate limits differ per venue, so
those are read from the exchange object rather than hard-coded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from pathlib import Path

import numpy as np

from backend.config import to_ccxt_symbol
from backend.market.base import TIMEFRAME_MS
from backend.market.simulated import BASE_PRICES, REGIME_NAMES, REGIMES, _Series

from .datasets import Dataset, DataSource

logger = logging.getLogger("research.collect")

# Most venues cap OHLCV at 500-1500 rows per request; ccxt reports the real
# limit per exchange where it knows it, and this is the fallback when it does not.
DEFAULT_PAGE_LIMIT = 1000


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------


def generate_synthetic(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    bars: int = 20_000,
    seed: int = 20250328,
    *,
    start_ts: int | None = None,
    sub_steps: int = 12,
    base_price: float | None = None,
) -> Dataset:
    """Deterministic synthetic OHLCV.

    Each bar is built from ``sub_steps`` intra-bar price steps so highs and lows
    are the real extremes of a path rather than invented spreads around the
    close — which matters, because stop and target logic reads high/low.

    Identical ``seed`` always yields identical bytes.
    """
    step_ms = TIMEFRAME_MS.get(timeframe)
    if step_ms is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}")

    rng = np.random.default_rng(seed)
    price = base_price if base_price is not None else BASE_PRICES.get(symbol, 100.0)
    series = _Series(symbol=symbol, price=price, rng=rng)
    series.regime = str(rng.choice(REGIME_NAMES))

    # Anchor so the series ends near "now" but on a clean bar boundary.
    if start_ts is None:
        now = int(time.time() * 1000)
        start_ts = (now - bars * step_ms) // step_ms * step_ms

    dt = step_ms / 1000.0 / sub_steps
    ts = np.empty(bars, dtype=np.int64)
    opens = np.empty(bars, dtype=np.float64)
    highs = np.empty(bars, dtype=np.float64)
    lows = np.empty(bars, dtype=np.float64)
    closes = np.empty(bars, dtype=np.float64)
    volumes = np.empty(bars, dtype=np.float64)

    for i in range(bars):
        bar_open = series.price
        high = low = bar_open
        volume = 0.0
        for _ in range(sub_steps):
            series.maybe_switch_regime(dt)
            p = series.step(dt)
            high = max(high, p)
            low = min(low, p)
            # Volume loosely tracks activity; scale-aware so it is not tiny for
            # cheap assets or enormous for expensive ones.
            volume += abs(rng.normal(1.0, 0.35)) * (bar_open / 1000.0 + 1.0)

        ts[i] = start_ts + i * step_ms
        opens[i] = bar_open
        highs[i] = high
        lows[i] = low
        closes[i] = series.price
        volumes[i] = volume

    return Dataset.from_arrays(
        ts=ts, open=opens, high=highs, low=lows, close=closes, volume=volumes,
        source=DataSource.SYNTHETIC,
        symbol=symbol,
        timeframe=timeframe,
        seed=seed,
        notes=(
            f"regime-switching GBM, sub_steps={sub_steps}, "
            f"regimes={'|'.join(REGIMES)}"
        ),
    )


# --------------------------------------------------------------------------
# Real venues
# --------------------------------------------------------------------------


def _page_limit(exchange) -> int:
    """The venue's own OHLCV page cap, when ccxt knows it."""
    try:
        limits = (exchange.options or {}).get("fetchOHLCVLimit")
        if isinstance(limits, int) and limits > 0:
            return min(limits, DEFAULT_PAGE_LIMIT)
    except Exception:
        pass
    return DEFAULT_PAGE_LIMIT


def build_exchange(exchange_id: str, *, testnet: bool = False, market_type: str = "spot"):
    """Instantiate a ccxt async exchange, or explain precisely why we cannot.

    Kept separate from :func:`collect_exchange` so the CLI can validate a venue
    argument without starting a download.
    """
    import ccxt.async_support as ccxt

    venue = DataSource(exchange_id)      # rejects unknown ids before any network call
    if venue.is_synthetic:
        raise ValueError("collect_exchange needs a real venue; use generate_synthetic")

    factory = getattr(ccxt, venue.exchange_id, None)
    if factory is None:
        raise ValueError(f"ccxt has no exchange class for {venue.exchange_id!r}")

    exchange = factory({
        "enableRateLimit": True,          # ccxt then paces requests to the venue's own limit
        "options": {"defaultType": market_type},
    })
    try:
        if not exchange.has.get("fetchOHLCV"):
            raise ValueError(
                f"{venue.exchange_id} does not expose fetchOHLCV, so it cannot supply "
                "historical bars"
            )
        if testnet:
            # Not every venue has a sandbox; say so rather than silently pulling
            # live data when the caller asked for the test environment.
            try:
                exchange.set_sandbox_mode(True)
            except Exception as exc:
                raise ValueError(f"{venue.exchange_id} has no sandbox mode: {exc}") from exc
    except Exception:
        _close_sync(exchange)
        raise
    return exchange


def _close_sync(exchange) -> None:
    """Best-effort close for an exchange we are abandoning before any request."""
    try:
        closer = exchange.close()
        if asyncio.iscoroutine(closer):
            closer.close()
    except Exception:
        pass


async def collect_exchange(
    exchange_id: str,
    symbol: str,
    timeframe: str = "5m",
    *,
    since: int | None = None,
    until: int | None = None,
    testnet: bool = False,
    market_type: str = "spot",
    cache_dir: str | Path | None = None,
    request_pause: float = 0.0,
    max_pages: int | None = None,
) -> Dataset:
    """Fetch real OHLCV history from any ccxt venue, resuming a partial cache.

    ``since``/``until`` are epoch milliseconds. Partial progress is written after
    every page so an interrupted multi-year pull can be resumed rather than
    restarted.
    """
    venue = DataSource(exchange_id)
    if venue.is_synthetic:
        raise ValueError(
            "collect_exchange fetches from a real venue; call generate_synthetic() "
            "for simulated bars"
        )
    resolved = to_ccxt_symbol(symbol)
    step_ms = TIMEFRAME_MS.get(timeframe)
    if step_ms is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}")

    if since is None:
        since = int(time.time() * 1000) - 365 * 86_400_000
    if until is None:
        until = int(time.time() * 1000)
    if since >= until:
        raise ValueError(f"since ({since}) must be before until ({until})")

    cache_path = (
        _cache_path(cache_dir, venue.exchange_id, resolved, timeframe) if cache_dir else None
    )
    rows = _load_partial(cache_path)
    if rows.size:
        # Resume one bar past whatever we already have.
        cursor = int(rows[-1, 0]) + step_ms
        logger.info(
            "resuming %s %s %s from %s (%d bars cached)",
            venue.exchange_id, resolved, timeframe, _iso(cursor), len(rows),
        )
    else:
        cursor = since

    exchange = build_exchange(venue.exchange_id, testnet=testnet, market_type=market_type)
    page_limit = _page_limit(exchange)

    pages = 0
    collected: list[np.ndarray] = [rows] if rows.size else []
    try:
        if timeframe not in (exchange.timeframes or {timeframe: None}):
            raise ValueError(
                f"{venue.exchange_id} does not offer the {timeframe} timeframe; "
                f"it has {sorted(exchange.timeframes)}"
            )

        while cursor < until:
            if max_pages is not None and pages >= max_pages:
                logger.info("stopping at max_pages=%d", max_pages)
                break

            batch = await exchange.fetch_ohlcv(
                resolved, timeframe, since=cursor, limit=page_limit
            )
            if not batch:
                logger.info("no more data returned at %s", _iso(cursor))
                break

            page = np.asarray(batch, dtype=np.float64)
            # Trim anything at or past `until` so the range is exact.
            page = page[page[:, 0] < until]
            if page.size == 0:
                break

            collected.append(page)
            pages += 1
            last_ts = int(page[-1, 0])

            if last_ts < cursor:
                # Defensive: a non-advancing cursor would spin forever.
                logger.warning("%s returned non-advancing data; stopping", venue.exchange_id)
                break
            cursor = last_ts + step_ms

            if cache_path is not None:
                _save_partial(cache_path, np.vstack(collected))
            if pages % 10 == 0:
                logger.info("  %d pages, through %s", pages, _iso(last_ts))
            if request_pause:
                await asyncio.sleep(request_pause)
    finally:
        await exchange.close()

    if not collected:
        raise RuntimeError(f"no data collected for {resolved} {timeframe} on {venue.exchange_id}")

    merged = _dedupe_sorted(np.vstack(collected))

    return Dataset.from_arrays(
        ts=merged[:, 0].astype(np.int64),
        open=merged[:, 1], high=merged[:, 2], low=merged[:, 3],
        close=merged[:, 4], volume=merged[:, 5],
        source=venue,
        symbol=resolved,
        timeframe=timeframe,
        notes=(
            f"ccxt {venue.exchange_id}{'-testnet' if testnet else ''} {market_type}, "
            f"{pages} pages of <={page_limit}"
        ),
    )


async def collect_binance(symbol: str, timeframe: str = "5m", **kwargs) -> Dataset:
    """Backwards-compatible shim for the venue-generic collector."""
    return await collect_exchange("binance", symbol, timeframe, **kwargs)


def _dedupe_sorted(rows: np.ndarray) -> np.ndarray:
    """Sort by timestamp and drop duplicates.

    Paginated fetches routinely overlap at page boundaries, and a duplicated bar
    would be silently double-counted by every downstream calculation.
    """
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    _, unique_idx = np.unique(rows[:, 0], return_index=True)
    return rows[np.sort(unique_idx)]


def _cache_path(cache_dir: str | Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    compact = symbol.replace("/", "").replace(":", "-")
    directory = Path(cache_dir) / "raw" / exchange_id / compact / timeframe
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "partial.npz"


def _load_partial(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.empty((0, 6), dtype=np.float64)
    try:
        with np.load(path) as payload:
            return payload["rows"]
    except Exception:
        logger.warning("ignoring unreadable partial cache at %s", path)
        return np.empty((0, 6), dtype=np.float64)


def _save_partial(path: Path, rows: np.ndarray) -> None:
    np.savez_compressed(path, rows=_dedupe_sorted(rows))


def _iso(ts_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def parse_date(value: str) -> int:
    """``2020-01-01`` or ``2020-01-01T12:00`` → epoch milliseconds (UTC)."""
    from datetime import datetime, timezone

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(
                datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000
            )
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {value!r}; use YYYY-MM-DD")


def estimate_pages(since: int, until: int, timeframe: str, page_limit: int = DEFAULT_PAGE_LIMIT) -> int:
    step = TIMEFRAME_MS.get(timeframe, 300_000)
    return max(1, math.ceil((until - since) / step / page_limit))
