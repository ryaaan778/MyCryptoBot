"""Async pub/sub between the trading engine and every connected client.

The cadence split lives here, and it is the mechanism behind "trading data has
priority over visual effects":

* **Lifecycle frames** (orders, fills, positions, trades, risk, emergency stop)
  go into a per-subscriber FIFO and are *never* coalesced or silently dropped.
  If a client is so slow that the FIFO overflows, that client is disconnected —
  we would rather drop the connection than quietly drop a fill.
* **Market ticks** are decorative price updates. They collapse into a single
  latest-value slot per symbol and are flushed at a fixed rate, so a busy market
  cannot starve the lifecycle stream.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Iterable

from .models import Event

logger = logging.getLogger("jojo.eventbus")

# Frame types that may be coalesced. Everything else is lifecycle-critical.
COALESCABLE = frozenset({"market.tick", "pnl.tick", "bot.heartbeat"})

# Beyond this backlog a subscriber is considered dead rather than slow.
MAX_LIFECYCLE_BACKLOG = 2048


def frame(type_: str, data: Any) -> dict[str, Any]:
    return {"type": type_, "data": data}


class Subscriber:
    """One connected client's outbound buffer."""

    __slots__ = ("_lifecycle", "_coalesced", "_wake", "_closed", "dropped", "name")

    def __init__(self, name: str = "client") -> None:
        self.name = name
        self._lifecycle: deque[dict[str, Any]] = deque()
        self._coalesced: dict[str, dict[str, Any]] = {}
        self._wake = asyncio.Event()
        self._closed = False
        self.dropped = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    def offer(self, item: dict[str, Any]) -> None:
        if self._closed:
            return
        ftype = item.get("type", "")
        if ftype in COALESCABLE:
            # Latest value wins; the key keeps per-symbol/per-bot streams separate.
            data = item.get("data") or {}
            key = ftype
            if isinstance(data, dict):
                key = f"{ftype}:{data.get('symbol') or data.get('bot_id') or ''}"
            if key in self._coalesced:
                self.dropped += 1
            self._coalesced[key] = item
        else:
            if len(self._lifecycle) >= MAX_LIFECYCLE_BACKLOG:
                # Never drop a lifecycle frame — cut the client loose instead.
                logger.warning("subscriber %s overflowed lifecycle backlog; closing", self.name)
                self.close()
                return
            self._lifecycle.append(item)
        self._wake.set()

    async def drain(self) -> list[dict[str, Any]]:
        """Wait for work, then return everything pending in priority order."""
        while not self._closed and not self._lifecycle and not self._coalesced:
            self._wake.clear()
            await self._wake.wait()
        if self._closed and not self._lifecycle and not self._coalesced:
            return []
        batch = list(self._lifecycle)
        self._lifecycle.clear()
        batch.extend(self._coalesced.values())
        self._coalesced.clear()
        self._wake.clear()
        return batch


class EventBus:
    """Fan-out to every subscriber, plus a replay buffer for late joiners."""

    def __init__(self, event_buffer: int = 400) -> None:
        self._subscribers: set[Subscriber] = set()
        self._events: deque[Event] = deque(maxlen=event_buffer)
        self._listeners: list[Any] = []

    # ---- subscription ------------------------------------------------------

    def subscribe(self, name: str = "client") -> Subscriber:
        sub = Subscriber(name)
        self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        sub.close()
        self._subscribers.discard(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def add_listener(self, callback: Any) -> None:
        """Register an in-process listener (used by the SQLite writer)."""
        self._listeners.append(callback)

    # ---- publishing --------------------------------------------------------

    def publish(self, type_: str, data: Any) -> None:
        item = frame(type_, data)
        dead = [s for s in self._subscribers if s.closed]
        for s in dead:
            self._subscribers.discard(s)
        for sub in self._subscribers:
            sub.offer(item)
        for listener in self._listeners:
            try:
                listener(item)
            except Exception:  # a bad listener must never break the engine
                logger.exception("event listener failed for %s", type_)

    def publish_event(self, event: Event) -> None:
        """Publish a log-worthy event: buffered for replay and fanned out."""
        self._events.append(event)
        self.publish("event.log", event.model_dump(mode="json"))

    def emit(
        self,
        type_: str,
        message: str,
        *,
        severity: str = "info",
        bot_id: str | None = None,
        symbol: str | None = None,
        **data: Any,
    ) -> Event:
        event = Event(
            type=type_,
            message=message,
            severity=severity,  # type: ignore[arg-type]
            bot_id=bot_id,
            symbol=symbol,
            data=data,
        )
        self.publish_event(event)
        return event

    # ---- replay ------------------------------------------------------------

    def recent_events(self, limit: int | None = None) -> list[Event]:
        events = list(self._events)
        return events[-limit:] if limit else events

    def close_all(self) -> None:
        for sub in list(self._subscribers):
            self.unsubscribe(sub)

    def broadcast_many(self, items: Iterable[tuple[str, Any]]) -> None:
        for type_, data in items:
            self.publish(type_, data)
