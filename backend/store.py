"""SQLite persistence.

Writes are queued and flushed by a background task so that disk I/O can never
block the trading loop. The ``audit_log`` table is append-only and records every
command and every state-changing decision the system makes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from .models import Event, Order, Position, Trade

logger = logging.getLogger("jojo.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    type TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    filled_quantity REAL NOT NULL,
    average_fill_price REAL NOT NULL,
    status TEXT NOT NULL,
    fee REAL NOT NULL,
    reason TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_bot ON orders(bot_id, created_at DESC);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    leverage REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    realized_pnl_pct REAL NOT NULL,
    fees REAL NOT NULL,
    reason TEXT NOT NULL,
    opened_at INTEGER NOT NULL,
    closed_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_bot ON trades(bot_id, closed_at DESC);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    leverage REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    margin REAL NOT NULL,
    opened_at INTEGER NOT NULL,
    closed_at INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    bot_id TEXT,
    symbol TEXT,
    message TEXT,
    severity TEXT,
    data TEXT,
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts INTEGER PRIMARY KEY,
    equity REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    exposure REAL NOT NULL,
    drawdown_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
"""


class Store:
    def __init__(self, data_dir: str | Path, filename: str = "jojo.db") -> None:
        self.dir = Path(data_dir)
        self.path = self.dir / filename
        self._db: aiosqlite.Connection | None = None
        self._queue: asyncio.Queue[tuple[str, tuple[Any, ...]]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closing = False

    async def open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        self._task = asyncio.create_task(self._writer(), name="store-writer")
        logger.info("store open at %s", self.path)

    async def close(self) -> None:
        self._closing = True
        if self._task:
            await self._queue.put(("", ()))       # wake the writer so it can exit
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        if self._db:
            await self._db.commit()
            await self._db.close()
            self._db = None

    # ---- background writer -------------------------------------------------

    async def _writer(self) -> None:
        assert self._db is not None
        pending = 0
        while True:
            sql, params = await self._queue.get()
            if not sql:
                if self._closing and self._queue.empty():
                    await self._db.commit()
                    return
                continue
            try:
                await self._db.execute(sql, params)
                pending += 1
            except Exception:
                logger.exception("store write failed: %s", sql.split("(")[0])
            if pending >= 32 or self._queue.empty():
                await self._db.commit()
                pending = 0

    def _enqueue(self, sql: str, params: tuple[Any, ...]) -> None:
        if self._db is None or self._closing:
            return
        self._queue.put_nowait((sql, params))

    async def flush(self) -> None:
        """Wait for the queue to drain — used by tests and shutdown."""
        while not self._queue.empty():
            await asyncio.sleep(0.01)
        if self._db is not None:
            await self._db.commit()

    # ---- writes ------------------------------------------------------------

    def save_order(self, order: Order) -> None:
        self._enqueue(
            """INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 filled_quantity=excluded.filled_quantity,
                 average_fill_price=excluded.average_fill_price,
                 status=excluded.status, fee=excluded.fee,
                 updated_at=excluded.updated_at""",
            (
                order.id, order.bot_id, order.symbol, order.side.value, order.type.value,
                order.quantity, order.price, order.filled_quantity, order.average_fill_price,
                order.status.value, order.fee, order.reason, order.created_at, order.updated_at,
            ),
        )

    def save_trade(self, trade: Trade) -> None:
        self._enqueue(
            "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade.id, trade.bot_id, trade.symbol, trade.side.value, trade.quantity,
                trade.entry_price, trade.exit_price, trade.leverage, trade.realized_pnl,
                trade.realized_pnl_pct, trade.fees, trade.reason.value,
                trade.opened_at, trade.closed_at,
            ),
        )

    def save_position(self, position: Position, closed_at: int | None = None) -> None:
        self._enqueue(
            """INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET closed_at=excluded.closed_at""",
            (
                position.id, position.bot_id, position.symbol, position.side.value,
                position.quantity, position.entry_price, position.leverage,
                position.stop_loss, position.take_profit, position.margin,
                position.opened_at, closed_at,
            ),
        )

    def save_event(self, event: Event) -> None:
        self._enqueue(
            "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?)",
            (
                event.id, event.type, event.bot_id, event.symbol, event.message,
                event.severity, json.dumps(event.data, default=str), event.ts,
            ),
        )

    def save_equity(
        self, ts: int, equity: float, realized: float, unrealized: float,
        exposure: float, drawdown_pct: float,
    ) -> None:
        self._enqueue(
            "INSERT OR REPLACE INTO equity_curve VALUES (?,?,?,?,?,?)",
            (ts, equity, realized, unrealized, exposure, drawdown_pct),
        )

    def audit(self, actor: str, action: str, target: str = "", detail: Any = None) -> None:
        from .models import now_ms

        self._enqueue(
            "INSERT INTO audit_log (ts, actor, action, target, detail) VALUES (?,?,?,?,?)",
            (now_ms(), actor, action, target, json.dumps(detail, default=str) if detail else None),
        )

    # ---- reads -------------------------------------------------------------

    async def _fetch(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        await self.flush()
        async with self._db.execute(sql, params) as cur:
            return [dict(row) for row in await cur.fetchall()]

    async def recent_trades(self, limit: int = 100, bot_id: str | None = None) -> list[dict[str, Any]]:
        if bot_id:
            return await self._fetch(
                "SELECT * FROM trades WHERE bot_id=? ORDER BY closed_at DESC LIMIT ?", (bot_id, limit)
            )
        return await self._fetch("SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?", (limit,))

    async def recent_orders(self, limit: int = 100, bot_id: str | None = None) -> list[dict[str, Any]]:
        if bot_id:
            return await self._fetch(
                "SELECT * FROM orders WHERE bot_id=? ORDER BY created_at DESC LIMIT ?", (bot_id, limit)
            )
        return await self._fetch("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))

    async def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = await self._fetch("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))
        for row in rows:
            row["data"] = json.loads(row["data"]) if row.get("data") else {}
        return rows

    async def equity_history(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = await self._fetch(
            "SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?", (limit,)
        )
        return list(reversed(rows))

    async def audit_trail(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = await self._fetch("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))
        for row in rows:
            row["detail"] = json.loads(row["detail"]) if row.get("detail") else None
        return rows
