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

#: Tables for the offline research pipeline. Kept in this module — rather than
#: under ``research/`` — because ``research`` imports ``backend`` and never the
#: reverse; putting the DDL here lets the research store share this database
#: without inverting that dependency. Purely additive: the six tables above are
#: untouched, and the trading server neither reads nor writes any table below.
#:
#: The load-bearing constraint is ``policy_metrics.dataset_id`` referencing
#: ``datasets``. There is no path to a metric that does not name the dataset it
#: came from, and therefore none to a metric whose data source is unknown — so a
#: synthetic result cannot be reported as real-market evidence by omission.
RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,              -- 'SYNTHETIC' or a ccxt exchange id
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    rows INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    collector_version TEXT NOT NULL,
    seed INTEGER,
    gaps INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(source, symbol, timeframe);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL,              -- PROPOSED|TRAINING|VALIDATING|REJECTED|CANDIDATE|...
    seed INTEGER,
    parent_experiment_id TEXT,
    decision_reason TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_experiments_agent ON experiments(agent, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,                -- 'strategy' | 'baseline' | 'learned'
    algo TEXT,
    experiment_id TEXT,
    hyperparams_json TEXT,
    feature_set TEXT,
    feature_set_version TEXT,
    reward_version TEXT,
    seed INTEGER,
    model_path TEXT,
    weights_path TEXT,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    -- Seed is part of the identity, not an attribute of it. A version is
    -- trained once per seed: inference is argmax and therefore deterministic,
    -- so seed diversity can only come from separate training runs, and five of
    -- them legitimately share one (agent, version).
    UNIQUE (agent, version, seed),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_policies_agent ON policies(agent, version DESC);

-- Long format: one row per (policy, dataset, split, window, regime, seed, metric).
CREATE TABLE IF NOT EXISTS policy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    split_id TEXT,
    window INTEGER,
    regime TEXT,
    seed INTEGER,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    metrics_version TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (policy_id) REFERENCES policies(policy_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON policy_metrics(policy_id, dataset_id, metric);
CREATE INDEX IF NOT EXISTS idx_metrics_metric ON policy_metrics(metric, value);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    split_id TEXT,
    stage TEXT NOT NULL,               -- WALK_FORWARD | HOLDOUT | PAPER | SHADOW
    passed INTEGER NOT NULL,
    gate_json TEXT NOT NULL,           -- each gate condition and whether it held
    reason TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (policy_id) REFERENCES policies(policy_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);
CREATE INDEX IF NOT EXISTS idx_evaluations_policy ON evaluations(policy_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,                -- HYPOTHESIS | OUTCOME | LESSON | DEAD_END
    content_json TEXT NOT NULL,
    experiment_id TEXT,
    policy_id TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory(agent, kind, created_at DESC);

CREATE TABLE IF NOT EXISTS champion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    action TEXT NOT NULL,              -- PROMOTED | RETIRED
    reason TEXT NOT NULL,
    previous_policy_id TEXT,
    ts INTEGER NOT NULL,
    FOREIGN KEY (policy_id) REFERENCES policies(policy_id)
);
CREATE INDEX IF NOT EXISTS idx_champion_agent ON champion_history(agent, ts DESC);

CREATE TABLE IF NOT EXISTS allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    policy_id TEXT,
    paper_capital REAL NOT NULL,
    reason TEXT NOT NULL,
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_allocations_agent ON allocations(agent, ts DESC);
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
        # Additive only: research tables are created so the two halves share one
        # database file. The trading loop never touches them.
        await self._db.executescript(RESEARCH_SCHEMA)
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

    # ---- research (read-only) ----------------------------------------------
    #
    # The trading server reads the research tables and never writes them. It
    # also does not import anything from ``research`` — the DDL lives in this
    # module precisely so this side can read the desk's research standing
    # without inverting the dependency. If the research pipeline has never run,
    # every query below simply returns nothing.

    async def research_state(self, agents: list[str] | None = None) -> dict[str, Any]:
        """A display-ready summary of the research half of the desk."""
        counts = {}
        for table in ("datasets", "policies", "experiments"):
            rows = await self._fetch(f"SELECT COUNT(*) AS c FROM {table}")
            counts[table] = int(rows[0]["c"]) if rows else 0

        sources = [
            row["source"] for row in await self._fetch(
                "SELECT DISTINCT source FROM datasets ORDER BY source")
        ]
        allocations = await self._fetch(
            """SELECT a.agent, a.paper_capital FROM allocations a
               JOIN (SELECT agent, MAX(ts) AS ts FROM allocations GROUP BY agent) latest
                 ON a.agent = latest.agent AND a.ts = latest.ts"""
        )
        deployed = float(sum(row["paper_capital"] for row in allocations))

        names = agents or [row["agent"] for row in await self._fetch(
            "SELECT DISTINCT agent FROM experiments ORDER BY agent")]
        return {
            "available": counts["experiments"] > 0 or counts["policies"] > 0,
            "agents": [await self.agent_research(name) for name in names],
            "datasets": counts["datasets"],
            "policies": counts["policies"],
            "experiments": counts["experiments"],
            "sources": sources,
            # Anything other than a pure-SYNTHETIC set means real-market data is
            # present; until then the interface must say so.
            "synthetic_only": all(s == "SYNTHETIC" for s in sources) if sources else True,
            "paper_capital_deployed": deployed,
        }

    async def agent_research(self, agent: str) -> dict[str, Any]:
        experiments = await self._fetch(
            "SELECT status, COUNT(*) AS c FROM experiments WHERE agent = ? GROUP BY status",
            (agent,),
        )
        by_status = {row["status"]: int(row["c"]) for row in experiments}

        latest = await self._fetch(
            """SELECT experiment_id, status, hypothesis, updated_at FROM experiments
               WHERE agent = ? ORDER BY updated_at DESC LIMIT 1""",
            (agent,),
        )
        current = latest[0] if latest else None

        champion = await self._fetch(
            """SELECT policy_id, action FROM champion_history
               WHERE agent = ? ORDER BY ts DESC, id DESC LIMIT 1""",
            (agent,),
        )
        champion_policy = None
        champion_version = None
        if champion and champion[0]["action"] == "PROMOTED":
            champion_policy = champion[0]["policy_id"]
            version = await self._fetch(
                "SELECT version FROM policies WHERE policy_id = ?", (champion_policy,))
            champion_version = int(version[0]["version"]) if version else None

        challengers = await self._fetch(
            "SELECT COUNT(*) AS c FROM policies WHERE agent = ? AND policy_id != ?",
            (agent, champion_policy or ""),
        )
        allocation = await self._fetch(
            "SELECT paper_capital FROM allocations WHERE agent = ? ORDER BY ts DESC LIMIT 1",
            (agent,),
        )
        return {
            "agent": agent,
            "champion_policy": champion_policy,
            "champion_version": champion_version,
            "challenger_count": int(challengers[0]["c"]) if challengers else 0,
            "experiments_total": sum(by_status.values()),
            "experiments_rejected": by_status.get("REJECTED", 0),
            "current_experiment": current["experiment_id"] if current else None,
            "current_status": current["status"] if current else None,
            "hypothesis": current["hypothesis"] if current else None,
            "paper_allocation": float(allocation[0]["paper_capital"]) if allocation else 0.0,
            "last_updated": int(current["updated_at"]) if current else 0,
        }

    async def audit_trail(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = await self._fetch("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))
        for row in rows:
            row["detail"] = json.loads(row["detail"]) if row.get("detail") else None
        return rows
