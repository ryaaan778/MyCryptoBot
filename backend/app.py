"""FastAPI application: REST surface, WebSocket hub, static hosting.

The WebSocket is the primary channel — one ``snapshot`` frame on connect, then
deltas. REST exists for cold reads, the 2D fallback, and anything that wants to
poll rather than stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, Settings, load_settings, to_ccxt_symbol
from .models import now_ms
from .orchestrator import JojoOrchestrator
from .store import Store

logger = logging.getLogger("jojo.api")

WEB_DIST = REPO_ROOT / "web" / "dist"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    app = FastAPI(
        title="JOJO Trading Command Center",
        version="1.0.0",
        description="Live multi-bot crypto trading engine behind a voxel world.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"engine": None, "store": None}
    app.state.settings = settings

    # ---------------------------------------------------------------- startup

    @app.on_event("startup")
    async def _startup() -> None:
        store = Store(Path(settings.data_dir))
        await store.open()
        engine = JojoOrchestrator(settings, store)
        await engine.start()
        state["engine"] = engine
        state["store"] = store
        app.state.engine = engine

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine: JojoOrchestrator | None = state.get("engine")
        store: Store | None = state.get("store")
        if engine:
            await engine.stop()
        if store:
            await store.close()

    def get_engine() -> JojoOrchestrator:
        engine = state.get("engine")
        if engine is None:
            raise HTTPException(503, "engine is still starting")
        return engine

    def get_store() -> Store:
        store = state.get("store")
        if store is None:
            raise HTTPException(503, "store is still starting")
        return store

    # ------------------------------------------------------------------- REST

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        engine = state.get("engine")
        return {
            "status": "ok" if engine and engine.running else "starting",
            "server_time": now_ms(),
            "mode": engine.mode.value if engine else "UNKNOWN",
            "provider": engine.provider.name if engine and engine.provider else "none",
            "provider_degraded": engine.provider_degraded if engine else False,
            "bots": len(engine.bots) if engine else 0,
        }

    @app.get("/api/state")
    async def full_state() -> Any:
        return get_engine().snapshot().model_dump(mode="json")

    @app.get("/api/system")
    async def system() -> Any:
        engine = get_engine()
        payload = engine.system_state().model_dump(mode="json")
        payload["live_gate"] = engine.live_gate_reason
        payload["live_allowed"] = settings.live_allowed
        return payload

    @app.get("/api/bots")
    async def bots() -> Any:
        return [b.state().model_dump(mode="json") for b in get_engine().bots.values()]

    @app.get("/api/bots/{bot_id}")
    async def bot_detail(bot_id: str) -> Any:
        engine = get_engine()
        agent = engine.bots.get(bot_id)
        if agent is None:
            raise HTTPException(404, f"unknown bot {bot_id!r}")
        store = get_store()
        return {
            "bot": agent.state().model_dump(mode="json"),
            "config": agent.config.model_dump(mode="json"),
            "positions": [
                p.model_dump(mode="json") for p in engine.portfolio.positions_for(bot_id)
            ],
            "trades": await store.recent_trades(40, bot_id),
            "orders": await store.recent_orders(40, bot_id),
        }

    @app.post("/api/bots/{bot_id}/command")
    async def bot_command(bot_id: str, payload: dict[str, str] = Body(...)) -> Any:
        action = payload.get("action", "")
        ok, message = await get_engine().command(bot_id, action)
        if not ok:
            raise HTTPException(400, message)
        return {"ok": True, "message": message}

    @app.get("/api/positions")
    async def positions() -> Any:
        engine = get_engine()
        return [p.model_dump(mode="json") for p in engine.portfolio.positions.values()]

    @app.get("/api/orders")
    async def orders(limit: int = Query(100, ge=1, le=1000)) -> Any:
        return await get_store().recent_orders(limit)

    @app.get("/api/trades")
    async def trades(limit: int = Query(100, ge=1, le=1000)) -> Any:
        return await get_store().recent_trades(limit)

    @app.get("/api/events")
    async def events(limit: int = Query(200, ge=1, le=1000)) -> Any:
        return [e.model_dump(mode="json") for e in get_engine().bus.recent_events(limit)]

    @app.get("/api/risk")
    async def risk() -> Any:
        return get_engine().risk.evaluate().model_dump(mode="json")

    @app.get("/api/portfolio")
    async def portfolio() -> Any:
        engine = get_engine()
        active = sum(1 for b in engine.bots.values() if b.status.value != "OFFLINE")
        return engine.portfolio.state(active, len(engine.bots)).model_dump(mode="json")

    @app.get("/api/equity")
    async def equity(limit: int = Query(400, ge=1, le=5000)) -> Any:
        return await get_store().equity_history(limit)

    @app.get("/api/market/{symbol}/klines")
    async def klines(
        symbol: str,
        timeframe: str = Query("5m"),
        limit: int = Query(180, ge=1, le=1000),
    ) -> Any:
        engine = get_engine()
        resolved = to_ccxt_symbol(symbol)
        if engine.provider is None:
            raise HTTPException(503, "market data provider is not ready")
        candles = engine.provider.candles(resolved, timeframe, limit)
        if not candles:
            raise HTTPException(404, f"no candles for {resolved} {timeframe}")
        ticker = engine.provider.ticker(resolved)
        return {
            "symbol": resolved,
            "timeframe": timeframe,
            "candles": [c.model_dump() for c in candles],
            "ticker": ticker.model_dump(mode="json") if ticker else None,
        }

    @app.get("/api/audit")
    async def audit(limit: int = Query(200, ge=1, le=1000)) -> Any:
        return await get_store().audit_trail(limit)

    @app.post("/api/system/emergency-stop")
    async def emergency_stop() -> Any:
        await get_engine().emergency_stop()
        return {"ok": True, "emergency_stop": True}

    @app.post("/api/system/resume")
    async def resume() -> Any:
        await get_engine().resume_system()
        return {"ok": True, "emergency_stop": False}

    # -------------------------------------------------------------- WebSocket

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        engine = state.get("engine")
        if engine is None:
            await ws.send_json({"type": "error", "data": {"message": "engine starting"}})
            await ws.close()
            return

        client = f"{ws.client.host}:{ws.client.port}" if ws.client else "client"
        sub = engine.bus.subscribe(client)

        # First frame is always the complete world state.
        await ws.send_json({
            "type": "snapshot",
            "data": engine.snapshot().model_dump(mode="json"),
        })

        async def pump() -> None:
            while True:
                batch = await sub.drain()
                if not batch:
                    if sub.closed:
                        return
                    continue
                for item in batch:
                    await ws.send_json(item)

        pump_task = asyncio.create_task(pump(), name=f"ws-pump-{client}")
        try:
            while True:
                message = await ws.receive_json()
                await _handle_client_message(engine, ws, message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("ws %s closed: %s", client, exc)
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task
            engine.bus.unsubscribe(sub)

    async def _handle_client_message(
        engine: JojoOrchestrator, ws: WebSocket, message: dict[str, Any]
    ) -> None:
        msg_type = message.get("type", "")
        data = message.get("data") or {}

        if msg_type == "ping":
            await ws.send_json({"type": "pong", "data": {"ts": now_ms()}})
            return
        if msg_type == "snapshot.request":
            await ws.send_json({
                "type": "snapshot", "data": engine.snapshot().model_dump(mode="json")
            })
            return
        if msg_type.startswith("bot."):
            action = msg_type.split(".", 1)[1]
            bot_id = data.get("bot_id", "")
            ok, detail = await engine.command(bot_id, action)
            await ws.send_json({"type": "command.result", "data": {
                "ok": ok, "message": detail, "bot_id": bot_id, "action": action
            }})
            return
        if msg_type == "system.emergency_stop":
            await engine.emergency_stop()
            return
        if msg_type == "system.resume":
            await engine.resume_system()
            return

        await ws.send_json({
            "type": "error", "data": {"message": f"unknown message type {msg_type!r}"}
        })

    # ----------------------------------------------------------- static hosting

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/")
        async def index() -> Any:
            return FileResponse(WEB_DIST / "index.html")

        @app.get("/{path:path}")
        async def spa(path: str) -> Any:
            candidate = WEB_DIST / path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/")
        async def placeholder() -> Any:
            return JSONResponse({
                "service": "JOJO Trading Command Center",
                "note": "Frontend is not built. Run `cd web && npm install && npm run build`, "
                        "or use the Vite dev server on :5173.",
                "api": "/api/health",
                "websocket": "/ws",
            })

    return app


app = create_app()
