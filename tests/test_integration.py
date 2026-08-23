"""End-to-end: the engine, the event bus cadence, and the WebSocket protocol."""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app import create_app
from backend.eventbus import MAX_LIFECYCLE_BACKLOG, EventBus
from backend.execution.base import Fill
from backend.models import CloseReason, PositionSide, Ticker
from backend.orchestrator import JojoOrchestrator


# --------------------------------------------------------------- event bus


def test_market_ticks_coalesce_but_fills_never_do():
    """The cadence split is the guarantee that data outranks decoration."""
    bus = EventBus()
    sub = bus.subscribe("test")

    for i in range(50):
        bus.publish("market.tick", {"symbol": "BTC/USDT", "price": float(i)})
    for i in range(5):
        bus.publish("order.filled", {"order_id": f"o{i}"})

    ticks = [f for f in list(sub._lifecycle) + list(sub._coalesced.values())
             if f["type"] == "market.tick"]
    fills = [f for f in sub._lifecycle if f["type"] == "order.filled"]

    assert len(ticks) == 1, "50 ticks for one symbol must collapse to the latest"
    assert ticks[0]["data"]["price"] == 49.0, "the surviving tick must be the newest"
    assert len(fills) == 5, "every fill must survive"
    assert sub.dropped == 49


def test_ticks_for_different_symbols_do_not_collapse_together():
    bus = EventBus()
    sub = bus.subscribe("test")
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        for i in range(10):
            bus.publish("market.tick", {"symbol": symbol, "price": float(i)})
    assert len(sub._coalesced) == 3


def test_a_hopelessly_slow_subscriber_is_dropped_not_starved():
    """Rather than silently discarding fills, the connection is cut."""
    bus = EventBus()
    sub = bus.subscribe("slow")
    for i in range(MAX_LIFECYCLE_BACKLOG + 10):
        bus.publish("order.filled", {"order_id": str(i)})
    assert sub.closed


async def test_drain_returns_lifecycle_before_coalesced():
    bus = EventBus()
    sub = bus.subscribe("test")
    bus.publish("market.tick", {"symbol": "BTC/USDT", "price": 1.0})
    bus.publish("order.filled", {"order_id": "o1"})
    batch = await sub.drain()
    assert batch[0]["type"] == "order.filled"
    assert batch[-1]["type"] == "market.tick"


def test_a_failing_listener_cannot_break_the_bus():
    bus = EventBus()
    delivered = []
    bus.add_listener(lambda item: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.add_listener(delivered.append)
    bus.publish("order.filled", {"order_id": "o1"})
    assert len(delivered) == 1


def test_recent_events_are_replayed_for_late_joiners():
    bus = EventBus(event_buffer=4)
    for i in range(10):
        bus.emit("test.event", f"event {i}")
    events = bus.recent_events()
    assert len(events) == 4, "the ring buffer must bound replay"
    assert events[-1].message == "event 9"


# ------------------------------------------------------------- orchestrator


@pytest.fixture
async def engine(settings):
    settings.provider = "simulated"
    orchestrator = JojoOrchestrator(settings, store=None)
    await orchestrator.start()
    try:
        yield orchestrator
    finally:
        await orchestrator.stop()


async def test_engine_starts_in_paper_mode_on_the_synthetic_feed(engine):
    assert engine.running
    assert not engine.execution.is_live
    assert engine.provider.name == "simulated"
    assert len(engine.bots) == 5
    assert {b.name for b in engine.bots.values()} == {
        "JONATHAN", "JOSEPH", "JOTARO", "JOLYAN", "KIRA"
    }


async def test_snapshot_is_complete_and_serialisable(engine):
    snapshot = engine.snapshot()
    payload = snapshot.model_dump(mode="json")
    json.dumps(payload)                                  # must not raise
    assert payload["system"]["mode"] == "PAPER"
    assert len(payload["bots"]) == 5
    assert payload["candles"], "the market room needs candle history"
    assert payload["tickers"], "the world needs live prices"
    for bot in payload["bots"]:
        assert bot["persona"]["palette"], f"{bot['name']} has no palette for its voxel model"
        assert len(bot["persona"]["district"]) == 2


async def test_stop_loss_closes_a_losing_long(engine):
    """Drive a real position into its stop and assert the round trip completes."""
    agent = engine.bots["jonathan"]
    ticker = engine.provider.ticker(agent.config.symbol)
    assert ticker is not None

    position = engine.portfolio.open_position(
        bot_id=agent.id, symbol=agent.config.symbol, side=PositionSide.LONG,
        fill=Fill(order_id="seed", price=ticker.price, quantity=0.01, fee=0.0),
        leverage=2.0,
        stop_loss=ticker.price * 0.99,
        take_profit=ticker.price * 1.10,
    )

    crash = Ticker(
        symbol=agent.config.symbol, price=position.stop_loss * 0.98,
        bid=position.stop_loss * 0.979, ask=position.stop_loss * 0.981,
    )
    await engine._check_stops(crash)

    assert position.id not in engine.portfolio.positions
    trade = engine.portfolio.trades[-1]
    assert trade.reason is CloseReason.STOP_LOSS
    assert trade.realized_pnl < 0


async def test_take_profit_closes_a_winning_long(engine):
    agent = engine.bots["jonathan"]
    ticker = engine.provider.ticker(agent.config.symbol)
    position = engine.portfolio.open_position(
        bot_id=agent.id, symbol=agent.config.symbol, side=PositionSide.LONG,
        fill=Fill(order_id="seed", price=ticker.price, quantity=0.01, fee=0.0),
        leverage=2.0,
        stop_loss=ticker.price * 0.90,
        take_profit=ticker.price * 1.01,
    )
    rally = Ticker(
        symbol=agent.config.symbol, price=position.take_profit * 1.02,
        bid=position.take_profit * 1.019, ask=position.take_profit * 1.021,
    )
    await engine._check_stops(rally)

    trade = engine.portfolio.trades[-1]
    assert trade.reason is CloseReason.TAKE_PROFIT
    assert trade.realized_pnl > 0


async def test_short_stop_triggers_when_price_rises(engine):
    agent = engine.bots["kira"]
    ticker = engine.provider.ticker(agent.config.symbol)
    position = engine.portfolio.open_position(
        bot_id=agent.id, symbol=agent.config.symbol, side=PositionSide.SHORT,
        fill=Fill(order_id="seed", price=ticker.price, quantity=0.5, fee=0.0),
        leverage=2.0,
        stop_loss=ticker.price * 1.01,
        take_profit=ticker.price * 0.90,
    )
    spike = Ticker(
        symbol=agent.config.symbol, price=position.stop_loss * 1.02,
        bid=position.stop_loss * 1.019, ask=position.stop_loss * 1.021,
    )
    await engine._check_stops(spike)
    assert engine.portfolio.trades[-1].reason is CloseReason.STOP_LOSS


async def test_emergency_stop_flattens_and_halts_everything(engine):
    agent = engine.bots["jotaro"]
    ticker = engine.provider.ticker(agent.config.symbol)
    engine.portfolio.open_position(
        bot_id=agent.id, symbol=agent.config.symbol, side=PositionSide.LONG,
        fill=Fill(order_id="seed", price=ticker.price, quantity=0.01, fee=0.0),
        leverage=1.0,
    )
    assert engine.portfolio.positions

    await engine.emergency_stop()

    assert engine.portfolio.positions == {}, "every position must be flattened"
    assert engine.risk.emergency_stop
    assert all(b.status.value == "HALTED" for b in engine.bots.values())
    assert engine.portfolio.trades[-1].reason is CloseReason.EMERGENCY_STOP

    await engine.resume_system()
    assert not engine.risk.emergency_stop
    assert not any(b.status.value == "HALTED" for b in engine.bots.values())


async def test_bot_commands_change_state(engine):
    ok, message = await engine.command("jotaro", "pause")
    assert ok and engine.bots["jotaro"].status.value == "PAUSED"

    ok, _ = await engine.command("jotaro", "resume")
    assert ok and engine.bots["jotaro"].status.value != "PAUSED"

    ok, message = await engine.command("jotaro", "moonwalk")
    assert not ok and "unknown action" in message

    ok, message = await engine.command("dio", "start")
    assert not ok and "unknown bot" in message


async def test_a_paused_bot_does_not_trade(engine):
    agent = engine.bots["jotaro"]
    agent.pause()
    before = len(engine.portfolio.positions)
    for _ in range(3):
        assert await agent.tick() is None
    assert len(engine.portfolio.positions) == before


async def test_engine_produces_trades_when_left_running(engine):
    """The real proof: run the desk and watch orders flow through to fills."""
    frames: list[str] = []
    engine.bus.add_listener(lambda item: frames.append(item["type"]))

    for _ in range(120):
        await asyncio.sleep(0.25)
        if engine.portfolio.positions or engine.portfolio.trades:
            break

    assert "order.submitted" in frames
    assert "order.filled" in frames
    assert "position.opened" in frames
    assert engine.portfolio.positions or engine.portfolio.trades


# ---------------------------------------------------------------------- REST


@pytest.fixture
async def client(settings, monkeypatch):
    monkeypatch.setenv("JOJO_PROVIDER", "simulated")
    settings.provider = "simulated"
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            yield ac


async def test_health_reports_paper_mode(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "PAPER"
    assert body["bots"] == 5


async def test_system_endpoint_exposes_the_live_gate(client):
    body = (await client.get("/api/system")).json()
    assert body["live_allowed"] is False
    assert body["live_gate"], "the reason the gate is shut must be visible"


async def test_bots_endpoint_carries_persona_for_the_world(client):
    bots = (await client.get("/api/bots")).json()
    assert len(bots) == 5
    for bot in bots:
        assert bot["persona"]["accent"].startswith("#")
        assert bot["persona"]["title"]


async def test_bot_detail_and_unknown_bot(client):
    detail = (await client.get("/api/bots/kira")).json()
    assert detail["bot"]["name"] == "KIRA"
    assert detail["config"]["strategy"] == "mean_reversion"
    assert (await client.get("/api/bots/dio")).status_code == 404


async def test_klines_accepts_display_symbols(client):
    body = (await client.get("/api/market/BTCUSDT/klines?timeframe=5m&limit=20")).json()
    assert body["symbol"] == "BTC/USDT", "BTCUSDT must resolve to the ccxt symbol"
    assert len(body["candles"]) == 20
    assert body["ticker"]["price"] > 0


async def test_unknown_symbol_is_a_404(client):
    assert (await client.get("/api/market/FAKEUSDT/klines")).status_code == 404


async def test_command_endpoint_rejects_a_bad_action(client):
    response = await client.post("/api/bots/kira/command", json={"action": "explode"})
    assert response.status_code == 400


async def test_emergency_stop_endpoint_round_trip(client):
    assert (await client.post("/api/system/emergency-stop")).json()["emergency_stop"]
    risk = (await client.get("/api/risk")).json()
    assert risk["emergency_stop"] is True
    assert risk["level"] == "CRITICAL"

    assert (await client.post("/api/system/resume")).json()["emergency_stop"] is False
    assert (await client.get("/api/risk")).json()["emergency_stop"] is False


async def test_state_endpoint_matches_the_snapshot_shape(client):
    body = (await client.get("/api/state")).json()
    assert set(body) >= {
        "system", "portfolio", "risk", "bots", "positions",
        "orders", "trades", "events", "tickers", "candles",
    }
