"""JOJO — the master orchestrator.

Owns the portfolio, the risk engine, the execution adapter and the roster. Every
bot's decision funnels through here, which is what makes the limits real: a
strategy can want anything, but only JOJO can actually open a position.

Concurrency note: bot loops and the price worker both mutate positions, so all
trade mutations are serialised behind ``_trade_lock``. Marking prices is not —
it only ever writes derived fields.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .bot_agent import BotAgent
from .config import Settings
from .eventbus import EventBus
from .execution import create_execution
from .execution.base import ExecutionAdapter
from .market import MarketDataProvider, create_provider
from .models import (
    Candle,
    CloseReason,
    ExecutionMode,
    Order,
    OrderType,
    Position,
    PositionSide,
    RiskLevel,
    Side,
    Signal,
    SignalAction,
    Snapshot,
    SystemState,
    Ticker,
    now_ms,
)
from .portfolio import Portfolio
from .risk import RiskEngine
from .store import Store

logger = logging.getLogger("jojo.orchestrator")

EQUITY_SAMPLE_SEC = 5.0


class JojoOrchestrator:
    def __init__(self, settings: Settings, store: Store | None = None) -> None:
        self.settings = settings
        self.bus = EventBus(event_buffer=settings.server.event_buffer)
        self.store = store
        self.portfolio = Portfolio(settings.initial_balance)
        self.risk = RiskEngine(settings, self.portfolio)
        self.execution: ExecutionAdapter | None = None
        self.provider: MarketDataProvider | None = None  # type: ignore[assignment]
        self.bots: dict[str, BotAgent] = {}

        self.mode = ExecutionMode.PAPER
        self.provider_degraded = False
        self.provider_note = ""
        self.started_at = 0
        self.running = False

        self._trade_lock = asyncio.Lock()
        self._price_queue: asyncio.Queue[Ticker] = asyncio.Queue(maxsize=2048)
        self._tasks: list[asyncio.Task[None]] = []
        self._live_gate_reason = ""

    # ---------------------------------------------------------------- startup

    async def start(self) -> None:
        if self.running:
            return

        symbols = self.settings.active_symbols
        timeframes = sorted({b.timeframe for b in self.settings.bots})

        provider, degraded, note = await create_provider(self.settings, symbols, timeframes)
        self.provider = provider
        self.provider_degraded = degraded
        self.provider_note = note
        provider.on_ticker(self._on_ticker)
        provider.on_candle(self._on_candle)

        execution, gate_reason = create_execution(self.settings, self.bus)
        self.execution = execution
        self.mode = execution.mode
        self._live_gate_reason = gate_reason

        if self.store:
            self.bus.add_listener(self._persist_frame)

        self.started_at = now_ms()
        self.running = True

        self._tasks = [
            asyncio.create_task(self._price_worker(), name="price-worker"),
            asyncio.create_task(self._pnl_broadcaster(), name="pnl-broadcaster"),
            asyncio.create_task(self._equity_recorder(), name="equity-recorder"),
        ]

        for config in self.settings.bots:
            agent = BotAgent(config, self)
            self.bots[agent.id] = agent
            if config.enabled:
                await agent.start()

        self.bus.emit(
            "system.started",
            f"JOJO is coordinating {len(self.bots)} bots · {execution.mode.value} execution · {note}",
            severity="success",
            mode=execution.mode.value, provider=provider.name, degraded=degraded,
        )
        if degraded:
            self.bus.publish(
                "market.provider_degraded",
                {"provider": provider.name, "note": note, "degraded": True},
            )
            self.bus.emit(
                "market.degraded",
                f"Market data degraded — {note}",
                severity="warning",
            )
        if self.store:
            self.store.audit("system", "start", detail={"mode": execution.mode.value, "note": note})

    async def stop(self) -> None:
        self.running = False
        for agent in self.bots.values():
            await agent.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        if self.provider:
            await self.provider.stop()
        if self.execution:
            await self.execution.close()
        self.bus.close_all()

    # ------------------------------------------------------------ market data

    def _on_ticker(self, ticker: Ticker) -> None:
        """Provider callback (sync). Hand off to the worker; never block here."""
        try:
            self._price_queue.put_nowait(ticker)
        except asyncio.QueueFull:
            # Prices are continuous — dropping a stale one is harmless, and it is
            # far better than stalling the feed.
            pass
        self.bus.publish(
            "market.tick",
            {
                "symbol": ticker.symbol, "price": ticker.price,
                "bid": ticker.bid, "ask": ticker.ask,
                "spread_bps": round(ticker.spread_bps, 4),
                "change_24h_pct": ticker.change_24h_pct,
                "ts": ticker.ts,
            },
        )

    def _on_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        self.bus.publish(
            "market.kline",
            {"symbol": symbol, "timeframe": timeframe, "candle": candle.model_dump()},
        )

    async def _price_worker(self) -> None:
        while True:
            ticker = await self._price_queue.get()
            try:
                self.portfolio.mark(ticker.symbol, ticker.price)
                await self._check_stops(ticker)
                await self._fill_resting(ticker)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("price worker failed for %s", ticker.symbol)

    async def _check_stops(self, ticker: Ticker) -> None:
        """Stop-loss and take-profit are evaluated on every tick, never throttled."""
        for position in list(self.portfolio.positions.values()):
            if position.symbol != ticker.symbol:
                continue
            hit = self._stop_hit(position, ticker.price)
            if hit is not None:
                await self.close_position(position, hit, ticker)

    @staticmethod
    def _stop_hit(position: Position, price: float) -> CloseReason | None:
        if position.side is PositionSide.LONG:
            if position.stop_loss is not None and price <= position.stop_loss:
                return CloseReason.STOP_LOSS
            if position.take_profit is not None and price >= position.take_profit:
                return CloseReason.TAKE_PROFIT
        else:
            if position.stop_loss is not None and price >= position.stop_loss:
                return CloseReason.STOP_LOSS
            if position.take_profit is not None and price <= position.take_profit:
                return CloseReason.TAKE_PROFIT
        return None

    async def _fill_resting(self, ticker: Ticker) -> None:
        assert self.execution is not None
        fills = await self.execution.on_price(ticker)
        for fill in fills:
            self.bus.publish("order.filled", {"order_id": fill.order_id, "price": fill.price,
                                              "quantity": fill.quantity, "fee": fill.fee})

    # ------------------------------------------------------------- signal flow

    async def handle_signal(self, agent: BotAgent, signal: Signal) -> None:
        ticker = self.provider.ticker(signal.symbol) if self.provider else None
        if ticker is None:
            return

        async with self._trade_lock:
            position = self.portfolio.position_for(agent.id, signal.symbol)

            if signal.action is SignalAction.CLOSE:
                if position is not None:
                    await self._close_locked(position, CloseReason.SIGNAL, ticker)
                return

            desired = (
                PositionSide.LONG if signal.action is SignalAction.LONG else PositionSide.SHORT
            )

            if position is not None:
                if position.side is desired:
                    return  # already positioned the way the signal points
                # Reversal: flatten first, then take the new side.
                await self._close_locked(position, CloseReason.SIGNAL, ticker)
                ticker = self.provider.ticker(signal.symbol) or ticker

            await self._open_locked(agent, signal, desired, ticker)

    async def _open_locked(
        self, agent: BotAgent, signal: Signal, side: PositionSide, ticker: Ticker
    ) -> Position | None:
        assert self.execution is not None
        bot = agent.config

        sizing = self.risk.size_position(bot, ticker.price, bot.stop_loss_pct)
        if not sizing.allowed:
            self._reject(agent, signal, sizing.reason)
            return None

        gate = self.risk.can_open(bot, sizing.quantity * ticker.price)
        if not gate.allowed:
            self._reject(agent, signal, gate.reason)
            return None

        order = Order(
            bot_id=bot.id,
            symbol=signal.symbol,
            side=Side.BUY if side is PositionSide.LONG else Side.SELL,
            type=OrderType.MARKET,
            quantity=sizing.quantity,
            reason=signal.reason,
        )
        self.bus.publish("order.submitted", order.model_dump(mode="json"))
        self.bus.emit(
            "order.submitted",
            f"{bot.name} submitted {order.side.value} {order.quantity:.6f} {signal.symbol}",
            severity="info", bot_id=bot.id, symbol=signal.symbol,
            order_id=order.id, side=order.side.value, quantity=order.quantity,
        )

        fill = await self.execution.place_order(order, ticker)
        if self.store:
            self.store.save_order(order)
        if fill is None:
            self.bus.publish("order.cancelled", order.model_dump(mode="json"))
            return None

        stop_loss, take_profit = self.risk.stop_levels(
            side, fill.price, bot.stop_loss_pct, bot.take_profit_pct
        )
        position = self.portfolio.open_position(
            bot_id=bot.id, symbol=signal.symbol, side=side, fill=fill,
            leverage=bot.leverage, stop_loss=stop_loss, take_profit=take_profit,
        )

        self.bus.publish("order.filled", {
            "order_id": order.id, "bot_id": bot.id, "symbol": signal.symbol,
            "side": order.side.value, "price": fill.price, "quantity": fill.quantity,
            "fee": fill.fee, "ts": fill.ts,
        })
        self.bus.publish("position.opened", position.model_dump(mode="json"))
        self.bus.emit(
            "position.opened",
            f"{bot.name} opened {side.value} {fill.quantity:.6f} {signal.symbol} "
            f"@ {fill.price:,.2f} (SL {stop_loss:,.2f} / TP {take_profit:,.2f})",
            severity="success", bot_id=bot.id, symbol=signal.symbol,
            position_id=position.id, side=side.value, price=fill.price,
            quantity=fill.quantity, leverage=bot.leverage,
        )
        if self.store:
            self.store.save_position(position)
            self.store.audit(bot.id, "open_position", position.id, {
                "symbol": signal.symbol, "side": side.value,
                "price": fill.price, "quantity": fill.quantity,
                "reason": signal.reason,
            })

        self._publish_risk()
        return position

    def _reject(self, agent: BotAgent, signal: Signal, reason: str) -> None:
        self.bus.emit(
            "order.blocked",
            f"{agent.name} blocked from {signal.action.value} {signal.symbol}: {reason}",
            severity="warning", bot_id=agent.id, symbol=signal.symbol, reason=reason,
        )

    async def close_position(
        self, position: Position, reason: CloseReason, ticker: Ticker | None = None
    ) -> None:
        async with self._trade_lock:
            if position.id not in self.portfolio.positions:
                return  # already closed by another path
            resolved = ticker or (self.provider.ticker(position.symbol) if self.provider else None)
            if resolved is None:
                return
            await self._close_locked(position, reason, resolved)

    async def _close_locked(
        self, position: Position, reason: CloseReason, ticker: Ticker
    ) -> None:
        assert self.execution is not None
        if position.id not in self.portfolio.positions:
            return

        order = Order(
            bot_id=position.bot_id,
            symbol=position.symbol,
            side=Side.SELL if position.side is PositionSide.LONG else Side.BUY,
            type=OrderType.MARKET,
            quantity=position.quantity,
            reduce_only=True,
            reason=reason.value,
        )
        self.bus.publish("order.submitted", order.model_dump(mode="json"))

        fill = await self.execution.place_order(order, ticker)
        if self.store:
            self.store.save_order(order)
        if fill is None:
            return

        trade = self.portfolio.close_position(position, fill, reason)
        bot_name = self.bots[position.bot_id].name if position.bot_id in self.bots else position.bot_id

        severity = "success" if trade.realized_pnl > 0 else "danger"
        if reason is CloseReason.STOP_LOSS:
            severity = "danger"
        elif reason is CloseReason.TAKE_PROFIT:
            severity = "success"

        self.bus.publish("order.filled", {
            "order_id": order.id, "bot_id": position.bot_id, "symbol": position.symbol,
            "side": order.side.value, "price": fill.price, "quantity": fill.quantity,
            "fee": fill.fee, "ts": fill.ts, "reduce_only": True,
        })
        self.bus.publish("position.closed", {
            "position_id": position.id, "bot_id": position.bot_id,
            "symbol": position.symbol, "reason": reason.value,
            "exit_price": fill.price, "realized_pnl": trade.realized_pnl,
        })
        self.bus.publish("trade.closed", trade.model_dump(mode="json"))
        self.bus.emit(
            "trade.closed",
            f"{bot_name} closed {position.side.value} {position.symbol} @ {fill.price:,.2f} "
            f"— {reason.value.replace('_', ' ').title()} · "
            f"{'+' if trade.realized_pnl >= 0 else ''}{trade.realized_pnl:,.2f} USDT",
            severity=severity, bot_id=position.bot_id, symbol=position.symbol,
            trade_id=trade.id, reason=reason.value, pnl=trade.realized_pnl,
            pnl_pct=trade.realized_pnl_pct,
        )
        if self.store:
            self.store.save_trade(trade)
            self.store.save_position(position, closed_at=trade.closed_at)
            self.store.audit(position.bot_id, "close_position", position.id, {
                "reason": reason.value, "pnl": trade.realized_pnl, "exit": fill.price,
            })

        self._publish_risk()

    # ------------------------------------------------------------------- risk

    def _publish_risk(self) -> None:
        state = self.risk.evaluate()
        self.bus.publish("risk.update", state.model_dump(mode="json"))

        # A breached limit halts the offending bots rather than merely warning.
        if state.breaches and not self.risk.emergency_stop:
            for breach in state.breaches:
                self.bus.emit("risk.breach", f"Risk limit breached: {breach}", severity="danger")

    async def _pnl_broadcaster(self) -> None:
        """Aggregate P&L at a fixed rate — the coalescable half of the protocol."""
        interval = 1.0 / max(self.settings.server.market_tick_hz, 0.5)
        while True:
            await asyncio.sleep(interval)
            try:
                active = sum(1 for b in self.bots.values() if b.status.value not in ("OFFLINE",))
                self.bus.publish("pnl.tick", {
                    "portfolio": self.portfolio.state(active, len(self.bots)).model_dump(mode="json"),
                    "risk": self.risk.evaluate().model_dump(mode="json"),
                    "positions": [p.model_dump(mode="json") for p in self.portfolio.positions.values()],
                    "bots": [
                        {
                            "id": b.id,
                            "status": b.status.value,
                            "realized_pnl": round(self.portfolio.bot_realized(b.id), 2),
                            "unrealized_pnl": round(self.portfolio.bot_unrealized(b.id), 2),
                            "exposure": round(self.portfolio.bot_exposure(b.id), 2),
                            "open_positions": len(self.portfolio.positions_for(b.id)),
                            "risk_level": self.risk.bot_risk(b.config)[0].value,
                            "risk_utilization": round(self.risk.bot_risk(b.config)[1], 4),
                        }
                        for b in self.bots.values()
                    ],
                    "ts": now_ms(),
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("pnl broadcast failed")

    async def _equity_recorder(self) -> None:
        while True:
            await asyncio.sleep(EQUITY_SAMPLE_SEC)
            if not self.store:
                continue
            try:
                self.store.save_equity(
                    now_ms(), self.portfolio.equity, self.portfolio.realized_pnl,
                    self.portfolio.unrealized_pnl, self.portfolio.total_exposure,
                    self.portfolio.drawdown_pct,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("equity sample failed")

    def _persist_frame(self, item: dict) -> None:
        if self.store and item.get("type") == "event.log":
            from .models import Event

            with contextlib.suppress(Exception):
                self.store.save_event(Event(**item["data"]))

    # --------------------------------------------------------------- commands

    async def command(self, bot_id: str, action: str) -> tuple[bool, str]:
        agent = self.bots.get(bot_id)
        if agent is None:
            return False, f"unknown bot {bot_id!r}"

        action = action.lower()
        if self.store:
            self.store.audit("user", f"bot.{action}", bot_id)

        if action == "start":
            self.risk.resume(bot_id)
            agent.error = None
            await agent.start()
            agent.resume()
            return True, f"{agent.name} started"
        if action == "pause":
            agent.pause()
            self.bus.emit("bot.paused", f"{agent.name} paused.", severity="warning", bot_id=bot_id)
            return True, f"{agent.name} paused"
        if action == "resume":
            agent.resume()
            self.bus.emit("bot.resumed", f"{agent.name} resumed.", severity="success", bot_id=bot_id)
            return True, f"{agent.name} resumed"
        if action == "stop":
            for position in self.portfolio.positions_for(bot_id):
                await self.close_position(position, CloseReason.MANUAL)
            await agent.stop()
            self.bus.emit(
                "bot.stopped", f"{agent.name} stopped and flattened.",
                severity="warning", bot_id=bot_id,
            )
            return True, f"{agent.name} stopped"
        if action == "flatten":
            for position in self.portfolio.positions_for(bot_id):
                await self.close_position(position, CloseReason.MANUAL)
            return True, f"{agent.name} flattened"

        return False, f"unknown action {action!r}"

    async def emergency_stop(self, actor: str = "user") -> None:
        """Flatten everything and halt the desk. The world goes red."""
        self.risk.trigger_emergency_stop()
        self.bus.emit(
            "system.emergency_stop",
            "EMERGENCY STOP — all bots halted, all positions being flattened.",
            severity="critical",
        )
        self.bus.publish("system.emergency_stop", {"active": True, "ts": now_ms()})

        for position in list(self.portfolio.positions.values()):
            await self.close_position(position, CloseReason.EMERGENCY_STOP)

        for agent in self.bots.values():
            agent.halt("emergency stop")
            self.risk.halt(agent.id)

        if self.store:
            self.store.audit(actor, "emergency_stop", detail={"positions_closed": True})
        self._publish_risk()

    async def resume_system(self, actor: str = "user") -> None:
        self.risk.clear_emergency_stop()
        for agent in self.bots.values():
            agent.error = None
            if agent.config.enabled:
                await agent.start()
                agent.resume()
        self.bus.emit(
            "system.resumed", "Emergency stop cleared — JOJO is coordinating again.",
            severity="success",
        )
        self.bus.publish("system.emergency_stop", {"active": False, "ts": now_ms()})
        if self.store:
            self.store.audit(actor, "resume")
        self._publish_risk()

    # --------------------------------------------------------------- snapshot

    def system_state(self) -> SystemState:
        return SystemState(
            mode=self.mode,
            provider=self.provider.name if self.provider else "none",
            provider_degraded=self.provider_degraded,
            provider_note=self.provider_note,
            running=self.running,
            emergency_stop=self.risk.emergency_stop,
            started_at=self.started_at,
        )

    def snapshot(self) -> Snapshot:
        active = sum(1 for b in self.bots.values() if b.status.value != "OFFLINE")
        tickers: dict[str, Ticker] = {}
        candles: dict[str, list[Candle]] = {}
        if self.provider:
            for bot in self.settings.bots:
                ticker = self.provider.ticker(bot.symbol)
                if ticker:
                    tickers[bot.symbol] = ticker
                key = f"{bot.symbol}|{bot.timeframe}"
                if key not in candles:
                    candles[key] = self.provider.candles(bot.symbol, bot.timeframe, limit=180)

        orders: list[Order] = []
        if isinstance(getattr(self.execution, "resting_orders", None), list):
            orders = self.execution.resting_orders  # type: ignore[union-attr]

        return Snapshot(
            system=self.system_state(),
            portfolio=self.portfolio.state(active, len(self.bots)),
            risk=self.risk.evaluate(),
            bots=[b.state() for b in self.bots.values()],
            positions=list(self.portfolio.positions.values()),
            orders=orders,
            trades=self.portfolio.trades[-60:],
            events=self.bus.recent_events(120),
            tickers=tickers,
            candles=candles,
        )

    @property
    def live_gate_reason(self) -> str:
        return self._live_gate_reason
