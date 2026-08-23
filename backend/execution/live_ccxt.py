"""Live execution against a real exchange, behind a deliberate three-part gate.

Constructing this class places real orders with real money. It refuses to
initialise unless **all three** conditions hold:

1. ``trading.live_enabled`` is true in ``config.json``
2. an API key *and* secret are configured
3. the environment carries ``LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK``

Any single missing condition raises :class:`LiveTradingRefused`, and the caller
falls back to paper execution. The gate is re-checked on every single order, not
just at construction — a config reload can never quietly open it mid-session.
"""

from __future__ import annotations

import logging

from ..models import (
    ExecutionMode,
    Order,
    OrderStatus,
    OrderType,
    Ticker,
    now_ms,
)
from .base import ExecutionAdapter, Fill

logger = logging.getLogger("jojo.execution.live")


class LiveTradingRefused(RuntimeError):
    """Raised when the live-trading gate is not fully open."""


class LiveExecution(ExecutionAdapter):
    mode = ExecutionMode.LIVE

    def __init__(self, settings, bus) -> None:  # type: ignore[no-untyped-def]
        allowed, reason = settings.live_gate_status()
        if not allowed:
            raise LiveTradingRefused(reason)

        self.settings = settings
        self.bus = bus
        self._exchange = None

        logger.warning(
            "LIVE TRADING ENABLED on %s (testnet=%s) — orders will use real funds",
            settings.exchange.name, settings.exchange.testnet,
        )

    def _client(self):  # type: ignore[no-untyped-def]
        if self._exchange is None:
            import ccxt.async_support as ccxt

            exchange_cls = getattr(ccxt, self.settings.exchange.name)
            self._exchange = exchange_cls({
                "apiKey": self.settings.exchange.api_key,
                "secret": self.settings.exchange.api_secret,
                "enableRateLimit": True,
            })
            if self.settings.exchange.testnet:
                self._exchange.set_sandbox_mode(True)
        return self._exchange

    def _assert_gate(self) -> None:
        allowed, reason = self.settings.live_gate_status()
        if not allowed:
            raise LiveTradingRefused(f"live gate closed at order time: {reason}")

    async def place_order(self, order: Order, ticker: Ticker) -> Fill | None:
        self._assert_gate()
        client = self._client()
        params = {"reduceOnly": True} if order.reduce_only else {}

        try:
            if order.type is OrderType.MARKET:
                raw = await client.create_order(
                    order.symbol, "market", order.side.value.lower(),
                    order.quantity, None, params,
                )
            else:
                raw = await client.create_order(
                    order.symbol, "limit", order.side.value.lower(),
                    order.quantity, order.price, params,
                )
        except Exception as exc:
            order.status = OrderStatus.REJECTED
            order.reason = f"exchange rejected: {exc}"
            order.updated_at = now_ms()
            logger.error("live order rejected: %s", exc)
            self.bus.emit(
                "order.rejected", f"Exchange rejected order: {exc}",
                severity="danger", bot_id=order.bot_id, symbol=order.symbol,
            )
            return None

        filled = float(raw.get("filled") or 0.0)
        average = float(raw.get("average") or raw.get("price") or ticker.price)
        fee_info = raw.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)

        order.filled_quantity = filled
        order.average_fill_price = average
        order.fee = fee
        order.updated_at = now_ms()

        if filled <= 0:
            order.status = OrderStatus.SUBMITTED
            return None

        order.status = (
            OrderStatus.FILLED if filled >= order.quantity - 1e-12 else OrderStatus.PARTIAL
        )
        return Fill(order_id=order.id, price=average, quantity=filled, fee=fee)

    async def cancel_order(self, order: Order) -> bool:
        self._assert_gate()
        try:
            await self._client().cancel_order(order.id, order.symbol)
        except Exception as exc:
            logger.error("cancel failed for %s: %s", order.id, exc)
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = now_ms()
        return True

    async def close(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            finally:
                self._exchange = None
