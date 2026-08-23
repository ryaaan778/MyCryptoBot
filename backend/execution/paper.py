"""Paper matching engine.

Fills against the live book rather than the mid price: a buy pays the ask, a
sell hits the bid, and both take configurable slippage on top. Fees are charged
on notional. This is what makes paper P&L resemble live P&L instead of
flattering it.
"""

from __future__ import annotations

import logging

from ..models import (
    ExecutionMode,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
    now_ms,
)
from .base import ExecutionAdapter, Fill

logger = logging.getLogger("jojo.execution.paper")


class PaperExecution(ExecutionAdapter):
    mode = ExecutionMode.PAPER

    def __init__(self, settings, bus) -> None:  # type: ignore[no-untyped-def]
        self.settings = settings
        self.bus = bus
        self.fee_rate = settings.trading.fee_rate
        self.slippage_bps = settings.trading.slippage_bps
        self._resting: dict[str, Order] = {}

    # ---- pricing -----------------------------------------------------------

    def fill_price(self, side: Side, ticker: Ticker) -> float:
        """Cross the spread, then pay slippage in the direction that hurts."""
        base = ticker.ask if side is Side.BUY else ticker.bid
        if base <= 0:
            base = ticker.price
        slip = base * (self.slippage_bps / 10_000.0)
        return base + slip if side is Side.BUY else base - slip

    def fee_for(self, price: float, quantity: float) -> float:
        return abs(price * quantity) * self.fee_rate

    # ---- order handling ----------------------------------------------------

    async def place_order(self, order: Order, ticker: Ticker) -> Fill | None:
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "quantity must be positive"
            order.updated_at = now_ms()
            return None

        order.status = OrderStatus.SUBMITTED
        order.updated_at = now_ms()

        if order.type is OrderType.MARKET:
            return self._execute(order, self.fill_price(order.side, ticker))

        # A limit order that is already marketable fills straight away;
        # otherwise it rests until on_price() sees the market come to it.
        if order.price is not None and self._crosses(order, ticker):
            return self._execute(order, self._limit_fill_price(order, ticker))

        self._resting[order.id] = order
        return None

    def _crosses(self, order: Order, ticker: Ticker) -> bool:
        if order.price is None:
            return False
        if order.side is Side.BUY:
            return ticker.ask <= order.price
        return ticker.bid >= order.price

    def _limit_fill_price(self, order: Order, ticker: Ticker) -> float:
        """A resting limit order fills at its own price, never worse."""
        assert order.price is not None
        market = ticker.ask if order.side is Side.BUY else ticker.bid
        return min(order.price, market) if order.side is Side.BUY else max(order.price, market)

    def _execute(self, order: Order, price: float) -> Fill:
        quantity = order.remaining
        fee = self.fee_for(price, quantity)
        filled_notional = order.average_fill_price * order.filled_quantity + price * quantity
        order.filled_quantity += quantity
        order.average_fill_price = filled_notional / order.filled_quantity
        order.fee += fee
        order.status = OrderStatus.FILLED
        order.updated_at = now_ms()
        self._resting.pop(order.id, None)
        return Fill(order_id=order.id, price=price, quantity=quantity, fee=fee)

    async def cancel_order(self, order: Order) -> bool:
        if self._resting.pop(order.id, None) is None:
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = now_ms()
        return True

    async def on_price(self, ticker: Ticker) -> list[Fill]:
        fills: list[Fill] = []
        for order in [o for o in self._resting.values() if o.symbol == ticker.symbol]:
            if self._crosses(order, ticker):
                fills.append(self._execute(order, self._limit_fill_price(order, ticker)))
        return fills

    @property
    def resting_orders(self) -> list[Order]:
        return list(self._resting.values())
