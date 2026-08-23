"""Execution adapter interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..models import ExecutionMode, Order, Ticker, now_ms


@dataclass
class Fill:
    order_id: str
    price: float
    quantity: float
    fee: float
    ts: int = 0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = now_ms()

    @property
    def notional(self) -> float:
        return self.price * self.quantity


class ExecutionAdapter(abc.ABC):
    """Turns an :class:`Order` into zero or more :class:`Fill`s."""

    mode: ExecutionMode = ExecutionMode.PAPER

    @abc.abstractmethod
    async def place_order(self, order: Order, ticker: Ticker) -> Fill | None:
        """Submit ``order``. Returns the fill, or None if it rests unfilled."""

    @abc.abstractmethod
    async def cancel_order(self, order: Order) -> bool: ...

    async def on_price(self, ticker: Ticker) -> list[Fill]:
        """Give resting orders a chance to fill against a new price."""
        return []

    async def close(self) -> None:
        return None

    @property
    def is_live(self) -> bool:
        return self.mode is ExecutionMode.LIVE
