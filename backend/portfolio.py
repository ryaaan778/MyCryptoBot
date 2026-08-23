"""Position and cash bookkeeping.

One portfolio backs the whole desk; each bot gets a slice of it. Margin is
reserved on open and released on close, so ``cash`` is genuinely spendable and
the exposure figure the risk engine reads is real rather than notional
hand-waving.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .execution.base import Fill
from .models import (
    CloseReason,
    Position,
    PositionSide,
    PortfolioState,
    Trade,
    now_ms,
)

logger = logging.getLogger("jojo.portfolio")


class Portfolio:
    def __init__(self, starting_equity: float) -> None:
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self.reserved_margin = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.marks: dict[str, float] = {}
        self.peak_equity = starting_equity
        self.session_start_equity = starting_equity
        self.session_start_ts = now_ms()
        self._bot_realized: dict[str, float] = defaultdict(float)
        self._bot_trades: dict[str, list[Trade]] = defaultdict(list)

    # ---- marking -----------------------------------------------------------

    def mark(self, symbol: str, price: float) -> list[Position]:
        """Update the mark for ``symbol``; returns the positions it touched."""
        self.marks[symbol] = price
        touched: list[Position] = []
        for position in self.positions.values():
            if position.symbol != symbol:
                continue
            position.mark_price = price
            position.unrealized_pnl, position.unrealized_pnl_pct = position.compute_pnl(price)
            position.updated_at = now_ms()
            touched.append(position)
        equity = self.equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        return touched

    # ---- lifecycle ---------------------------------------------------------

    def open_position(
        self,
        *,
        bot_id: str,
        symbol: str,
        side: PositionSide,
        fill: Fill,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        notional = fill.price * fill.quantity
        margin = notional / max(leverage, 1e-9)

        self.cash -= margin + fill.fee
        self.reserved_margin += margin
        self.fees_paid += fill.fee

        position = Position(
            bot_id=bot_id,
            symbol=symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            mark_price=fill.price,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin=margin,
            fees_paid=fill.fee,
        )
        self.positions[position.id] = position
        self.marks[symbol] = fill.price
        return position

    def close_position(
        self, position: Position, fill: Fill, reason: CloseReason = CloseReason.SIGNAL
    ) -> Trade:
        direction = 1.0 if position.side is PositionSide.LONG else -1.0
        gross = (fill.price - position.entry_price) * fill.quantity * direction
        total_fees = position.fees_paid + fill.fee
        net = gross - fill.fee

        self.cash += position.margin + net
        self.reserved_margin -= position.margin
        self.realized_pnl += gross - fill.fee
        self.fees_paid += fill.fee
        self._bot_realized[position.bot_id] += net

        trade = Trade(
            bot_id=position.bot_id,
            symbol=position.symbol,
            side=position.side,
            quantity=fill.quantity,
            entry_price=position.entry_price,
            exit_price=fill.price,
            leverage=position.leverage,
            realized_pnl=round(net, 8),
            realized_pnl_pct=round((net / position.margin * 100.0) if position.margin else 0.0, 6),
            fees=round(total_fees, 8),
            reason=reason,
            opened_at=position.opened_at,
        )
        self.trades.append(trade)
        self._bot_trades[position.bot_id].append(trade)
        self.positions.pop(position.id, None)
        return trade

    # ---- queries -----------------------------------------------------------

    def position_for(self, bot_id: str, symbol: str | None = None) -> Position | None:
        for position in self.positions.values():
            if position.bot_id == bot_id and (symbol is None or position.symbol == symbol):
                return position
        return None

    def positions_for(self, bot_id: str) -> list[Position]:
        return [p for p in self.positions.values() if p.bot_id == bot_id]

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.reserved_margin + self.unrealized_pnl

    @property
    def total_exposure(self) -> float:
        return sum(abs(p.notional) for p in self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)

    def bot_realized(self, bot_id: str) -> float:
        return self._bot_realized.get(bot_id, 0.0)

    def bot_unrealized(self, bot_id: str) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.bot_id == bot_id)

    def bot_exposure(self, bot_id: str) -> float:
        return sum(abs(p.notional) for p in self.positions.values() if p.bot_id == bot_id)

    def bot_trades(self, bot_id: str) -> list[Trade]:
        return self._bot_trades.get(bot_id, [])

    def bot_win_rate(self, bot_id: str) -> tuple[int, int, float]:
        trades = self.bot_trades(bot_id)
        won = sum(1 for t in trades if t.is_win)
        rate = (won / len(trades) * 100.0) if trades else 0.0
        return len(trades), won, rate

    def state(self, active_bots: int = 0, total_bots: int = 0) -> PortfolioState:
        equity = self.equity
        unrealized = self.unrealized_pnl
        total_pnl = equity - self.starting_equity
        won = sum(1 for t in self.trades if t.is_win)
        return PortfolioState(
            starting_equity=round(self.starting_equity, 2),
            equity=round(equity, 2),
            cash=round(self.cash, 2),
            realized_pnl=round(self.realized_pnl, 2),
            unrealized_pnl=round(unrealized, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(
                total_pnl / self.starting_equity * 100.0 if self.starting_equity else 0.0, 4
            ),
            today_pnl=round(equity - self.session_start_equity, 2),
            today_pnl_pct=round(
                (equity - self.session_start_equity) / self.session_start_equity * 100.0
                if self.session_start_equity else 0.0,
                4,
            ),
            peak_equity=round(self.peak_equity, 2),
            open_positions=len(self.positions),
            total_exposure=round(self.total_exposure, 2),
            active_bots=active_bots,
            total_bots=total_bots,
            trades_total=len(self.trades),
            win_rate=round(won / len(self.trades) * 100.0 if self.trades else 0.0, 2),
        )
