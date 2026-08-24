"""Domain types shared by the engine, the API and the wire protocol.

Everything the frontend receives is one of these models serialised to JSON, so
field names here are also the wire field names.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_ms() -> int:
    """Milliseconds since epoch — the single timestamp unit across the system."""
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class BotStatus(str, Enum):
    OFFLINE = "OFFLINE"
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    TRADING = "TRADING"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    # Research states. A bot in one of these is not trading — it is off the desk
    # working on a hypothesis, which the world should show as visibly different
    # from being idle.
    RESEARCHING = "RESEARCHING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CloseReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    LIQUIDATION = "LIQUIDATION"
    MANUAL = "MANUAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------


class Candle(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_tuple(self) -> tuple[int, float, float, float, float, float]:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


class Ticker(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float = 0.0
    change_24h_pct: float = 0.0
    ts: int = Field(default_factory=now_ms)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.price * 10_000 if self.price else 0.0


# --------------------------------------------------------------------------
# Trading
# --------------------------------------------------------------------------


class Signal(BaseModel):
    bot_id: str
    symbol: str
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    strategy: str = ""
    indicators: dict[str, float] = Field(default_factory=dict)
    ts: int = Field(default_factory=now_ms)


class Order(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ord"))
    bot_id: str
    symbol: str
    side: Side
    type: OrderType = OrderType.MARKET
    quantity: float
    price: float | None = None          # limit price; None for market orders
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fee: float = 0.0
    reduce_only: bool = False
    reason: str = ""
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)


class Position(BaseModel):
    id: str = Field(default_factory=lambda: new_id("pos"))
    bot_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float
    leverage: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    margin: float = 0.0
    fees_paid: float = 0.0
    opened_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)

    @property
    def notional(self) -> float:
        return self.quantity * self.mark_price

    def compute_pnl(self, mark: float) -> tuple[float, float]:
        """Unrealised PnL in quote currency and as a percentage of margin."""
        direction = 1.0 if self.side is PositionSide.LONG else -1.0
        pnl = (mark - self.entry_price) * self.quantity * direction
        pct = (pnl / self.margin * 100.0) if self.margin else 0.0
        return pnl, pct


class Trade(BaseModel):
    """A closed round-trip. This is what win rate and realised PnL are computed from."""

    id: str = Field(default_factory=lambda: new_id("trd"))
    bot_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float
    leverage: float = 1.0
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    fees: float = 0.0
    reason: CloseReason = CloseReason.SIGNAL
    opened_at: int = 0
    closed_at: int = Field(default_factory=now_ms)

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


# --------------------------------------------------------------------------
# Bots & portfolio
# --------------------------------------------------------------------------


class BotPersona(BaseModel):
    """Presentation metadata. The world reads this; the engine ignores it."""

    title: str = ""
    stand: str = ""
    quote: str = ""
    palette: list[str] = Field(default_factory=list)
    accent: str = "#ffffff"
    district: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    facing: float = 0.0


class BotConfig(BaseModel):
    id: str
    name: str
    strategy: str
    symbol: str
    timeframe: str = "5m"
    allocation: float = 0.2           # fraction of portfolio equity
    leverage: float = 1.0
    max_positions: int = 1
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 7.0
    interval_sec: float = 3.0
    enabled: bool = True
    persona: BotPersona = Field(default_factory=BotPersona)


class BotState(BaseModel):
    id: str
    name: str
    strategy: str
    symbol: str
    timeframe: str
    status: BotStatus = BotStatus.OFFLINE
    allocation: float = 0.0
    equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    exposure: float = 0.0
    leverage: float = 1.0
    trades_total: int = 0
    trades_won: int = 0
    win_rate: float = 0.0
    open_positions: int = 0
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_utilization: float = 0.0     # 0..1 of the bot's exposure budget
    last_signal: Signal | None = None
    last_heartbeat: int = 0
    error: str | None = None
    persona: BotPersona = Field(default_factory=BotPersona)


class AgentResearch(BaseModel):
    """One agent's research standing, read from the research tables.

    Everything here is *research* status. A promoted champion is a champion
    inside the research system and nothing more — it has not been deployed, and
    the live-trading gate is untouched by anything in this model.
    """

    agent: str
    champion_policy: str | None = None
    champion_version: int | None = None
    challenger_count: int = 0
    experiments_total: int = 0
    experiments_rejected: int = 0
    current_experiment: str | None = None
    current_status: str | None = None
    hypothesis: str | None = None
    paper_allocation: float = 0.0
    bias_drift: float = 0.0
    last_updated: int = 0

    @property
    def is_busy(self) -> bool:
        return self.current_status in {"PROPOSED", "TRAINING", "VALIDATING"}


class ResearchState(BaseModel):
    """The research half of the desk, for display.

    ``synthetic_only`` is load-bearing rather than decorative. Simulator results
    must never be presented as evidence about live markets, and the surface that
    shows them is exactly where that mistake would be made — so the flag travels
    with the numbers and the interface is expected to say so out loud.
    """

    available: bool = False
    agents: list[AgentResearch] = Field(default_factory=list)
    datasets: int = 0
    policies: int = 0
    experiments: int = 0
    sources: list[str] = Field(default_factory=list)
    synthetic_only: bool = True
    paper_capital_deployed: float = 0.0
    updated_at: int = Field(default_factory=now_ms)


class RiskState(BaseModel):
    level: RiskLevel = RiskLevel.SAFE
    total_exposure: float = 0.0
    exposure_pct: float = 0.0
    max_exposure_pct: float = 300.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_limit_pct: float = 25.0
    open_positions: int = 0
    max_open_positions: int = 3
    utilization: float = 0.0          # 0..1, drives the voxel risk meter
    emergency_stop: bool = False
    halted_bots: list[str] = Field(default_factory=list)
    breaches: list[str] = Field(default_factory=list)
    ts: int = Field(default_factory=now_ms)


class PortfolioState(BaseModel):
    starting_equity: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    today_pnl: float = 0.0
    today_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    open_positions: int = 0
    total_exposure: float = 0.0
    active_bots: int = 0
    total_bots: int = 0
    trades_total: int = 0
    win_rate: float = 0.0
    ts: int = Field(default_factory=now_ms)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


EventSeverity = Literal["info", "success", "warning", "danger", "critical"]


class Event(BaseModel):
    """One line in the JOJO EVENT LOG, and one row in the audit table."""

    id: str = Field(default_factory=lambda: new_id("evt"))
    type: str
    bot_id: str | None = None
    symbol: str | None = None
    message: str = ""
    severity: EventSeverity = "info"
    data: dict[str, Any] = Field(default_factory=dict)
    ts: int = Field(default_factory=now_ms)


class SystemState(BaseModel):
    mode: ExecutionMode = ExecutionMode.PAPER
    provider: str = "simulated"
    provider_degraded: bool = False
    provider_note: str = ""
    running: bool = False
    emergency_stop: bool = False
    started_at: int = 0
    server_time: int = Field(default_factory=now_ms)


class Snapshot(BaseModel):
    """The complete world state, sent as the first frame on every WS connect."""

    system: SystemState
    portfolio: PortfolioState
    risk: RiskState
    bots: list[BotState]
    positions: list[Position]
    orders: list[Order]
    trades: list[Trade]
    events: list[Event]
    tickers: dict[str, Ticker]
    candles: dict[str, list[Candle]]
    equity_curve: list[dict[str, float]] = Field(default_factory=list)
