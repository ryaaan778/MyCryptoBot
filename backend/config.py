"""Configuration.

The existing ``config.json`` at the repo root stays the source of truth for
symbols, risk limits and strategy parameters. This module layers the bot roster
and server settings on top of it, and lets environment variables override
anything that matters operationally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .models import BotConfig, BotPersona

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.json"

# The magic phrase that must be present in the environment before any real
# order can be placed. See execution/live_ccxt.py.
LIVE_CONFIRM_ENV = "LIVE_TRADING_CONFIRM"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_THE_RISK"


def to_display_symbol(symbol: str) -> str:
    """``BTC/USDT`` -> ``BTCUSDT`` (what the world's tickers show)."""
    return symbol.replace("/", "")


def to_ccxt_symbol(symbol: str) -> str:
    """``BTCUSDT`` -> ``BTC/USDT`` when unambiguous; otherwise pass through."""
    if "/" in symbol:
        return symbol
    for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[: -len(quote)]}/{quote}"
    return symbol


# --------------------------------------------------------------------------
# The roster
# --------------------------------------------------------------------------

# Each bot is a distinct strategy with a distinct temperament, and the persona
# block drives its voxel character, palette and district placement in the world.
# District coordinates are world-space [x, z] around the headquarters at origin.
DEFAULT_ROSTER: list[dict[str, Any]] = [
    {
        "id": "jonathan",
        "name": "JONATHAN",
        "strategy": "momentum",
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "allocation": 0.22,
        "leverage": 3.0,
        "risk_per_trade": 0.015,
        "stop_loss_pct": 1.2,
        "take_profit_pct": 4.0,
        "interval_sec": 4.0,
        "persona": {
            "title": "THE FOUNDATION",
            "stand": "STEADY RESOLVE",
            "quote": "A true trader does not flinch.",
            "palette": ["#2f5fd0", "#7ba7ff", "#e8c56a", "#1a2f5e", "#f2e7cf"],
            "accent": "#7ba7ff",
            "district": [0.0, -46.0],
            "facing": 0.0,
        },
    },
    {
        "id": "joseph",
        "name": "JOSEPH",
        "strategy": "volatility_breakout",
        "symbol": "ETH/USDT",
        "timeframe": "15m",
        "allocation": 0.2,
        "leverage": 4.0,
        "risk_per_trade": 0.02,
        "stop_loss_pct": 1.5,
        "take_profit_pct": 6.0,
        "interval_sec": 3.5,
        "persona": {
            "title": "THE TRICKSTER",
            "stand": "NEXT MOVE FORESIGHT",
            "quote": "Your next trade is 'I'll buy the breakout'.",
            "palette": ["#3fa86b", "#8ce0a8", "#e08a3c", "#1e5c3c", "#f4ead6"],
            "accent": "#8ce0a8",
            "district": [-44.0, -22.0],
            "facing": 1.05,
        },
    },
    {
        "id": "jotaro",
        "name": "JOTARO",
        "strategy": "scalping",
        "symbol": "BTC/USDT",
        "timeframe": "5m",
        "allocation": 0.24,
        "leverage": 5.0,
        "risk_per_trade": 0.02,
        "stop_loss_pct": 0.8,
        "take_profit_pct": 2.2,
        "interval_sec": 2.0,
        "persona": {
            "title": "THE STRIKER",
            "stand": "ORA RUSH",
            "quote": "Yare yare daze... another fill.",
            "palette": ["#1b1d2e", "#4b3f8f", "#c9a227", "#2e3350", "#e6e8f5"],
            "accent": "#c9a227",
            "district": [44.0, -22.0],
            "facing": -1.05,
        },
    },
    {
        "id": "jolyan",
        "name": "JOLYAN",
        "strategy": "hybrid",
        "symbol": "SOL/USDT",
        "timeframe": "5m",
        "allocation": 0.2,
        "leverage": 4.0,
        "risk_per_trade": 0.02,
        "stop_loss_pct": 1.0,
        "take_profit_pct": 5.0,
        "interval_sec": 3.0,
        "persona": {
            "title": "THE UNBOUND",
            "stand": "STRING THEORY",
            "quote": "Every thread leads somewhere.",
            "palette": ["#3ec9a7", "#a8f0dc", "#f2f2f2", "#1d7a66", "#ffd9ec"],
            "accent": "#a8f0dc",
            "district": [-44.0, 26.0],
            "facing": 2.1,
        },
    },
    {
        "id": "kira",
        "name": "KIRA",
        "strategy": "mean_reversion",
        "symbol": "BNB/USDT",
        "timeframe": "15m",
        "allocation": 0.14,
        "leverage": 8.0,
        "risk_per_trade": 0.03,
        "stop_loss_pct": 0.9,
        "take_profit_pct": 3.5,
        "interval_sec": 5.0,
        "persona": {
            "title": "THE QUIET ONE",
            "stand": "DEADLY PATIENCE",
            "quote": "I simply wish to trade in peace.",
            "palette": ["#d94f9c", "#f7a8d0", "#2fb8a8", "#5a1f47", "#f0e6ef"],
            "accent": "#f7a8d0",
            "district": [44.0, 26.0],
            "facing": -2.1,
        },
    },
]

# JOJO is the orchestrator: it holds no positions and runs no strategy.
JOJO_PERSONA = BotPersona(
    title="THE MASTER",
    stand="COMMAND OVER ALL",
    quote="Every bot answers to me.",
    palette=["#f0c04a", "#ffe9a8", "#8b2f8f", "#2a1b3d", "#ffffff"],
    accent="#f0c04a",
    district=[0.0, 0.0],
    facing=0.0,
)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class ExchangeSettings(BaseModel):
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class TradingSettings(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h"])
    base_currency: str = "USDT"
    leverage: float = 10.0
    live_enabled: bool = False
    fee_rate: float = 0.0004        # 4 bps taker, roughly Binance futures
    slippage_bps: float = 2.0


class RiskSettings(BaseModel):
    max_risk_per_trade: float = 0.1
    # config.json's ``max_open_trades`` was written for a single-bot system, so
    # it is the *per-bot* cap. The desk-wide cap is separate — otherwise three
    # bots fill the book and the other two can never trade at all.
    max_open_trades: int = 3
    max_portfolio_positions: int = 0    # 0 => derive as max_open_trades * bot count
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 7.0
    max_exposure_pct: float = 300.0
    drawdown_limit_pct: float = 25.0
    elevated_utilization: float = 0.5
    high_utilization: float = 0.75
    critical_utilization: float = 0.92


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    market_tick_hz: float = 4.0     # coalescing rate for decorative price ticks
    event_buffer: int = 400
    candle_history: int = 240
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )


class LLMSettings(BaseModel):
    """Model-driven trading. Off by default — it costs money to turn on.

    ``decide_every_sec`` is the one number worth thinking about before you run
    this. It is the interval between reasoning calls *per bot*, so five bots at
    300s is roughly 1,440 calls a day. ``daily_call_budget`` is the hard stop
    underneath it, per bot, in case a misconfiguration would otherwise bill you
    all night.
    """

    enabled: bool = False
    #: Put every bot on the model, overriding each one's configured strategy.
    #: The five keep their own markets, personas, memories and budgets — this
    #: only changes what does the deciding.
    all_bots: bool = False
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    api_key: str = ""                   # blank => read ANTHROPIC_API_KEY
    decide_every_sec: float = 300.0     # per bot; the API cadence, not the tick
    stance_ttl_sec: float = 1800.0      # a stance older than this is not traded
    min_confidence: float = 0.5         # below this, a stance is only an opinion
    web_search: bool = True
    max_searches: int = 4               # per decision
    max_tokens: int = 4000
    effort: str = "medium"              # low | medium | high
    memory_window: int = 20             # past decisions fed back into the prompt
    daily_call_budget: int = 500        # per bot, hard stop
    timeout_sec: float = 120.0


class AllocatorSettings(BaseModel):
    """JOJO moving capital toward whoever is earning it.

    Off by default: it rewrites bot allocations at runtime, and that should be a
    decision you make rather than something that starts happening.
    """

    enabled: bool = False
    interval_sec: float = 900.0     # how often to re-score the roster
    prior_trades: int = 20          # shrinkage strength; higher = more patient
    min_trades_for_conviction: int = 25   # trades needed to exceed an even share
    floor: float = 0.05             # nobody is starved to zero
    ceiling: float = 0.40           # nobody takes the whole desk
    total: float = 1.0              # share of equity available to deploy
    min_change: float = 0.01        # ignore reshuffles smaller than this

    def to_config(self) -> "AllocatorConfig":
        from .allocator import AllocatorConfig

        return AllocatorConfig(
            prior_trades=self.prior_trades,
            min_trades_for_conviction=self.min_trades_for_conviction,
            floor=self.floor, ceiling=self.ceiling, total=self.total,
        )


class Settings(BaseModel):
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    allocator: AllocatorSettings = Field(default_factory=AllocatorSettings)
    strategy_parameters: dict[str, float] = Field(default_factory=dict)
    bots: list[BotConfig] = Field(default_factory=list)

    provider: str = "auto"          # auto | binance | simulated
    initial_balance: float = 10_000.0
    data_dir: str = "trading_bot_data"
    seed: int = 20250328

    # ---- live-trading gate -------------------------------------------------

    def live_gate_status(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. All three conditions must hold."""
        if not self.trading.live_enabled:
            return False, "trading.live_enabled is false in config.json"
        if not (self.exchange.api_key and self.exchange.api_secret):
            return False, "exchange API key/secret missing from config.json"
        if os.environ.get(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
            return False, f"{LIVE_CONFIRM_ENV} env var is not set to {LIVE_CONFIRM_VALUE}"
        return True, "live trading gate open"

    @property
    def live_allowed(self) -> bool:
        return self.live_gate_status()[0]

    def bot(self, bot_id: str) -> BotConfig | None:
        return next((b for b in self.bots if b.id == bot_id), None)

    @property
    def active_symbols(self) -> list[str]:
        seen: list[str] = []
        for b in self.bots:
            if b.symbol not in seen:
                seen.append(b.symbol)
        return seen


def _coerce_roster(raw: list[dict[str, Any]], risk: RiskSettings) -> list[BotConfig]:
    bots: list[BotConfig] = []
    for entry in raw:
        data = dict(entry)
        persona = BotPersona(**data.pop("persona", {}))
        data.setdefault("max_positions", risk.max_open_trades)
        bots.append(BotConfig(persona=persona, **data))
    return bots


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load ``config.json``, apply env overrides, then explicit overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = json.loads(path.read_text())

    exchange = ExchangeSettings(**raw.get("exchange", {}))
    trading_raw = dict(raw.get("trading", {}))
    trading = TradingSettings(**trading_raw)

    rm = dict(raw.get("risk_management", {}))
    risk = RiskSettings(**{k: v for k, v in rm.items() if k in RiskSettings.model_fields})
    for key in ("max_exposure_pct", "drawdown_limit_pct"):
        if key in raw.get("risk_management", {}):
            setattr(risk, key, raw["risk_management"][key])

    server = ServerSettings(**raw.get("server", {}))
    roster = raw.get("bots") or DEFAULT_ROSTER
    bots = _coerce_roster(roster, risk)
    if risk.max_portfolio_positions <= 0:
        risk.max_portfolio_positions = max(risk.max_open_trades, risk.max_open_trades * len(bots))

    settings = Settings(
        exchange=exchange,
        trading=trading,
        risk=risk,
        server=server,
        strategy_parameters=raw.get("strategy_parameters", {}),
        bots=bots,
        initial_balance=float(raw.get("initial_balance", 10_000.0)),
        data_dir=raw.get("database", {}).get("path", "trading_bot_data"),
    )

    # Environment overrides — operational knobs only, never credentials-by-default.
    env_map = {
        "JOJO_PROVIDER": ("provider", str),
        "JOJO_PORT": ("server.port", int),
        "JOJO_HOST": ("server.host", str),
        "JOJO_SEED": ("seed", int),
        "JOJO_INITIAL_BALANCE": ("initial_balance", float),
        "JOJO_DATA_DIR": ("data_dir", str),
    }
    for env_key, (dotted, caster) in env_map.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        target: Any = settings
        *parents, leaf = dotted.split(".")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, caster(value))

    if os.environ.get("BINANCE_API_KEY"):
        settings.exchange.api_key = os.environ["BINANCE_API_KEY"]
    if os.environ.get("BINANCE_API_SECRET"):
        settings.exchange.api_secret = os.environ["BINANCE_API_SECRET"]

    for dotted, value in (overrides or {}).items():
        target = settings
        *parents, leaf = dotted.split(".")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)

    return settings
