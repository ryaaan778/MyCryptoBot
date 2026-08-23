"""Paper matching engine, portfolio accounting, and the live-trading gate."""

from __future__ import annotations

import pytest

from backend.eventbus import EventBus
from backend.execution import create_execution
from backend.execution.live_ccxt import LiveExecution, LiveTradingRefused
from backend.execution.paper import PaperExecution
from backend.models import (
    CloseReason,
    ExecutionMode,
    Order,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    Ticker,
)
from backend.portfolio import Portfolio


def ticker(price: float = 100.0, half_spread: float = 0.05) -> Ticker:
    return Ticker(
        symbol="BTC/USDT", price=price,
        bid=price - half_spread, ask=price + half_spread,
    )


def market_order(side: Side = Side.BUY, quantity: float = 1.0) -> Order:
    return Order(
        bot_id="jotaro", symbol="BTC/USDT", side=side,
        type=OrderType.MARKET, quantity=quantity,
    )


# ------------------------------------------------------------------- fill model


async def test_market_buy_pays_the_ask_plus_slippage(settings):
    engine = PaperExecution(settings, EventBus())
    t = ticker(100.0)
    fill = await engine.place_order(market_order(Side.BUY), t)
    expected = t.ask * (1 + settings.trading.slippage_bps / 10_000)
    assert fill is not None
    assert fill.price == pytest.approx(expected)
    assert fill.price > t.price, "a buy must never fill better than the mid"


async def test_market_sell_hits_the_bid_minus_slippage(settings):
    engine = PaperExecution(settings, EventBus())
    t = ticker(100.0)
    fill = await engine.place_order(market_order(Side.SELL), t)
    expected = t.bid * (1 - settings.trading.slippage_bps / 10_000)
    assert fill is not None
    assert fill.price == pytest.approx(expected)
    assert fill.price < t.price


async def test_fee_is_charged_on_notional(settings):
    engine = PaperExecution(settings, EventBus())
    order = market_order(Side.BUY, quantity=2.0)
    fill = await engine.place_order(order, ticker(100.0))
    assert fill is not None
    assert fill.fee == pytest.approx(fill.price * 2.0 * settings.trading.fee_rate)
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(2.0)


async def test_zero_quantity_order_is_rejected(settings):
    engine = PaperExecution(settings, EventBus())
    order = market_order(quantity=0.0)
    assert await engine.place_order(order, ticker()) is None
    assert order.status is OrderStatus.REJECTED


# ------------------------------------------------------------- limit behaviour


async def test_unmarketable_limit_rests_until_price_arrives(settings):
    engine = PaperExecution(settings, EventBus())
    order = Order(
        bot_id="kira", symbol="BTC/USDT", side=Side.BUY,
        type=OrderType.LIMIT, quantity=1.0, price=95.0,
    )
    assert await engine.place_order(order, ticker(100.0)) is None
    assert order.id in {o.id for o in engine.resting_orders}

    assert await engine.on_price(ticker(97.0)) == []      # still above the limit

    fills = await engine.on_price(ticker(94.5))
    assert len(fills) == 1
    assert fills[0].price <= 95.0, "a limit buy must never fill above its price"
    assert order.status is OrderStatus.FILLED
    assert engine.resting_orders == []


async def test_marketable_limit_fills_immediately(settings):
    engine = PaperExecution(settings, EventBus())
    order = Order(
        bot_id="kira", symbol="BTC/USDT", side=Side.BUY,
        type=OrderType.LIMIT, quantity=1.0, price=105.0,
    )
    fill = await engine.place_order(order, ticker(100.0))
    assert fill is not None and fill.price <= 105.0


async def test_resting_order_can_be_cancelled(settings):
    engine = PaperExecution(settings, EventBus())
    order = Order(
        bot_id="kira", symbol="BTC/USDT", side=Side.BUY,
        type=OrderType.LIMIT, quantity=1.0, price=1.0,
    )
    await engine.place_order(order, ticker(100.0))
    assert await engine.cancel_order(order) is True
    assert order.status is OrderStatus.CANCELLED
    assert await engine.cancel_order(order) is False, "cancelling twice is a no-op"


async def test_resting_orders_ignore_other_symbols(settings):
    engine = PaperExecution(settings, EventBus())
    order = Order(
        bot_id="kira", symbol="BTC/USDT", side=Side.BUY,
        type=OrderType.LIMIT, quantity=1.0, price=95.0,
    )
    await engine.place_order(order, ticker(100.0))
    other = Ticker(symbol="ETH/USDT", price=1.0, bid=0.9, ask=1.1)
    assert await engine.on_price(other) == []


# ------------------------------------------------------------------- portfolio


async def test_round_trip_accounting_is_consistent(settings):
    engine = PaperExecution(settings, EventBus())
    portfolio = Portfolio(10_000.0)

    entry = await engine.place_order(market_order(Side.BUY, 1.0), ticker(100.0))
    assert entry is not None
    position = portfolio.open_position(
        bot_id="jotaro", symbol="BTC/USDT", side=PositionSide.LONG,
        fill=entry, leverage=2.0,
    )
    assert position.margin == pytest.approx(entry.price * 1.0 / 2.0)
    assert portfolio.cash == pytest.approx(10_000.0 - position.margin - entry.fee)

    portfolio.mark("BTC/USDT", 110.0)
    assert position.unrealized_pnl == pytest.approx((110.0 - entry.price) * 1.0)

    exit_fill = await engine.place_order(
        market_order(Side.SELL, 1.0), ticker(110.0)
    )
    assert exit_fill is not None
    trade = portfolio.close_position(position, exit_fill, CloseReason.TAKE_PROFIT)

    assert trade.is_win
    assert trade.reason is CloseReason.TAKE_PROFIT
    assert portfolio.positions == {}
    assert portfolio.reserved_margin == pytest.approx(0.0)
    # Cash back = starting - entry fee + net proceeds of the round trip.
    expected_cash = 10_000.0 - entry.fee + (exit_fill.price - entry.price) - exit_fill.fee
    assert portfolio.cash == pytest.approx(expected_cash)


async def test_short_position_profits_when_price_falls(settings):
    engine = PaperExecution(settings, EventBus())
    portfolio = Portfolio(10_000.0)
    entry = await engine.place_order(market_order(Side.SELL, 1.0), ticker(100.0))
    position = portfolio.open_position(
        bot_id="kira", symbol="BTC/USDT", side=PositionSide.SHORT,
        fill=entry, leverage=1.0,
    )
    portfolio.mark("BTC/USDT", 90.0)
    assert position.unrealized_pnl > 0

    exit_fill = await engine.place_order(market_order(Side.BUY, 1.0), ticker(90.0))
    trade = portfolio.close_position(position, exit_fill, CloseReason.SIGNAL)
    assert trade.realized_pnl > 0
    assert trade.side is PositionSide.SHORT


async def test_win_rate_tracks_closed_trades(settings):
    engine = PaperExecution(settings, EventBus())
    portfolio = Portfolio(10_000.0)
    for exit_price in (110.0, 95.0, 120.0):
        entry = await engine.place_order(market_order(Side.BUY, 1.0), ticker(100.0))
        position = portfolio.open_position(
            bot_id="jotaro", symbol="BTC/USDT", side=PositionSide.LONG,
            fill=entry, leverage=1.0,
        )
        exit_fill = await engine.place_order(
            market_order(Side.SELL, 1.0), ticker(exit_price)
        )
        portfolio.close_position(position, exit_fill, CloseReason.SIGNAL)

    total, won, rate = portfolio.bot_win_rate("jotaro")
    assert (total, won) == (3, 2)
    assert rate == pytest.approx(200 / 3)


async def test_drawdown_is_measured_from_the_peak(settings):
    portfolio = Portfolio(10_000.0)
    engine = PaperExecution(settings, EventBus())
    entry = await engine.place_order(market_order(Side.BUY, 10.0), ticker(100.0))
    portfolio.open_position(
        bot_id="jotaro", symbol="BTC/USDT", side=PositionSide.LONG,
        fill=entry, leverage=1.0,
    )
    portfolio.mark("BTC/USDT", 120.0)
    peak = portfolio.peak_equity
    assert portfolio.drawdown_pct == pytest.approx(0.0)

    portfolio.mark("BTC/USDT", 100.0)
    assert portfolio.peak_equity == pytest.approx(peak), "peak must not retreat"
    assert portfolio.drawdown_pct > 0


# ----------------------------------------------------------- live trading gate


def test_live_execution_refuses_without_the_flag(settings):
    settings.trading.live_enabled = False
    settings.exchange.api_key = "key"
    settings.exchange.api_secret = "secret"
    with pytest.raises(LiveTradingRefused, match="live_enabled"):
        LiveExecution(settings, EventBus())


def test_live_execution_refuses_without_credentials(settings, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_THE_RISK")
    settings.trading.live_enabled = True
    settings.exchange.api_key = ""
    settings.exchange.api_secret = ""
    with pytest.raises(LiveTradingRefused, match="API key"):
        LiveExecution(settings, EventBus())


def test_live_execution_refuses_without_the_env_confirmation(settings, monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    settings.trading.live_enabled = True
    settings.exchange.api_key = "key"
    settings.exchange.api_secret = "secret"
    with pytest.raises(LiveTradingRefused, match="LIVE_TRADING_CONFIRM"):
        LiveExecution(settings, EventBus())


def test_wrong_env_confirmation_value_is_refused(settings, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "yes")
    settings.trading.live_enabled = True
    settings.exchange.api_key = "key"
    settings.exchange.api_secret = "secret"
    with pytest.raises(LiveTradingRefused):
        LiveExecution(settings, EventBus())


def test_all_three_conditions_open_the_gate(settings, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "I_UNDERSTAND_THE_RISK")
    settings.trading.live_enabled = True
    settings.exchange.api_key = "key"
    settings.exchange.api_secret = "secret"
    allowed, reason = settings.live_gate_status()
    assert allowed, reason
    adapter = LiveExecution(settings, EventBus())
    assert adapter.mode is ExecutionMode.LIVE and adapter.is_live


def test_factory_falls_back_to_paper_when_the_gate_is_shut(settings):
    settings.trading.live_enabled = False
    adapter, reason = create_execution(settings, EventBus())
    assert isinstance(adapter, PaperExecution)
    assert adapter.mode is ExecutionMode.PAPER
    assert "live_enabled" in reason


def test_default_configuration_is_never_live(settings):
    """The shipped config must not place real orders under any circumstance."""
    allowed, _ = settings.live_gate_status()
    assert not allowed
    adapter, _ = create_execution(settings, EventBus())
    assert not adapter.is_live
