"""Deterministic bar-by-bar backtester.

The engine deliberately does **not** reimplement trading mechanics. It drives
the production classes:

* ``PaperExecution.fill_price`` / ``fee_for`` — the same spread-crossing,
  slippage and fee math the paper desk uses, so a backtest number and a paper
  number differ because of the data, not because of two different fill models.
* ``Portfolio`` — the same equity, margin, realised-P&L and drawdown accounting.
* ``RiskEngine`` — the same sizing and the same caps. This is the important one:
  the risk engine sits *above* the policy here exactly as it does in the live
  orchestrator, so a policy cannot size its own position or trade past an
  exposure limit even in simulation. A learned policy that discovers a way to
  make money by taking 40x exposure will find the backtest refusing it, which
  is the only honest way to evaluate something that will later be gated the
  same way.

**The bar protocol**, which is where backtests usually lie:

1. The decision made on bar ``t-1`` executes at bar ``t``'s **open**. Never at
   the close the decision was made on — that is the classic free lunch.
2. Stops and targets are then checked against bar ``t``'s high/low. **The stop
   wins ties**: if a bar's range contains both, we assume the loss. No intra-bar
   path is reconstructed, because the data does not contain one and assuming a
   favourable path is how a backtest manufactures its edge.
3. A gap through the level fills at the open, not the level. You do not get your
   stop price when the market jumps past it.
4. Equity is marked at bar ``t``'s close, and only then is the policy asked for
   its next decision — from bars ``0..t``, which is all ``PolicyContext`` can
   reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from backend.config import Settings, load_settings
from backend.execution.base import Fill
from backend.execution.paper import PaperExecution
from backend.models import BotConfig, CloseReason, PositionSide, Side, Ticker
from backend.portfolio import Portfolio
from backend.risk import RiskEngine

from .datasets import Dataset
from .metrics import ExitReason, MetricSet, TradeRecord, compute_metrics
from .policy import AccountState, BasePolicy, PolicyAction, PolicyContext
from .splits import SplitPlan, Window

logger = logging.getLogger("research.backtest")

BACKTEST_VERSION = "v1"


@dataclass(frozen=True)
class BacktestConfig:
    """Account and market-microstructure assumptions for one run.

    Defaults are intentionally conservative rather than flattering:
    ``spread_bps`` and ``slippage_bps`` both cost you, and ``risk_per_trade`` is
    2% rather than the 10% in ``config.json`` — which the earlier forensics
    showed to be several times over-sized for the only strategy with positive
    expectancy.
    """

    starting_equity: float = 10_000.0
    risk_per_trade: float = 0.02
    leverage: float = 1.0
    #: ``None`` means no protective exit at all. A hold-forever baseline needs
    #: this: run buy-and-hold with the strategies' 1% stop and it is stopped out
    #: and re-entered a hundred times, which measures something nobody meant by
    #: "buy and hold" and makes the baseline easier to beat than it should be.
    stop_loss_pct: float | None = 1.0
    take_profit_pct: float | None = 7.0
    #: Stop distance used for risk sizing, when there is no protective stop to
    #: derive it from. Position size still has to come from somewhere, and the
    #: risk engine is not bypassed just because the policy holds forever.
    sizing_stop_pct: float = 5.0
    spread_bps: float = 2.0
    slippage_bps: float | None = None      # None => take the value from Settings
    fee_rate: float | None = None
    allow_short: bool = True
    max_exposure_pct: float = 300.0
    drawdown_limit_pct: float = 25.0
    close_at_end: bool = True

    def validate(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive or None")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive or None")
        if self.sizing_stop_pct <= 0:
            raise ValueError("sizing_stop_pct must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")


@dataclass
class BacktestResult:
    policy: str
    dataset_id: str
    dataset_source: str
    symbol: str
    timeframe: str
    start_index: int
    stop_index: int
    ts: np.ndarray
    equity: np.ndarray
    trades: list[TradeRecord]
    metrics: MetricSet
    actions: np.ndarray
    blocked_by_risk: int = 0
    block_reasons: dict[str, int] = field(default_factory=dict)
    slippage_cost: float = 0.0
    seed: int | None = None
    window: int | None = None
    version: str = BACKTEST_VERSION

    @property
    def bars(self) -> int:
        return int(self.equity.size)

    def describe(self) -> str:
        tag = f"w{self.window}" if self.window is not None else "full"
        return f"{self.policy:<20} {tag:<5} {self.metrics.describe()}"


def _ticker(symbol: str, price: float, spread_bps: float) -> Ticker:
    """A synthetic top-of-book around ``price``.

    OHLCV carries no bid/ask, so one has to be assumed to use the real fill
    model. Assuming a spread costs the backtest money on every trade, which is
    the correct direction to be wrong in.
    """
    half = price * (spread_bps / 20_000.0)
    return Ticker(symbol=symbol, price=price, bid=price - half, ask=price + half,
                  volume_24h=0.0, change_24h_pct=0.0, ts=0)


def _settings_for(config: BacktestConfig, base: Settings | None = None) -> Settings:
    settings = (base or load_settings()).model_copy(deep=True)
    if config.slippage_bps is not None:
        settings.trading.slippage_bps = config.slippage_bps
    if config.fee_rate is not None:
        settings.trading.fee_rate = config.fee_rate
    settings.trading.live_enabled = False          # never, from research code
    settings.risk.max_risk_per_trade = max(settings.risk.max_risk_per_trade,
                                           config.risk_per_trade)
    settings.risk.max_portfolio_positions = 1      # one policy, one position
    settings.risk.max_exposure_pct = config.max_exposure_pct
    settings.risk.drawdown_limit_pct = config.drawdown_limit_pct
    return settings


def run_backtest(
    policy: BasePolicy,
    dataset: Dataset,
    *,
    start: int = 0,
    stop: int | None = None,
    config: BacktestConfig | None = None,
    settings: Settings | None = None,
    seed: int | None = None,
    window: int | None = None,
) -> BacktestResult:
    """Run ``policy`` over ``dataset[start:stop]`` and return equity, trades, metrics."""
    cfg = config or BacktestConfig()
    cfg.validate()
    stop = len(dataset) if stop is None else min(stop, len(dataset))
    if stop - start < 2:
        raise ValueError(f"need at least 2 bars, got {stop - start}")

    resolved = _settings_for(cfg, settings)
    portfolio = Portfolio(cfg.starting_equity)
    risk = RiskEngine(resolved, portfolio)
    execution = PaperExecution(resolved, bus=None)

    bot = BotConfig(
        id="research", name="research", strategy=policy.name,
        symbol=dataset.manifest.symbol, timeframe=dataset.manifest.timeframe,
        allocation=1.0, leverage=cfg.leverage, max_positions=1,
        risk_per_trade=cfg.risk_per_trade,
        stop_loss_pct=cfg.stop_loss_pct if cfg.stop_loss_pct is not None else 0.0,
        take_profit_pct=cfg.take_profit_pct if cfg.take_profit_pct is not None else 0.0,
    )
    symbol = dataset.manifest.symbol

    policy.reset(seed)

    bars = stop - start
    equity_curve = np.empty(bars, dtype=np.float64)
    actions = np.zeros(bars, dtype=np.int8)

    trades: list[TradeRecord] = []
    block_reasons: dict[str, int] = {}
    slippage_cost = 0.0
    exposure_bars = 0

    pending: PolicyAction | None = None
    entry_index: int | None = None
    bars_in_position = 0
    bars_since_trade = 0
    trades_taken = 0

    for offset in range(bars):
        i = start + offset
        position = next(iter(portfolio.positions.values()), None)

        # --- 1. execute the decision made on the previous bar, at this open ---
        if pending is not None:
            position, entry_index, opened, slip = _apply(
                pending, dataset, i, portfolio, risk, execution, bot, cfg, symbol,
                position, entry_index, trades, block_reasons,
            )
            slippage_cost += slip
            if opened:
                trades_taken += 1
                bars_in_position = 0
                bars_since_trade = 0
            pending = None

        # --- 2. stops and targets against this bar's range; the stop wins ties ---
        if position is not None:
            exit_ref, reason = _triggered_exit(position, dataset, i)
            if exit_ref is not None:
                side = Side.SELL if position.side is PositionSide.LONG else Side.BUY
                price = execution.fill_price(side, _ticker(symbol, exit_ref, cfg.spread_bps))
                slippage_cost += abs(price - exit_ref) * position.quantity
                fee = execution.fee_for(price, position.quantity)
                entry_fee = position.fees_paid
                trade = portfolio.close_position(
                    position, Fill(order_id="bt", price=price, quantity=position.quantity, fee=fee),
                    reason=reason,
                )
                trades.append(_record(trade, entry_index or i, i, reason, entry_fee))
                position, entry_index = None, None
                bars_since_trade = 0

        # --- 3. mark to this bar's close ---
        close = float(dataset.close[i])
        portfolio.mark(symbol, close)
        equity_curve[offset] = portfolio.equity
        if portfolio.positions:
            exposure_bars += 1
            bars_in_position += 1
        else:
            bars_in_position = 0
        bars_since_trade += 1

        # --- 4. ask for the next decision, from bars 0..i only ---
        position = next(iter(portfolio.positions.values()), None)
        account = _account_state(
            portfolio, position, close, cfg, bars_in_position, bars_since_trade, trades_taken
        )
        decision = policy.decide(
            PolicyContext(dataset=dataset, index=i, account=account, bot_id=bot.id)
        )
        actions[offset] = int(decision.action)
        pending = decision.action if offset < bars - 1 else None

    # --- close anything still open, so the equity curve ends realised ---
    position = next(iter(portfolio.positions.values()), None)
    if position is not None and cfg.close_at_end:
        last = start + bars - 1
        reference = float(dataset.close[last])
        side = Side.SELL if position.side is PositionSide.LONG else Side.BUY
        price = execution.fill_price(side, _ticker(symbol, reference, cfg.spread_bps))
        slippage_cost += abs(price - reference) * position.quantity
        fee = execution.fee_for(price, position.quantity)
        entry_fee = position.fees_paid
        trade = portfolio.close_position(
            position, Fill(order_id="bt", price=price, quantity=position.quantity, fee=fee),
            reason=CloseReason.MANUAL,
        )
        trades.append(_record(trade, entry_index or last, last, CloseReason.MANUAL,
                              entry_fee, override=ExitReason.END_OF_DATA))
        equity_curve[-1] = portfolio.equity

    ts = dataset.ts[start:stop]
    metrics = compute_metrics(
        equity_curve, ts, trades,
        bars_per_year=dataset.bars_per_day * 365.0,
        exposure_bars=exposure_bars,
        slippage_cost=slippage_cost,
    )
    return BacktestResult(
        policy=policy.name,
        dataset_id=dataset.manifest.dataset_id,
        dataset_source=dataset.manifest.source,
        symbol=symbol,
        timeframe=dataset.manifest.timeframe,
        start_index=start, stop_index=stop,
        ts=ts, equity=equity_curve, trades=trades, metrics=metrics, actions=actions,
        blocked_by_risk=sum(block_reasons.values()), block_reasons=block_reasons,
        slippage_cost=slippage_cost, seed=seed, window=window,
    )


# --------------------------------------------------------------------------
# Bar mechanics
# --------------------------------------------------------------------------


def _apply(
    action, dataset, i, portfolio, risk, execution, bot, cfg, symbol,
    position, entry_index, trades, block_reasons,
):
    """Execute one queued action at bar ``i``'s open.

    Returns ``(position, entry_index, opened, slippage_cost)``. Bar counters are
    the caller's business — an earlier version returned them from here and reset
    them on every bar, including the ones where the queued action was HOLD and
    nothing happened at all.
    """
    reference = float(dataset.open[i])
    slippage = 0.0

    wants_long = action is PolicyAction.LONG
    wants_short = action is PolicyAction.SHORT and cfg.allow_short
    wants_close = action is PolicyAction.CLOSE

    # A reversal is a close followed by an open, never a silent flip.
    if position is not None:
        opposing = (
            (position.side is PositionSide.LONG and wants_short)
            or (position.side is PositionSide.SHORT and wants_long)
        )
        if wants_close or opposing:
            side = Side.SELL if position.side is PositionSide.LONG else Side.BUY
            price = execution.fill_price(side, _ticker(symbol, reference, cfg.spread_bps))
            slippage += abs(price - reference) * position.quantity
            fee = execution.fee_for(price, position.quantity)
            entry_fee = position.fees_paid
            trade = portfolio.close_position(
                position, Fill(order_id="bt", price=price, quantity=position.quantity, fee=fee),
                reason=CloseReason.SIGNAL,
            )
            trades.append(_record(trade, entry_index if entry_index is not None else i, i,
                                  CloseReason.SIGNAL, entry_fee))
            position, entry_index = None, None
        else:
            return position, entry_index, False, slippage

    if not (wants_long or wants_short) or position is not None:
        return position, entry_index, False, slippage

    # --- sizing and gating go through the real risk engine ---
    stop_for_sizing = cfg.stop_loss_pct if cfg.stop_loss_pct is not None else cfg.sizing_stop_pct
    sizing = risk.size_position(bot, reference, stop_for_sizing)
    if not sizing.allowed:
        block_reasons[sizing.reason] = block_reasons.get(sizing.reason, 0) + 1
        return None, None, False, slippage

    gate = risk.can_open(bot, sizing.quantity * reference)
    if not gate.allowed:
        block_reasons[gate.reason] = block_reasons.get(gate.reason, 0) + 1
        return None, None, False, slippage

    side = Side.BUY if wants_long else Side.SELL
    price = execution.fill_price(side, _ticker(symbol, reference, cfg.spread_bps))
    slippage += abs(price - reference) * sizing.quantity
    fee = execution.fee_for(price, sizing.quantity)
    position_side = PositionSide.LONG if wants_long else PositionSide.SHORT
    direction = 1.0 if wants_long else -1.0
    opened = portfolio.open_position(
        bot_id=bot.id, symbol=symbol, side=position_side,
        fill=Fill(order_id="bt", price=price, quantity=sizing.quantity, fee=fee),
        leverage=cfg.leverage,
        stop_loss=(None if cfg.stop_loss_pct is None
                   else price * (1 - direction * cfg.stop_loss_pct / 100.0)),
        take_profit=(None if cfg.take_profit_pct is None
                     else price * (1 + direction * cfg.take_profit_pct / 100.0)),
    )
    return opened, i, True, slippage


def _triggered_exit(position, dataset, i):
    """Which protective level fired on bar ``i``, and at what reference price.

    The stop is tested first and returns first, so a bar whose range spans both
    levels is booked as a loss. Without that rule every wide bar becomes a
    winner and the whole backtest inherits an edge the market never offered.
    """
    high = float(dataset.high[i])
    low = float(dataset.low[i])
    open_ = float(dataset.open[i])

    if position.side is PositionSide.LONG:
        if position.stop_loss is not None and low <= position.stop_loss:
            # A gap through the stop fills at the open, which is worse.
            return min(open_, position.stop_loss), CloseReason.STOP_LOSS
        if position.take_profit is not None and high >= position.take_profit:
            return max(open_, position.take_profit), CloseReason.TAKE_PROFIT
    else:
        if position.stop_loss is not None and high >= position.stop_loss:
            return max(open_, position.stop_loss), CloseReason.STOP_LOSS
        if position.take_profit is not None and low <= position.take_profit:
            return min(open_, position.take_profit), CloseReason.TAKE_PROFIT
    return None, CloseReason.SIGNAL


_REASON_MAP = {
    CloseReason.STOP_LOSS: ExitReason.STOP_LOSS,
    CloseReason.TAKE_PROFIT: ExitReason.TAKE_PROFIT,
    CloseReason.SIGNAL: ExitReason.SIGNAL,
    CloseReason.MANUAL: ExitReason.END_OF_DATA,
    CloseReason.LIQUIDATION: ExitReason.RISK,
    CloseReason.EMERGENCY_STOP: ExitReason.RISK,
}


def _record(
    trade, entry_index: int, exit_index: int, reason, entry_fee: float, override=None
) -> TradeRecord:
    """Convert a portfolio ``Trade`` into a metrics ``TradeRecord``.

    ``Trade.realized_pnl`` nets off only the *exit* fee — the entry fee was
    already taken out of cash when the position opened, so it never appears in
    the trade's own P&L. Metrics treat ``TradeRecord.pnl`` as fully net, so the
    entry fee is subtracted here. Skipping this step leaves every expectancy,
    profit factor and win rate overstated by exactly the entry cost, and the sum
    of trade P&L no longer reconciles with the equity curve.
    """
    return TradeRecord(
        entry_index=entry_index,
        exit_index=exit_index,
        side=trade.side.value,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        quantity=trade.quantity,
        pnl=trade.realized_pnl - entry_fee,
        fees=trade.fees,
        reason=(override or _REASON_MAP.get(reason, ExitReason.SIGNAL)).value,
    )


def _account_state(
    portfolio, position, mark: float, cfg: BacktestConfig,
    bars_in_position: int, bars_since_trade: int, trades_taken: int,
) -> AccountState:
    equity = portfolio.equity
    if position is None:
        return AccountState(
            equity=equity, starting_equity=cfg.starting_equity,
            drawdown_pct=portfolio.drawdown_pct,
            bars_since_trade=bars_since_trade, trades_taken=trades_taken,
            realised_pnl_pct=(equity / cfg.starting_equity - 1.0) * 100.0,
        )
    _, pnl_pct = position.compute_pnl(mark)
    return AccountState(
        equity=equity, starting_equity=cfg.starting_equity,
        position_side=1 if position.side is PositionSide.LONG else -1,
        position_qty=position.quantity,
        entry_price=position.entry_price,
        mark_price=mark,
        unrealised_pnl_pct=pnl_pct,
        drawdown_pct=portfolio.drawdown_pct,
        exposure_pct=(portfolio.total_exposure / equity * 100.0) if equity > 0 else 0.0,
        bars_in_position=bars_in_position,
        bars_since_trade=bars_since_trade,
        trades_taken=trades_taken,
        realised_pnl_pct=(equity / cfg.starting_equity - 1.0) * 100.0,
    )


# --------------------------------------------------------------------------
# Walk-forward driver
# --------------------------------------------------------------------------


def run_walk_forward(
    policy_factory: Callable[[], BasePolicy],
    dataset: Dataset,
    plan: SplitPlan,
    *,
    config: BacktestConfig | None = None,
    settings: Settings | None = None,
    seeds: Sequence[int] = (0,),
    windows: Sequence[Window] | None = None,
) -> list[BacktestResult]:
    """Evaluate a policy on every window's **validation** range.

    Training ranges are not touched here — a policy with no learning has nothing
    to do with them, and a learned one is fitted elsewhere and handed in already
    trained. What this guarantees is that the reported score always comes from
    bars the policy was not fitted on.
    """
    plan.bind(dataset)
    results: list[BacktestResult] = []
    for window in (windows if windows is not None else plan.windows):
        for seed in seeds:
            policy = policy_factory()
            results.append(
                run_backtest(
                    policy, dataset,
                    start=window.validate.start, stop=window.validate.stop,
                    config=config, settings=settings, seed=seed, window=window.index,
                )
            )
    return results
