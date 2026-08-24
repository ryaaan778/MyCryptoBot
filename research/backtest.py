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


@dataclass
class StepOutcome:
    """What one bar did. Returned by :meth:`Simulator.step`."""

    index: int
    equity: float
    equity_before: float
    account: AccountState
    opened: bool = False
    closed: TradeRecord | None = None
    blocked: str | None = None
    drawdown_pct: float = 0.0
    drawdown_before: float = 0.0

    @property
    def log_return(self) -> float:
        if self.equity_before <= 1e-12 or self.equity <= 1e-12:
            return 0.0
        return float(np.log(self.equity / self.equity_before))

    @property
    def position_changed(self) -> bool:
        return self.opened or self.closed is not None


class Simulator:
    """The bar protocol, owned in one place.

    Both :func:`run_backtest` and the reinforcement-learning environment drive
    this object, and that is deliberate rather than tidy: if the environment had
    its own copy of the fill and exit logic, a policy would be trained against
    one set of mechanics and evaluated against another, and the gap between the
    two would look exactly like an edge. Keeping a single implementation means a
    learned policy and a hand-written strategy are measured by the same rules.

    Usage is a loop:

        sim = Simulator(dataset, config)
        sim.reset()
        while not sim.done:
            action = decide(sim.account)
            sim.step(action)
        sim.finish()
    """

    def __init__(
        self,
        dataset: Dataset,
        config: BacktestConfig | None = None,
        *,
        settings: Settings | None = None,
        start: int = 0,
        stop: int | None = None,
        label: str = "policy",
    ) -> None:
        self.dataset = dataset
        self.config = config or BacktestConfig()
        self.config.validate()
        self.start = max(0, start)
        self.stop = len(dataset) if stop is None else min(stop, len(dataset))
        if self.stop - self.start < 2:
            raise ValueError(f"need at least 2 bars, got {self.stop - self.start}")
        self.label = label
        self.symbol = dataset.manifest.symbol
        self._settings = _settings_for(self.config, settings)
        self.reset()

    # ---- lifecycle ---------------------------------------------------------

    def reset(self) -> AccountState:
        cfg = self.config
        self.portfolio = Portfolio(cfg.starting_equity)
        self.risk = RiskEngine(self._settings, self.portfolio)
        self.execution = PaperExecution(self._settings, bus=None)
        self.bot = BotConfig(
            id="research", name="research", strategy=self.label,
            symbol=self.symbol, timeframe=self.dataset.manifest.timeframe,
            allocation=1.0, leverage=cfg.leverage, max_positions=1,
            risk_per_trade=cfg.risk_per_trade,
            stop_loss_pct=cfg.stop_loss_pct if cfg.stop_loss_pct is not None else 0.0,
            take_profit_pct=cfg.take_profit_pct if cfg.take_profit_pct is not None else 0.0,
        )

        self.bars = self.stop - self.start
        self.equity_curve = np.empty(self.bars, dtype=np.float64)
        self.actions = np.zeros(self.bars, dtype=np.int8)
        self.trades: list[TradeRecord] = []
        self.block_reasons: dict[str, int] = {}
        self.slippage_cost = 0.0
        self.exposure_bars = 0

        self.index = self.start
        self._entry_index: int | None = None
        self._bars_in_position = 0
        self._bars_since_trade = 0
        self._trades_taken = 0
        self._finished = False
        self._drawdown = 0.0

        # Bar `start` has no prior decision to execute and no position to stop
        # out, so it is only marked. Everything after it goes through `step`.
        self._mark()
        self.account = self._account_state()
        return self.account

    @property
    def offset(self) -> int:
        return self.index - self.start

    @property
    def done(self) -> bool:
        """True on the final bar: its decision would have nowhere to execute."""
        return self.index >= self.stop - 1

    @property
    def position(self):
        return next(iter(self.portfolio.positions.values()), None)

    def record_action(self, action: PolicyAction) -> None:
        self.actions[self.offset] = int(action)

    # ---- one bar -----------------------------------------------------------

    def step(self, action: PolicyAction) -> StepOutcome:
        """Advance one bar, executing ``action`` at the next bar's open."""
        if self.done:
            raise RuntimeError("simulator is at the final bar; call finish()")

        equity_before = float(self.portfolio.equity)
        drawdown_before = self._drawdown
        self.index += 1
        i = self.index
        cfg = self.config

        # 1. the decision made on the previous bar fills at this bar's open
        position, self._entry_index, opened, slip = _apply(
            action, self.dataset, i, self.portfolio, self.risk, self.execution,
            self.bot, cfg, self.symbol, self.position, self._entry_index,
            self.trades, self.block_reasons,
        )
        self.slippage_cost += slip
        blocked = None
        if not opened and action in (PolicyAction.LONG, PolicyAction.SHORT) and position is None:
            blocked = next(reversed(self.block_reasons), None) if self.block_reasons else None
        if opened:
            self._trades_taken += 1
            self._bars_in_position = 0
            self._bars_since_trade = 0

        # 2. stops and targets against this bar's range; the stop wins ties
        closed = self._check_protective_exit()

        # 3. mark to this bar's close
        self._mark()
        self.account = self._account_state()

        return StepOutcome(
            index=i, equity=float(self.portfolio.equity), equity_before=equity_before,
            account=self.account, opened=opened, closed=closed, blocked=blocked,
            drawdown_pct=self._drawdown, drawdown_before=drawdown_before,
        )

    def finish(self) -> None:
        """Close anything still open so the equity curve ends realised."""
        if self._finished:
            return
        self._finished = True
        position = self.position
        if position is None or not self.config.close_at_end:
            return

        last = self.stop - 1
        reference = float(self.dataset.close[last])
        side = Side.SELL if position.side is PositionSide.LONG else Side.BUY
        price = self.execution.fill_price(
            side, _ticker(self.symbol, reference, self.config.spread_bps))
        self.slippage_cost += abs(price - reference) * position.quantity
        fee = self.execution.fee_for(price, position.quantity)
        entry_fee = position.fees_paid
        trade = self.portfolio.close_position(
            position,
            Fill(order_id="bt", price=price, quantity=position.quantity, fee=fee),
            reason=CloseReason.MANUAL,
        )
        self.trades.append(_record(trade, self._entry_index or last, last,
                                   CloseReason.MANUAL, entry_fee,
                                   override=ExitReason.END_OF_DATA))
        self.equity_curve[-1] = self.portfolio.equity

    # ---- internals ---------------------------------------------------------

    def _check_protective_exit(self) -> TradeRecord | None:
        position = self.position
        if position is None:
            return None
        exit_ref, reason = _triggered_exit(position, self.dataset, self.index)
        if exit_ref is None:
            return None

        side = Side.SELL if position.side is PositionSide.LONG else Side.BUY
        price = self.execution.fill_price(
            side, _ticker(self.symbol, exit_ref, self.config.spread_bps))
        self.slippage_cost += abs(price - exit_ref) * position.quantity
        fee = self.execution.fee_for(price, position.quantity)
        entry_fee = position.fees_paid
        trade = self.portfolio.close_position(
            position, Fill(order_id="bt", price=price, quantity=position.quantity, fee=fee),
            reason=reason,
        )
        record = _record(trade, self._entry_index or self.index, self.index, reason, entry_fee)
        self.trades.append(record)
        self._entry_index = None
        self._bars_since_trade = 0
        return record

    def _mark(self) -> None:
        close = float(self.dataset.close[self.index])
        self.portfolio.mark(self.symbol, close)
        self.equity_curve[self.offset] = self.portfolio.equity
        if self.portfolio.positions:
            self.exposure_bars += 1
            self._bars_in_position += 1
        else:
            self._bars_in_position = 0
        self._bars_since_trade += 1
        self._drawdown = float(self.portfolio.drawdown_pct)

    def _account_state(self) -> AccountState:
        return _account_state(
            self.portfolio, self.position, float(self.dataset.close[self.index]),
            self.config, self._bars_in_position, self._bars_since_trade, self._trades_taken,
        )

    def context(self, features: np.ndarray | None = None) -> PolicyContext:
        return PolicyContext(dataset=self.dataset, index=self.index,
                             account=self.account, bot_id="research", features=features)

    # ---- output ------------------------------------------------------------

    def result(self, *, seed: int | None = None, window: int | None = None) -> BacktestResult:
        ts = self.dataset.ts[self.start:self.stop]
        metrics = compute_metrics(
            self.equity_curve, ts, self.trades,
            bars_per_year=self.dataset.bars_per_day * 365.0,
            exposure_bars=self.exposure_bars,
            slippage_cost=self.slippage_cost,
        )
        return BacktestResult(
            policy=self.label,
            dataset_id=self.dataset.manifest.dataset_id,
            dataset_source=self.dataset.manifest.source,
            symbol=self.symbol,
            timeframe=self.dataset.manifest.timeframe,
            start_index=self.start, stop_index=self.stop,
            ts=ts, equity=self.equity_curve, trades=self.trades, metrics=metrics,
            actions=self.actions,
            blocked_by_risk=sum(self.block_reasons.values()),
            block_reasons=self.block_reasons,
            slippage_cost=self.slippage_cost, seed=seed, window=window,
        )


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
    simulator = Simulator(dataset, config, settings=settings, start=start, stop=stop,
                          label=policy.name)
    policy.reset(seed)

    while True:
        decision = policy.decide(simulator.context())
        simulator.record_action(decision.action)
        if simulator.done:
            break
        simulator.step(decision.action)

    simulator.finish()
    return simulator.result(seed=seed, window=window)


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
