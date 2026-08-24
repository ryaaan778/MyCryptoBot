# Research pipeline — Phases 1–2

Offline infrastructure for the multi-agent RL work: reproducible datasets,
causal features, leak-resistant splits, honest metrics, and a backtester that
drives the production trading classes rather than reimplementing them.

Nothing here places an order. `research/` imports `backend/` and never the
reverse, and the backtester forces `live_enabled=False` regardless of what the
settings say.

## Quick start

```bash
python research.py venues                        # 103 exchanges ccxt can reach
python research.py synth --bars 60000            # deterministic simulated bars
python research.py collect --venue bybit --symbol BTC/USDT --from 2023-01-01
python research.py datasets                      # what's on disk, and where it came from
python research.py features
python research.py regimes
python research.py split --scale 0.25
python research.py backtest --policy mean_reversion
python research.py baselines --scale 0.25 --seeds 3 --record
python research.py report --metric sortino
```

Training, promotion and the RL agents arrive in Phase 3–5; those subcommands are
absent rather than stubbed, so `--help` describes what exists today.

## What each module guarantees

| Module | Guarantee | How it is enforced |
|---|---|---|
| `datasets.py` | Every dataset names its source — `SYNTHETIC` or a ccxt exchange id — and carries a sha256 over its bytes | Unknown venue ids rejected against `ccxt.exchanges`; checksum verified on load |
| `collect.py` | One collector for every venue; paginated and resumable | Page and rate limits read from the exchange object, not hard-coded |
| `features.py` | 24 causal, relative-not-absolute features | Test mutates every future bar and asserts the row is byte-identical |
| `regimes.py` | Labels use only trailing data — *including the thresholds* | Test labels a prefix alone and asserts it matches the full-series labels |
| `splits.py` | Purged walk-forward, embargo, sealed holdout | Coverage, purge gap and seal all asserted; unlock needs a phrase and writes an audit record |
| `metrics.py` | Returns compound; median and IQR across seeds | No best-of-N helper exists in the module |
| `policy.py` | Learned policies and hand-written strategies share one interface | Adapter verified action-and-confidence identical to the live `BotAgent` path |
| `backtest.py` | The risk engine sits above the policy in simulation too | Uses the real `RiskEngine`, `Portfolio` and `PaperExecution` |
| `experiments.py` | No metric can exist without provenance | `policy_metrics.dataset_id` is a foreign key, and foreign keys are switched on |

### The rules that cost something to keep

- **Synthetic never aggregates with real.** `assert_single_source` raises rather
  than returning a blended number, and `allow_cross_venue=True` does not relax
  it. Real venues also refuse to blend with each other by default: fees,
  spreads and leverage caps differ per exchange, so an average across venues
  describes no venue anyone can trade.
- **The holdout is sealed.** `plan.holdout()` raises until someone passes
  `I_ACCEPT_THIS_BURNS_THE_HOLDOUT` with an actor and a reason. Repeat unlocks
  are recorded cumulatively, because a holdout looked at five times is a
  validation set and the record is what keeps that visible.
- **The stop wins ties.** A bar whose range contains both the stop and the
  target is booked as a loss. No intra-bar path is knowable, and assuming a
  favourable one is how a backtest manufactures an edge.
- **A decision on bar `t` fills at bar `t+1`'s open**, and a gap through a level
  fills at the open rather than at the level.

## Phase 1 results

`python research.py baselines --scale 0.25 --seeds 3` on 60,000 synthetic 5m
bars: 11 walk-forward windows × 3 seeds, 9,000 bars held back sealed.

```
policy                median ret%     IQR  sortino  maxDD%   win%   R:R  trades   TP%   SL%
-------------------------------------------------------------------------------------------
always_short                +0.39    3.24     0.68    4.26  100.0100.00      33   0.0   0.0
flat                        +0.00    0.00     0.00    0.00    0.0  0.00       0   0.0   0.0
buy_and_hold                -0.50    3.24    -0.88    4.59    0.0  0.00      33   0.0   0.0
random                      -5.31   10.87    -6.20   10.06   38.5  1.11    1338   0.0  34.2
volatility_breakout         -5.90    4.05    -9.32    7.31   36.1  1.36    1233   0.0   8.8
mean_reversion              -6.72    8.43    -8.01    9.31   53.8  0.61    2670   0.0  26.3
hybrid                     -15.76   24.50   -15.35   17.03   28.7  1.54    3744   0.0   5.8
momentum                   -24.47    3.80   -24.05   25.07   28.7  1.41    5931   0.0   2.5
scalping                   -24.80    1.01   -25.75   25.17   25.3  1.53    6450   0.0   1.5
```

**These are simulator results, not evidence about live markets.** They describe
the shipped strategies and the shipped parameters, and that is all.

### Finding 1 — the configured take-profit does not exist

The 7% target fired on **0.00% of 21,432 trades**. Sweeping it shows why:

```
 target%    ATR  TP hit%  SL hit%  signal%  median ret%   win%   R:R
    0.50    2.1    33.44    24.20    42.24       -11.21   57.0  0.41
    1.00    4.2     9.21    26.07    64.61        -7.91   53.5  0.56
    2.00    8.4     1.12    26.29    72.47        -6.49   53.3  0.59
    3.00   12.6     0.00    26.29    73.60        -6.72   53.3  0.58
    7.00   29.3     0.00    26.29    73.60        -6.72   53.3  0.58
```

Mean ATR(14) on this tape is 0.239% of price, so the configured target sits
**29 ATR away** on a 5-minute chart. Beyond about 10 ATR the rows are byte-identical
— the level is simply not part of the system. The configured 1:7 reward-to-risk
is therefore fictional: realised R:R is 0.6–1.5 because every trade exits on a
stop or a signal instead.

### Finding 2 — no target distance rescues the strategy

The best row in that sweep is still −6.49%. Fixing the geometry is necessary and
nowhere near sufficient.

### Finding 3 — every shipped strategy loses to doing nothing

`flat` returns exactly 0.00% and beats all five. Two of them (`momentum`,
`scalping`) also lose to seeded coin-flipping. Trade counts explain most of it:
scalping takes 6,450 trades where volatility_breakout takes 1,233, and each
round trip costs roughly 14 bps (2×4 bps fee + 2×1 bp half-spread + 2×2 bps
slippage).

Removing costs entirely flips **every** strategy positive:

| strategy | with costs | cost-free | cost per trade |
|---|---:|---:|---:|
| scalping | −80.7 | +35.9 | 0.076 pts |
| momentum | −77.4 | +36.3 | 0.087 pts |
| hybrid | −50.3 | +25.2 | 0.127 pts |
| mean_reversion | −58.4 | −8.9 | 0.113 pts |
| volatility_breakout | −12.9 | +14.2 | 0.137 pts |

(raw per-trade percentage sums, the same quantity the earlier analysis reported)

So the shipped strategies have edges that exist gross and vanish net. That is a
much more specific diagnosis than "they lose money".

### On the Phase 1 success criterion

The plan set this bar: *the harness independently reproduces the ranking already
measured — `mean_reversion` the only positive-expectancy strategy,
`volatility_breakout` the worst, TP-hit rate ≈0. If it can't, the harness is
wrong and we fix it before a line of RL is written.*

**The structural finding reproduces, decisively** — the TP-hit rate is 0.00%
against the earlier 0.1%, measured on 16× more trades.

**The ranking does not reproduce, and the harness is not what changed.** The
earlier analysis measured raw per-trade percentage moves with no fees, no
spread, no slippage and no position sizing, summed rather than compounded, with
unbounded indicator history and entry at the signal bar's close. This harness
measures equity after all of those. They are different quantities, and the cost
A/B above isolates which difference dominates.

The harness is verified against the production code in five independent ways,
each a permanent test:

1. Entry prices are byte-equal to `PaperExecution.fill_price` at every spread.
2. Strategy decisions are action-and-confidence identical to the live
   `BotAgent` path across 860 decisions on five strategies.
3. Final equity minus starting equity equals the sum of trade P&L exactly.
4. Mutating every bar after the evaluation window leaves the equity curve
   byte-identical.
5. Repeated runs are byte-identical; a changed seed diverges.

Two real accounting bugs were found by those checks and fixed: entry fees were
missing from trade P&L (overstating every expectancy, profit factor and win
rate), and the buy-and-hold baseline inherited the strategies' 1% stop, which
churned it through 126 trades and made it far too easy to beat.

### What this means for the RL work

The baselines are not a formality. On this tape `flat` beats every shipped
strategy, so **the bar a learned policy has to clear is "better than not
trading"** — and it has to clear it after costs, which is where all five
currently fail. A policy that trades often will need an edge above ~14 bps per
round trip before it is worth anything at all.

## Phase 2 — the RL environment

`research/env.py` is a Gymnasium environment wrapped around
`research.backtest.Simulator`, which was extracted from `run_backtest` for
exactly this purpose. Both drive the same object, so the mechanics an agent
learns against are byte-for-byte the mechanics its results are measured with.
That refactor was verified by fingerprinting 54 runs (2 datasets × 9 policies ×
3 configs) before and after: identical.

```
observation  24 causal market features + 7 account features = 31, clipped to ±10
action       Discrete(4) — HOLD, LONG, SHORT, CLOSE
fills        PaperExecution (spread, slippage, fees)
sizing       RiskEngine.size_position — the agent picks a direction, never a size
gating       RiskEngine.can_open — a blocked entry appears in info["blocked"]
```

**The risk engine is inside the loop, not around it.** From inside the
environment an exposure or drawdown cap is not a penalty to trade off, it is a
wall: the action simply does not happen. A learned policy therefore inherits the
same authority structure as a hand-written strategy, structurally rather than by
convention. A test drives 600 leveraged entries into a 0.001% exposure cap and
asserts zero trades result.

**The scaler is required, not fitted by the environment.** `TradingEnv` has no
code path that computes normalisation statistics, because the natural place to
do it is over the whole episode range — which leaks the future into every
observation. `make_training_env` fits on training rows only; `make_eval_env`
takes that same scaler. Refitting on the evaluation window would defeat the
walk-forward split entirely.

**`terminated` and `truncated` are kept distinct.** Ruin terminates; running out
of bars truncates. Conflating them teaches a value function that reaching the
end of the data is as bad as going broke.

### Reward v1

```
r_t = Δlog_equity_t
    − 1.0    · max(0, DD_t − DD_{t−1})     # NEW drawdown only
    − 2e-4   · 1[position changed]
    − 0.5    · max(0, −Δlog_equity_t)²
```

Four choices there are deliberate, and each is tested:

- **Costs are not a term.** They are already inside equity via `PaperExecution`.
  Subtracting them again would charge every trade twice and teach the agent that
  trading costs about double what it really does.
- **New drawdown, not absolute.** Penalising the *level* every step charges the
  agent repeatedly for a mistake it can no longer undo, and produces paralysis.
  Recovering from a drawdown scores zero, not a penalty.
- **No inactivity penalty.** Doing nothing scores *exactly* zero — verified over
  a full episode, not approximately. Paying an agent to act teaches churn, and
  churn is precisely how the shipped strategies lost their edge to fees. The
  minimum-trade requirement lives in the validation gate instead.
- **The tail term is convex.** A −1% bar costs 5e-5; a −50% bar costs 0.125.

The units matter more than they look: `Portfolio` reports drawdown in percent,
and feeding that straight in with `λ_dd = 1.0` would make the term a hundred
times too heavy — enough to swamp the return term entirely and teach the agent
never to open a position, while the training curve looked perfectly healthy.
There is a test for the scaling.

One invariant ties the whole thing to reality: **Σ Δlog_equity must equal
log(final / initial equity)**, exactly. If those ever diverge the agent is being
paid for something the equity curve did not do.

The environment passes `gymnasium.utils.env_checker.check_env`.

## Tests

`pytest` — 293 tests, of which 188 are new:

- `test_research_leakage.py` — feature causality, regime threshold causality,
  backtest look-ahead, scaler fitting
- `test_research_splits.py` — purge, embargo, coverage, the holdout seal
- `test_research_provenance.py` — venue validation, source segregation,
  foreign-key enforcement, experiment lifecycle
- `test_research_backtest.py` — bar protocol, stop-wins-ties, gap fills, cost
  accounting, risk authority
- `test_research_determinism.py` — reproducibility and metric definitions
- `test_research_env.py` — reward terms, Gymnasium conformance, env/backtester
  bar-for-bar parity, risk authority inside the environment

The original 105 backend tests are unchanged and green.

## Data layout

```
data/
  datasets/{dataset_id}/manifest.json + candles.npz
  splits/{split_id}.json
  raw/{exchange}/{symbol}/{tf}/partial.npz     # resumable download cache
  research.db                                  # 8 research tables
```

`dataset_id` embeds the source and a checksum prefix, so
`binance_btcusdt_5m_1704067200000_a3f21b9c` and
`synthetic_btcusdt_5m_1704067200000_a3f21b9c` can never be confused for one
another even at a glance.
