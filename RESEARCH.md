# Research pipeline — Phases 1–5

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
python research.py agents                        # the five identities and their drift
python research.py campaign --agent KIRA --experiments 5
python research.py desk --baselines               # JOJO: ranking and allocation
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

## Phase 3 — JOLYAN, trained and rejected

The first learned policy, end to end: PPO on one walk-forward window, exported,
walk-forwarded against all nine baselines, put through the promotion gate.

```bash
python research.py train    --agent JOLYAN --scale 0.25 --window 0 --seeds 5 --record
python research.py evaluate --agent JOLYAN --scale 0.25 --record
```

### The result

```
JOLYAN_v1: REJECT
  [FAIL] beats_best_baseline        sortino -4.356 vs best baseline always_short 0.683
  [PASS] beats_champion             no incumbent champion; baseline condition governs
  [PASS] minimum_trades             837 trades, need 30
  [PASS] minimum_seeds              5 distinct seed(s), need 5
  [FAIL] positive_median_return     median return -1.81% across 55 runs
  [PASS] no_catastrophic_regime     worst regime DD 51.53% vs allowed 103.05%
  [FAIL] recent_windows_hold_up     0 of the last 3 windows at or above baseline, need 2
```

**This is the pipeline working, not failing.** RL on a single price series
overfits aggressively, and most trained policies *should* lose out of sample.
The point of building the baselines and the gate first was so that outcome would
be visible instead of flattering.

What the five seeds actually learned is more interesting than the rejection:

```
 seed   trades  median ret%  action mix
    0        0        +0.00  HOLD 42%  LONG  0%  SHORT  0%  CLOSE 58%
    1      193        -2.72  HOLD 39%  LONG 12%  SHORT  0%  CLOSE 49%
    2      183        -3.13  HOLD 58%  LONG  9%  SHORT  2%  CLOSE 31%
    3       90        -1.47  HOLD 19%  LONG  1%  SHORT 10%  CLOSE 69%
    4      371        -6.91  HOLD 50%  LONG  5%  SHORT  7%  CLOSE 38%
```

Seed 0 learned to **never open a position** — it emits only HOLD and CLOSE, and
CLOSE while flat is a no-op. That is the lazy local optimum the reward design
predicts out loud: doing nothing scores exactly zero, zero beats negative, and
there is deliberately no inactivity penalty to push it off that optimum. It is
also the correct answer on this tape, where `flat` beats every shipped strategy.
The four seeds that did trade all lost, and lost more the more they traded.

### Keeping torch out of the trading server

Training exports two artefacts: `model.zip` for SB3, and `weights.npz` +
`metadata.json` that `backend/policy/` reads with a hand-written numpy forward
pass. A process holding real positions should not be able to die because of a
broken CUDA install, and 500MB of GPU libraries have no business in a loop that
does four matrix multiplies per bar.

The guarantee is tested two ways: the numpy and SB3 forward passes agree to
**1.4e-7** on 300 random observations with identical `argmax` actions, and a
subprocess test imports the whole trading path (`backend.app`,
`backend.orchestrator`, `backend.policy`) in a clean interpreter and asserts
that `torch`, `stable_baselines3` and `gymnasium` are all absent from
`sys.modules`. That test previously passed by accident of collection order —
another test file imports torch — which is why it now runs in its own process.

### Four bugs this phase surfaced

Each was found by running the thing rather than by reading it, and each is now a
test.

**Evaluation seeds were fake diversity.** The gate requires a positive median
across ≥5 seeds. Inference is `argmax` and therefore deterministic, so running
one policy under five evaluation seeds produces five *identical* runs — and a
single lucky training run would have satisfied a requirement meant to prove
robustness. Seeds now come from `train_seed_ensemble`, which trains one policy
per seed.

**`UNIQUE (agent, version)` could not hold a seed ensemble.** Five policies
legitimately share one version. The constraint is now
`(agent, version, seed)`, with a migration that rebuilds the table rather than
telling anyone to delete their research history.

**The migration then corrupted three tables.** `ALTER TABLE ... RENAME` silently
rewrites every foreign key pointing at the renamed table — so `policy_metrics`,
`evaluations` and `champion_history` ended up referencing a temp table that was
then dropped, and every subsequent insert failed with "no such table". The
rename now runs under `PRAGMA legacy_alter_table`, and `_repair_dangling_references`
fixes databases already damaged this way.

**The regime veto was vacuous, then wrongly calibrated.** It reported one
arbitrary seed — which happened to be the do-nothing seed — so every regime
showed 0.00% drawdown and the condition passed without checking anything. Now it
pools all seeds and keeps the worst drawdown per regime. Fixing that exposed a
second problem: the 51.53% regime figure is compounded across windows while the
5.50% basis was a per-window median, a threshold the basis could never reach.
Both sides are now measured on the same stitched curve.

Stitching itself had two defects worth naming: concatenating per-window equity
curves injects a fabricated jump at every boundary (each window restarts at the
same capital, so 9,200 followed by 10,000 reads as a +8.7% bar nobody traded),
and trade indices are absolute dataset positions while the stitched arrays are
not — leaving them unmapped silently dropped every trade or filed it under the
wrong regime.

## Phase 4 — five agents that can be wrong about themselves

The five bots stop being hard-coded strategies and become research agents. Each
has a **bias** — JOLYAN explores, JONATHAN looks for trend, JOSEPH watches
regime, JOTARO minds risk, KIRA fades extension — and the important word is
*starts*.

A bias here is a prior over the search distribution. It skews which features are
**likely** to be sampled, never which are possible, and it is blended with
observed evidence on a shrinkage that shifts toward evidence as experiments
accumulate. At `PRIOR_STRENGTH` observations the two contribute equally; beyond
that, evidence dominates.

The test that matters is `test_an_agent_can_discover_its_starting_bias_is_wrong`:
JONATHAN begins trend-biased, is fed forty results in which reversion features
score well and trend features score badly, and must end up favouring reversion —
**without anyone editing its definition**. An agent that cannot be wrong about
its own premise is a hard-coded strategy wearing a costume, which is the thing
this phase replaces.

Two guardrails on that mechanism, both tested: two unlucky experiments must not
flip an agent's identity, and no feature is ever driven to zero weight, so a
comprehensively discredited input can still be rediscovered if the market
changes.

### What an agent may propose

`ExperimentSpec` is a complete, hashable description of one experiment — feature
subset, reward weights, PPO hyperparameters, and the risk envelope. The search
is **unbounded in kind, bounded in consequence**: any subset of the 24 features,
any reward weighting, any hyperparameters, because a bad idea costs CPU and
nothing else and restricting the search is how you forbid the thing that would
have worked. What it may not do is propose past the risk engine —
`risk_per_trade` and `leverage` are clamped to the hard caps at construction.

Specs are content-addressed, so "have we tried this?" is a lookup. The generator
rejects proposals that are near-duplicates of known failures, and the exclusion
radius **widens as dead ends accumulate**, so a search that keeps rediscovering
the same dead end is pushed further from it each time.

### The safeguard that stops the gate being gamed

Baselines are evaluated under a **fixed, neutral envelope — never the agent's**.
If they inherited the spec's proposed risk and stop parameters, an agent could
clear `beats_best_baseline` by proposing settings that cripple buy-and-hold
rather than by learning anything, and the gate would wave it through. Holding
the baselines fixed keeps the comparison honest: can this policy, with the
envelope it wants, beat a sensible strategy with a sensible one?

Baselines are also cached per (dataset, plan, seeds), since they do not depend on
the spec and recomputing nine of them per experiment would dominate a campaign.

### A campaign, end to end

```
--- KIRA: contrarian — fades extension, expects reversion
  e825d00c5113  score  -22.568    8165 trades  rejected
  650d428d2acc  score    0.000      47 trades  rejected
  -> 2 experiments, 0 candidate(s), bias drift 4%

--- JONATHAN: trend — leans on directional persistence
  943d673cd6d1  score    0.000       0 trades  rejected
      minimum_trades: 0 trades, need 30
  a9e3b3929235  score  -20.106    6705 trades  rejected
  -> 2 experiments, 0 candidate(s), bias drift 7%
```

Propose → train a seed ensemble → walk-forward every window → gate → record →
update the prior. Campaigns are sequential rather than batched so each proposal
sees the previous outcome; an agent that cannot react to its own last result is
running a random search with extra steps.

JONATHAN's first hypothesis produced **zero trades** — the do-nothing optimum
again, now caught by the gate's minimum-trade condition rather than by the
reward, exactly as designed.

Hypothesis generation is **programmatic, not an LLM**: a seeded search over a
typed space, so the whole pipeline stays deterministic and testable before any
language model is involved. The LLM enters in Phase 6 as an additional proposer
subject to identical gates — it can emit a hypothesis, never a trade.

## Phase 5 — JOJO manages the desk

JOJO ranks the agents, decides each one's champion, allocates **paper** capital,
and retires champions that have decayed. It does not trade.

### What JOJO cannot do

`research/jojo.py` never imports `backend.risk`, `backend.config`,
`backend.execution` or `backend.orchestrator`. Its only lever is a row in the
`allocations` table. That is asserted against the module's **import graph** with
an AST walk, not promised in a docstring — "JOJO must not be able to override
hard risk controls" is a property of the dependency graph here. A second test
runs an allocation and asserts `allocations` is the only table that changed.

### Correlation is a risk, not a win

Five agents that all discovered the same edge are not five edges; they are one
bet with five names on it, and a desk splitting capital equally across them is
far more concentrated than its paperwork suggests.

Correlation is measured on **per-bar equity returns**, not positions — positions
answer "are they holding the same thing", returns answer "do they lose money at
the same time", which is what a portfolio cares about. Window boundaries are
dropped from the return series, since each window restarts at the same capital
and that reset is nobody's return.

The haircut **reduces deployment, it does not reshuffle it.** That distinction
was a bug I found by testing it: applying the penalty only to relative weights
left a fully converged desk deploying exactly as much capital as a diversified
one — the concentration was invisible in the only number that matters. Now three
identical agents receive less than 75% of what three independent ones do. Only
*positive* correlation is charged; genuine diversification is not taxed.

A desk whose mean pairwise correlation exceeds 0.7 gets a loud warning in the
report.

### Allocation rules

- an agent scoring at or below zero gets **nothing** — zero is what doing
  nothing scores, and paying an agent to underperform doing nothing is how a
  desk funds its own losses;
- no agent exceeds `max_share` of deployable capital;
- `reserve_fraction` is never deployed, so a promising newcomer can be funded
  without first taking capital from an incumbent — a desk that always deploys
  100% is structurally biased toward whoever is already there;
- capital freed by capping returns to the reserve rather than being
  redistributed, so one strong agent cannot exceed its cap via everyone else's
  leftovers.

### Promotion and retirement are deliberately asymmetric

A challenger that clears the gate becomes its agent's champion — a **research**
status that deploys nothing. Live trading stays behind the existing three-part
gate and an explicit human decision, per policy and per venue.

Retirement, by contrast, is automatic: a champion that loses to the best
baseline in enough recent windows, or that decays past a fraction of the score
it was promoted on, stands down without asking. Standing down reduces exposure,
so erring toward it is cheap; deploying is the direction that costs money when
wrong.

### The desk today

```
agent           sortino     maxDD  trades    corr     paper $   note
KIRA              0.000     0.00%      15   +0.01           0   beaten by doing nothing
flat              0.000     0.00%       0   +0.00           0   beaten by doing nothing
buy_and_hold     -0.876     4.59%      33   -0.06           0   beaten by doing nothing
JOLYAN           -4.356     5.50%     837   +0.00           0   beaten by doing nothing
JONATHAN        -19.741    18.47%    3947   -0.09           0   beaten by doing nothing

desk mean pairwise correlation: -0.03
deployed 0 of 50,000 paper capital
```

**JOJO allocates zero to everyone**, because nothing on the desk beats doing
nothing. That is the correct answer, and it is the one the system is built to be
able to give.

## Tests

`pytest` — 436 tests, of which 331 are new:

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
- `test_research_train.py` — export, numpy/SB3 parity, the training contract,
  and the seed-diversity distinction
- `test_research_gate.py` — every promotion condition, each failed in isolation
- `test_policy_runtime.py` — the numpy inference path and the torch-isolation
  guarantee
- `test_research_agents.py` — the hypothesis space, memory, the baseline-gaming
  safeguard, and above all that a bias can be overturned by evidence
- `test_research_jojo.py` — correlation, allocation caps and reserve, promotion
  and retirement, and the import-graph proof that JOJO cannot reach the risk
  engine

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
