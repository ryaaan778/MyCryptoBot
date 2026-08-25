# JOJO Trading Command Center

A live multi-bot crypto trading desk whose interface is a voxel world.

Five autonomous bots — **JONATHAN**, **JOSEPH**, **JOTARO**, **JOLYAN** and
**KIRA** — each run their own strategy on their own market. **JOJO** orchestrates
them from the headquarters at the centre of the world. Every character, aura,
particle and floating board is driven by real backend state: bot status, live
P&L, open positions, risk utilisation and order flow. Clicking a character flies
the camera to their district and opens their trading panel.

The world is the dashboard. The bots are characters. Trades are events happening
inside the world.

![The world](web/screenshots/01-world.png)

---

## Quick start

```bash
python start.py
# → http://localhost:8000
```

`start.py` checks the Python dependencies, builds the voxel world if it has not
been built, and starts the desk. It prints every step before taking it. If Node
is missing it says so and starts the API anyway rather than refusing.

By hand, it is these three steps:

```bash
pip install -r requirements.txt
cd web && npm install && npm run build && cd ..
python run_bot.py --provider simulated
```

`--provider simulated` runs against a synthetic market feed, which is the
fastest way to see the whole system working. Drop the flag to use live Binance
market data (still paper-traded).

For frontend development, run the backend and the Vite dev server side by side:

```bash
python run_bot.py --provider simulated     # :8000
cd web && npm run dev                      # :5173, proxies /api and /ws
```

---

## Paper trading vs. live trading

**The system paper-trades by default and cannot place a real order unless you
open a three-part gate.** All three conditions must hold:

1. `"live_enabled": true` under `trading` in `config.json`
2. an API key *and* secret under `exchange`
3. `LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK` in the environment

Miss any one and the paper matching engine is used instead, with the reason
reported through `/api/system`. The gate is re-checked on **every order**, not
only at startup, so a config reload can't quietly open it mid-session. The HUD
shows a `PAPER TRADING` or `LIVE TRADING` badge at all times.

The paper engine is not a flattering simulation: buys pay the ask, sells hit the
bid, both take slippage, and fees are charged on notional.

### A note on the configured targets

`USER_GUIDE.md` describes a target of 5–10% daily returns at 10x leverage with
10% of the balance risked per trade. The backtest recorded in `trading_bot.log`
found the opposite: 33 trades, a 21.21% win rate, **-7.93% total**, and **0.00%
of days** reaching the target. The strategy parameters in `config.json` are left
exactly as they were — but the risk surfaces in this build (drawdown, exposure,
per-bot risk auras, the emergency stop) exist to make that reality visible
rather than hide it. Treat the defaults as untested, and start on paper.

---

## Model-driven trading

Any bot can run on a language model instead of a hand-written strategy. Set its
`strategy` to `"llm"`, turn the feature on, and give it a key:

```jsonc
// config.json
"llm": {
  "enabled": true,
  "all_bots": true,             // put the whole roster on the model
  "model": "claude-opus-5",
  "web_search": true,
  "decide_every_sec": 300,      // per bot — this is what the API bill tracks
  "daily_call_budget": 500,     // per bot, hard stop
  "min_confidence": 0.5         // drop to 0 to act on every stance
}
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python start.py
```

The agent reads its market, its own position, the account's drawdown, and its
record of past decisions on that market — then searches the web if current
context would change its read, and returns a stance.

**What the model actually controls, and what it does not.** It returns a
direction and a conviction. That is the whole vocabulary. A stance has no
quantity field, no leverage field and no price, so there is no sentence the
model can produce that moves more capital than the envelope allows. Size comes
from `RiskEngine.size_position`, which takes the bot config, the price and the
stop distance — confidence is not one of its arguments, and a test asserts the
word does not appear in `backend/risk.py` at all. Conviction decides *whether*
to act, never *how much*.

This matters most for the thing the feature makes possible: the agent reads the
open web, and web pages are written by people who may benefit from what it does
next. Some are written specifically to manipulate systems like this one. A
successful prompt injection can change which way an agent leans. It cannot
change the size, cannot reach the risk engine, and cannot place an order — the
risk engine never sees the retrieved content at all. The realistic worst case is
one bad trade, sized exactly as any other trade would be.

**It learns from its own record.** Every stance is written to an append-only
JSONL file per bot, and when the position it opened closes — by the model's
call, a stop, a take-profit, or the risk engine flattening it — the realised
P&L is attached. The next prompt carries the recent decisions with outcomes, a
calibration table of stated confidence against actual hit rate, and, for losing
calls, the invalidation the model wrote for itself before it found out. An agent
that has been claiming 0.9 and hitting 45% is shown that in the same breath as
it is asked for a new number.

**Cadence is separate from the tick.** A reasoning call takes tens of seconds; a
tick is a few. The brain runs on its own schedule in the background and the
trading loop reads whatever stance it currently holds, so the world never stalls
on the API. A stance expires on the horizon the model gave it, capped by
`stance_ttl_sec` — a six-hour-old opinion about a market that has since moved 8%
is worse than no opinion. No live stance reads as HOLD, which is also what a
missing key, an expired card or a network partition degrades to. A bot that
cannot reach its model stops trading; it never falls back to guessing.

**Agents choose their own exits.** A stance may name its own stop distance and
target, not just a direction. This looks like handing over risk control and is
the opposite: sizing solves `quantity = risk_budget / (price * stop_distance)`,
so a wider stop buys a *smaller* position and a tighter stop a larger one with
less room. Money at risk on the trade is identical at every stop distance —
pinned by `risk_per_trade`, which the agent cannot see or set. The risk engine
clamps the ask to a band where the arithmetic stays meaningful and nothing else.
So the agent owns trade *structure* completely, and cannot convert that into
exposure.

---

## JOJO's capital allocation

JOJO does not trade. It decides how much money each agent gets, and revises it as
results come in — winners are given more to work with, losers less.

```jsonc
"allocator": {
  "enabled": true,
  "interval_sec": 900,
  "min_trades_for_conviction": 25,   // trades needed to exceed an even share
  "floor": 0.05,                     // nobody is starved to zero
  "ceiling": 0.40,                   // nobody takes the whole desk
  "smoothing": 0.35                  // move part of the way each round
}
```

Allocating in proportion to recent P&L is a well-known way to lose money — you
buy a streak at its peak and cut an agent right before it recovers. Four things
stop this from being that, and none of them limits how freely an agent trades:

- **Edge is measured per unit of notional traded**, never against the agent's own
  budget. Scoring against the budget creates a feedback loop where cutting an
  agent inflates its apparent edge and wins the money straight back. That bug
  put the *worst* agent on the desk at the ceiling within a few rounds; there is
  now a regression test for it.
- **A short streak cannot capture the desk.** Below `min_trades_for_conviction`
  an agent is held to an even share no matter how good three trades looked.
  Shrinkage alone does not handle this: three trades averaging +300 really is a
  high point estimate. What is missing at n=3 is grounds, not magnitude.
- **A floor and a ceiling.** An agent on zero capital can never generate the
  evidence that would win it back — an absorbing state, and how a portfolio
  quietly becomes one strategy.
- **Budgets ease toward the target** rather than jumping to each fresh estimate,
  because allocation computed on noisy samples is itself noisy.

When nobody is profitable, everyone drops to the floor and the rest of the desk
sits in cash rather than being deployed behind a roster with no demonstrated
edge. Running all five with a known synthetic edge per agent, budget converges on
the right one within a few hundred trades:

```
 round  trades   JONATHAN     JOSEPH     JOTARO     JOLYAN       KIRA
 start       0     22.0%      20.0%      24.0%      20.0%      14.0%
     4     100      8.2%       7.8%      11.4%      36.4%      35.4%
     8     200      6.3%      10.9%      33.2%      10.8%      38.7%

 true edge/trade    -0.6       -1.4       +0.2       -0.3       +2.6
```

Allocation is a budget, never an instruction. It changes how large an agent's
positions may be and nothing about what it decides to trade.

This is paper trading. Real-money execution is still behind the unchanged
three-part gate in **[Paper trading vs. live trading](#paper-trading-vs-live-trading)**.

---

## Research pipeline

The five bots also exist as **research agents** that form hypotheses, train
reinforcement-learning policies, and evaluate them against baselines out of
sample. That half of the system is entirely offline — it places no orders, and
`research/` imports `backend/` but never the reverse.

```bash
python research.py synth --bars 60000          # deterministic simulated bars
python research.py baselines --scale 0.25      # every strategy, walk-forward
python research.py campaign --agent KIRA       # propose -> train -> gate
python research.py desk --baselines            # JOJO's ranking and allocation
```

Training needs the isolated extras (`pip install -r requirements-research.txt`);
everything else runs on the base install. Full write-up, including what the
baselines actually show, is in **[RESEARCH.md](RESEARCH.md)**.

The short version of the findings so far: on the synthetic tape, **doing nothing
beats every shipped strategy**, and the 7% take-profit in `config.json` sits
29 ATR away on a 5-minute chart and fires on 0.00% of trades. Those are
simulator results and are labelled as such everywhere they appear.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
   Binance / synth  │  MarketDataProvider (pluggable)      │
        feed  ─────▶│    binance.py  |  simulated.py       │
                    └──────────────┬───────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────┐
   │  JOJO Orchestrator — registry, heartbeats, capital,   │
   │                      emergency stop, coordination     │
   └───┬───────┬───────┬───────┬───────┬───────────────────┘
       ▼       ▼       ▼       ▼       ▼
    JOLYAN JONATHAN JOSEPH JOTARO  KIRA      (one async agent each)
       │       │       │       │       │
       └───────┴───┬───┴───────┴───────┘
                   ▼
        Strategy ▶ RiskEngine ▶ ExecutionAdapter (paper | live-gated)
                   │
                   ▼
              EventBus ──▶ SQLite (trades, orders, events, audit, equity)
                   │
                   ▼
        FastAPI:  REST /api/*   +   WebSocket /ws
                   │
                   ▼
        React + R3F voxel world  ──(no WebGL / low FPS)──▶  2D dashboard
```

### The roster

| Bot | Strategy | Market | Leverage | Character |
|---|---|---|---|---|
| JONATHAN | momentum (MACD + RSI) | BTC/USDT 15m | 3x | The Foundation — broad silhouette, long coat, shoulder guards |
| JOSEPH | volatility breakout | ETH/USDT 15m | 4x | The Trickster — wide-brimmed hat, scarf |
| JOTARO | scalping (EMA cross + RSI) | BTC/USDT 5m | 5x | The Striker — peaked cap merged into the hair, gold chain |
| JOLYAN | hybrid weighted vote | SOL/USDT 5m | 4x | The Unbound — twin buns, trailing threads |
| KIRA | mean reversion (z-score) | BNB/USDT 15m | 8x | The Quiet One — neat suit and tie, one raised hand |

JOJO orchestrates and holds no positions.

### How the world shows state

| State | The world does this |
|---|---|
| Online | Character breathes and shifts weight, district lit |
| Analyzing | Head down, scanning; holographic data nearby |
| Trading | Striking pose, order projectiles fly to the market |
| Profitable | Gold particle burst, warm district ground, rising P&L |
| Loss | Desaturated district, dark motes, red P&L |
| High risk | Pulsing warning ring and coloured light around the character |
| Offline / halted | Character slumps, lanterns and windows go dark |
| Emergency stop | The whole world turns red, a warning pillar rises over the HQ |

---

## Why trading data always wins

Two mechanisms, because a dashboard that stutters over live prices is worse than
no dashboard:

**On the wire.** Order, fill, position, trade and risk frames are *never*
coalesced or dropped. Decorative price ticks collapse to the latest value per
symbol and flush at a fixed rate, so a fast market cannot starve the lifecycle
stream. If a client is so slow its lifecycle queue overflows, the connection is
closed rather than silently losing a fill.

**In the browser.** Socket frames are applied to the store synchronously, never
behind an animation frame. Visual effects go into a bounded queue with a
per-frame budget and are dropped oldest-first when the budget is exceeded. If
the frame rate still can't hold, quality degrades a tier at a time; if it
*still* can't, the app switches to the 2D dashboard, where every bot, position,
order, trade, event and command remains available.

Force either view with `?mode=2d` or `?mode=3d`.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness, mode, provider |
| `GET /api/state` | the full snapshot the world renders from |
| `GET /api/system` | execution mode and why the live gate is open or shut |
| `GET /api/bots`, `/api/bots/{id}` | roster and per-bot detail |
| `POST /api/bots/{id}/command` | `start` · `pause` · `resume` · `stop` · `flatten` |
| `GET /api/positions`, `/orders`, `/trades`, `/events`, `/risk`, `/portfolio`, `/equity` | book and history |
| `GET /api/market/{symbol}/klines` | candles (accepts `BTCUSDT` or `BTC/USDT`) |
| `GET /api/audit` | append-only audit trail |
| `POST /api/system/emergency-stop`, `/resume` | kill switch |

**WebSocket `/ws`** sends one `snapshot` frame on connect, then deltas:
`bot.status` · `bot.signal` · `bot.heartbeat` · `order.submitted` ·
`order.filled` · `order.cancelled` · `position.opened` · `position.updated` ·
`position.closed` · `trade.closed` · `pnl.tick` · `risk.update` ·
`market.tick` · `market.kline` · `event.log` · `system.emergency_stop` ·
`market.provider_degraded`.

Clients send `bot.<action>`, `system.emergency_stop`, `system.resume`,
`snapshot.request` and `ping`.

---

## Configuration

`config.json` is the source of truth. Strategy parameters, symbols and risk
limits are read from the blocks that were already there.

One semantic change worth knowing: `risk_management.max_open_trades` was written
for a single-bot system, so it is now the **per-bot** cap, and a desk-wide cap
(`max_portfolio_positions`) is derived from it. Without that, three bots filled
the book and the other two could never trade at all.

Environment overrides: `JOJO_PROVIDER`, `JOJO_HOST`, `JOJO_PORT`, `JOJO_SEED`,
`JOJO_DATA_DIR`, `JOJO_INITIAL_BALANCE`, `BINANCE_API_KEY`,
`BINANCE_API_SECRET`.

---

## Tests

```bash
pytest                       # 105 backend tests
cd web && npm run build      # typecheck + production build
cd web && npx playwright test # 9 end-to-end tests against a live backend
```

The end-to-end suite needs a backend running on the port in
`web/playwright.config.ts` (8012 by default):

```bash
JOJO_PROVIDER=simulated python -m uvicorn backend.app:app --port 8012
```

It drives the real world in a real browser: renders it, flies the camera,
selects bots, opens the tables, triggers the emergency stop and checks the
backend actually halted, and confirms the 2D fallback carries the full desk.
Screenshots land in `web/screenshots/`.

---

## Keyboard

`1`–`5` select a bot · `W` world view · `J` headquarters · `M` market room ·
`T` data tables · `Esc` deselect

---

## Disclaimer

Trading carries real financial risk, and leverage multiplies it. Past results —
including the backtest in this repository — do not predict future results. Run
on paper until you have your own evidence, and never trade money you cannot
afford to lose.
