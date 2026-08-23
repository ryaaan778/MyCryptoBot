# JOJO Trading Command Center — User Guide

## Overview

A multi-bot crypto trading desk presented as a voxel world. Five bots run
independent strategies while JOJO orchestrates them; the world visualises their
state, and the backend does the actual trading.

> **This guide previously documented `bot/final_bot.py`, `bot/strategy_optimizer.py`,
> `ui/dashboard.py` and `ui/data_generator.py`. Those files never existed in this
> repository — every command that referenced them failed at import.** The
> instructions below describe the system that is actually here.

The bots combine four approaches:

1. **Scalping** — fast EMA cross confirmed by RSI, on short timeframes
2. **Volatility breakout** — trades escapes from a Bollinger squeeze, filtered by ATR
3. **Momentum** — MACD cross gated by RSI's position against its midline
4. **Mean reversion** — fades z-score extremes back toward the rolling mean

plus a **hybrid** that takes a confidence-weighted vote of all four using the
`weight_*` values in `config.json`.

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (only to build the voxel world)
- A Binance account — optional; needed only for live market data or live trading

### Setup

```bash
git clone <this repository>
cd MyCryptoBot

pip install -r requirements.txt

cd web && npm install && npm run build && cd ..
```

---

## Running

### Paper trading on a synthetic feed (recommended first run)

```bash
python run_bot.py --provider simulated
```

Opens on <http://localhost:8000>. No exchange connection, no API keys, and the
market feed is generated locally — the fastest way to see the whole system work.

### Paper trading on live Binance market data

```bash
python run_bot.py
```

Real prices, simulated fills. If Binance is unreachable the system falls back to
the synthetic feed and says so, both in the logs and as a `MARKET DATA DEGRADED`
banner in the HUD — you will never be shown synthetic prices believing they are
real.

### Live trading

```bash
export LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK
python run_bot.py --mode live
```

This also requires `"live_enabled": true` under `trading` in `config.json` and a
configured API key and secret. Missing any of the three and the system runs on
paper and prints the reason. **Start on the exchange testnet**
(`"testnet": true`) before anything else.

### Options

| Flag | Meaning |
|---|---|
| `--mode paper\|live` | execution mode; `paper` forces the gate shut |
| `--provider auto\|binance\|simulated` | market data source |
| `--port`, `--host` | where to serve |
| `--data-dir` | where the SQLite database lives |
| `--balance` | starting paper balance |
| `--seed` | seed for the synthetic feed (makes runs reproducible) |
| `--log-level` | `debug` to see every decision |

### Frontend development

```bash
python run_bot.py --provider simulated   # backend on :8000
cd web && npm run dev                    # world on :5173
```

---

## Using the world

- **Click a character or their building** to fly the camera to that district and
  open their trading panel.
- **Camera bar** along the bottom: `WORLD`, `JOJO`, each bot, `MARKET`,
  `TRADING FLOOR`, `RISK CENTER`.
- **Drag** to orbit, **scroll** to zoom, **right-drag** to pan.
- **`DATA TABLES`** opens conventional sortable views of positions, orders,
  trades, the event log and the audit trail.
- **`2D MODE`** switches to the plain dashboard at any time. `?mode=2d` in the
  URL makes that the default; `?mode=3d` pins the world on.
- **Keyboard**: `1`–`5` select a bot, `W`/`J`/`M` jump to world / HQ / market,
  `T` toggles tables, `Esc` deselects.

### Reading a bot's state from the world

| What you see | What it means |
|---|---|
| Character breathing, district lit | Online and idle |
| Head down, scanning | Analysing the market |
| Striking pose, projectiles flying to the market | Submitting orders |
| Gold burst, warm ground, rising number | Closed a winner |
| Desaturated district, dark motes | Losing |
| Pulsing warning ring | Risk elevated toward its limit |
| Slumped, lanterns out | Offline or halted |
| The whole world red, pillar over the HQ | Emergency stop |

### Bot controls

The detail panel has `START`, `PAUSE`, `RESUME` and `STOP`. `STOP` flattens that
bot's positions before taking it offline. The red **EMERGENCY STOP** in the top
bar halts every bot and immediately closes every open position; it asks for
confirmation first, and turns into `RESUME SYSTEM` afterwards.

---

## Risk management

Enforced centrally — a strategy cannot bypass a limit:

- **Position sizing** — quantity is chosen so that being stopped out costs
  exactly the bot's configured per-trade risk budget.
- **Stop-loss / take-profit** — set on entry from the bot's percentages and
  checked on *every* price tick, never on a throttled schedule.
- **Per-bot position cap** — `max_open_trades` in `config.json`.
- **Desk-wide position cap** — derived as `max_open_trades × bot count`, or set
  explicitly with `max_portfolio_positions`.
- **Exposure limit** — total notional as a percentage of equity.
- **Drawdown limit** — new entries stop once drawdown reaches the limit.
- **Emergency stop** — flattens everything and halts the desk.

The risk meter beside JOJO shows the worst of exposure, drawdown and position
pressure as a single 0–100% reading. Being fully deployed alone reads as
elevated, not critical — only real exposure or drawdown pressure takes the meter
into the danger bands, so the reading keeps carrying information when the desk
is simply busy.

---

## Configuration

`config.json` drives everything:

```json
{
  "exchange":  { "name": "binance", "api_key": "", "api_secret": "", "testnet": true },
  "trading":   { "symbols": ["BTC/USDT", "..."], "leverage": 10, "live_enabled": false },
  "risk_management": {
    "max_risk_per_trade": 0.1,
    "max_open_trades": 3,
    "stop_loss_pct": 1.0,
    "take_profit_pct": 7.0
  },
  "strategy_parameters": { "scalping_ema_short": 3, "...": "..." },
  "initial_balance": 10000
}
```

Add a `bots` array to change the roster — each entry takes `id`, `name`,
`strategy`, `symbol`, `timeframe`, `allocation`, `leverage`, `risk_per_trade`,
`stop_loss_pct`, `take_profit_pct`, `interval_sec` and a `persona` block that
controls the character's palette, title and position in the world.

Key parameters:

- **`leverage`** — multiplies gains *and* losses, and brings liquidation closer
- **`max_risk_per_trade`** — ceiling on the fraction of allocated equity a
  single trade may lose at its stop
- **`stop_loss_pct` / `take_profit_pct`** — distance from entry to each level

---

## About the performance target

Earlier versions of this guide advertised 5–10% daily returns. The backtest
recorded in `trading_bot.log` found the opposite result with these parameters:

| Metric | Result |
|---|---|
| Trades | 33 |
| Win rate | 21.21% |
| Total return | **-7.93%** |
| Average daily | -0.72% |
| Days hitting the 5–10% target | **0.00%** |

The parameters in `config.json` have been left untouched, but they should be
treated as unvalidated rather than as a proven configuration. A 10x leverage
setting combined with risking 10% of the balance per trade can lose the account
quickly. Run on paper, watch the drawdown reading, and change the parameters
based on your own results.

---

## Troubleshooting

**"Dependencies are missing"** — `pip install -r requirements.txt`.

**The page says the world is not built** — `cd web && npm install && npm run build`,
or use `npm run dev` for the dev server.

**`MARKET DATA DEGRADED` banner** — Binance was unreachable and the synthetic
feed took over. Check connectivity and regional access; prices shown are *not*
real while this banner is up.

**Everything shows as PAPER when you wanted live** — check `/api/system`, which
reports exactly which of the three gate conditions is unmet.

**The world switched itself to the 2D dashboard** — the frame rate could not be
sustained. All trading data and controls are still there. `?mode=3d` forces the
world back on.

**No trades appear** — strategies need enough candle history to warm up, and
they legitimately decline to trade in flat markets. `--log-level debug` shows
each decision and its reason.

---

## Disclaimer

This software trades financial instruments and can lose money. Past performance,
including any backtest in this repository, does not indicate future results. Use
at your own risk and never trade with money you cannot afford to lose.
