/**
 * The 2D fallback.
 *
 * Reached when WebGL is unavailable, when the frame rate can't sustain the
 * world, or when the user asks for it with `?mode=2d`. It reads the *same*
 * store as the world, so nothing about the trading data is second-class here:
 * bots, positions, orders, P&L, risk, events and every command are all present.
 *
 * A world that stutters over live prices is worse than no world at all — this
 * is the view that guarantees the desk stays visible either way.
 */

import { useMemo } from 'react'

import { worldSocket } from '../net/ws'
import { selectIsEmergency, useBots, useWorldStore } from '../state/store'
import { RISK_COLOR, THEME } from '../world/theme'
import {
  displaySymbol,
  money,
  pct,
  plainPct,
  price,
  signedMoney,
  timeOfDay,
} from '../world/format'
import {
  EventsTable,
  OrdersTable,
  PositionsTable,
  TradesTable,
} from '../hud/tables/DataTables'

export function Fallback2D({ reason }: { reason: string }) {
  const portfolio = useWorldStore((s) => s.portfolio)
  const risk = useWorldStore((s) => s.risk)
  const system = useWorldStore((s) => s.system)
  const connection = useWorldStore((s) => s.connection)
  const bots = useBots()
  const emergency = useWorldStore(selectIsEmergency)
  const setRenderMode = useWorldStore((s) => s.setRenderMode)
  const showResearch = useWorldStore((s) => s.showResearch)
  const toggleResearch = useWorldStore((s) => s.toggleResearch)
  const events = useWorldStore((s) => s.events)
  const tickers = useWorldStore((s) => s.tickers)

  const pnlColorFor = (value: number) =>
    value > 0 ? THEME.profit : value < 0 ? THEME.loss : undefined

  return (
    <div className="fallback" data-testid="fallback-2d">
      {emergency && (
        <div className="banner banner--emergency">
          ⚠ EMERGENCY STOP ACTIVE — ALL BOTS HALTED, BOOK FLATTENED
        </div>
      )}
      {connection !== 'live' && (
        <div className="banner banner--offline">
          {connection === 'reconnecting'
            ? 'CONNECTION LOST — RECONNECTING. DISPLAYED DATA MAY BE STALE.'
            : connection.toUpperCase()}
        </div>
      )}
      {system?.provider_degraded && (
        <div className="banner banner--degraded">
          ⚠ MARKET DATA DEGRADED — {system.provider_note.toUpperCase()}
        </div>
      )}

      <div className="topbar">
        <div className="topbar__brand">
          <div className="topbar__title">JOJO TRADING COMMAND CENTER</div>
          <div className="topbar__subtitle">SIMPLIFIED VIEW</div>
        </div>
        <div className="stat stat--grow">
          <span className="stat__label">Controls</span>
          <div className="row">
            <span className={`badge ${system?.mode === 'LIVE' ? 'badge--real' : 'badge--paper'}`}>
              {system?.mode === 'LIVE' ? '● LIVE TRADING' : '◆ PAPER TRADING'}
            </span>
            <button className="iconbutton" onClick={() => setRenderMode('3d')}>
              TRY 3D WORLD
            </button>
            <button className="iconbutton" onClick={toggleResearch}>
              {showResearch ? 'CLOSE RESEARCH' : 'RESEARCH'}
            </button>
            <button
              className={`button ${emergency ? 'button--go' : 'button--danger'}`}
              style={{ flex: 'none', padding: '5px 12px' }}
              onClick={() => {
                if (emergency) worldSocket.resume()
                else if (window.confirm(
                  'EMERGENCY STOP\n\nThis halts every bot and flattens all open positions. Continue?',
                )) worldSocket.emergencyStop()
              }}
            >
              {emergency ? 'RESUME SYSTEM' : 'EMERGENCY STOP'}
            </button>
          </div>
        </div>
      </div>

      <div className="fallback__notice">{reason}</div>

      {/* Headline numbers */}
      <div className="grid grid--stats">
        <Stat label="Total Equity" value={portfolio ? money(portfolio.equity) : '—'} />
        <Stat
          label="Total P&L"
          value={portfolio ? signedMoney(portfolio.total_pnl) : '—'}
          color={pnlColorFor(portfolio?.total_pnl ?? 0)}
          sub={portfolio ? pct(portfolio.total_pnl_pct) : undefined}
        />
        <Stat
          label="Today's P&L"
          value={portfolio ? signedMoney(portfolio.today_pnl) : '—'}
          color={pnlColorFor(portfolio?.today_pnl ?? 0)}
        />
        <Stat
          label="Open Positions"
          value={risk ? `${risk.open_positions}/${risk.max_open_positions}` : '—'}
        />
        <Stat
          label="Total Exposure"
          value={portfolio ? money(portfolio.total_exposure) : '—'}
          sub={risk ? `${plainPct(risk.exposure_pct, 0)} of equity` : undefined}
        />
        <Stat
          label="Drawdown"
          value={risk ? plainPct(risk.current_drawdown_pct, 2) : '—'}
          sub={risk ? `limit ${plainPct(risk.drawdown_limit_pct, 0)}` : undefined}
        />
        <Stat
          label="Risk Status"
          value={risk?.level ?? '—'}
          color={risk ? RISK_COLOR[risk.level] : undefined}
          sub={risk ? `${plainPct(risk.utilization * 100, 0)} utilized` : undefined}
        />
        <Stat
          label="Active Bots"
          value={`${bots.filter((b) => b.status !== 'OFFLINE' && b.status !== 'HALTED').length} / ${bots.length}`}
        />
        <Stat
          label="Win Rate"
          value={portfolio ? plainPct(portfolio.win_rate, 1) : '—'}
          sub={portfolio ? `${portfolio.trades_total} trades` : undefined}
        />
      </div>

      {/* Live prices */}
      <div className="section-title">Market</div>
      <div className="grid grid--stats">
        {Object.values(tickers).map((ticker) => (
          <Stat
            key={ticker.symbol}
            label={displaySymbol(ticker.symbol)}
            value={price(ticker.price)}
            color={pnlColorFor(ticker.change_24h_pct)}
            sub={`${pct(ticker.change_24h_pct)} · spread ${(((ticker.ask - ticker.bid) / ticker.price) * 10_000).toFixed(2)} bps`}
          />
        ))}
        {Object.keys(tickers).length === 0 && (
          <div className="empty">AWAITING MARKET DATA</div>
        )}
      </div>

      {/* Bots */}
      <div className="section-title">Bots</div>
      <div className="grid grid--bots">
        {bots.map((bot) => (
          <BotCard key={bot.id} botId={bot.id} />
        ))}
        {bots.length === 0 && <div className="empty">AWAITING ROSTER</div>}
      </div>

      <div className="section-title">Open Positions</div>
      <div className="card" style={{ margin: '0 12px', padding: 0, overflowX: 'auto' }}>
        <PositionsTable limit={12} />
      </div>

      <div className="section-title">Recent Trades</div>
      <div className="card" style={{ margin: '0 12px', padding: 0, overflowX: 'auto' }}>
        <TradesTable limit={20} />
      </div>

      <div className="section-title">Orders</div>
      <div className="card" style={{ margin: '0 12px', padding: 0, overflowX: 'auto' }}>
        <OrdersTable limit={20} />
      </div>

      <div className="section-title">Event Log ({events.length})</div>
      <div
        className="card"
        style={{ margin: '0 12px', padding: 0, maxHeight: 380, overflow: 'auto' }}
      >
        <EventsTable limit={30} />
      </div>
    </div>
  )
}

function Stat({
  label, value, color, sub,
}: {
  label: string
  value: string
  color?: string
  sub?: string
}) {
  return (
    <div className="card" style={color ? { borderTopColor: color } : undefined}>
      <div className="card__label">{label}</div>
      <div className="card__value" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="card__sub">{sub}</div>}
    </div>
  )
}

function BotCard({ botId }: { botId: string }) {
  const bot = useWorldStore((s) => s.bots[botId])
  const positions = useWorldStore((s) => s.positions)
  const trades = useWorldStore((s) => s.trades)

  const botPositions = useMemo(
    () => Object.values(positions).filter((p) => p.bot_id === botId),
    [positions, botId],
  )
  const equityTrail = useMemo(
    () =>
      trades
        .filter((t) => t.bot_id === botId)
        .slice(-24)
        .reduce<number[]>((acc, trade) => {
          acc.push((acc[acc.length - 1] ?? 0) + trade.realized_pnl)
          return acc
        }, []),
    [trades, botId],
  )

  if (!bot) return null

  const offline = bot.status === 'OFFLINE' || bot.status === 'HALTED'
  const pnlColor = bot.total_pnl > 0 ? THEME.profit : bot.total_pnl < 0 ? THEME.loss : undefined

  return (
    <div className="card" style={{ borderTopColor: bot.persona.accent }}>
      <div className="botcard__head">
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: bot.persona.accent }}>
            {bot.name}
          </div>
          <div className="card__sub" style={{ marginTop: 2 }}>{bot.persona.title}</div>
        </div>
        <span
          className="badge"
          style={{
            color: offline ? THEME.loss
              : bot.status === 'TRADING' ? THEME.warning : THEME.profit,
          }}
        >
          {bot.status}
        </span>
      </div>

      <div className="card__value" style={{ color: pnlColor }}>
        {signedMoney(bot.total_pnl)}
      </div>

      {equityTrail.length > 1 && <Sparkline values={equityTrail} />}

      <Row label="Strategy" value={bot.strategy.replace(/_/g, ' ')} />
      <Row label="Symbol" value={`${displaySymbol(bot.symbol)} · ${bot.timeframe}`} />
      <Row label="Leverage" value={`${bot.leverage.toFixed(0)}x`} />
      <Row label="Win Rate" value={`${plainPct(bot.win_rate, 1)} (${bot.trades_won}/${bot.trades_total})`} />
      <Row label="Exposure" value={money(bot.exposure)} />
      <Row
        label="Risk"
        value={`${bot.risk_level} · ${plainPct(bot.risk_utilization * 100, 0)}`}
        color={RISK_COLOR[bot.risk_level]}
      />
      <Row
        label="Position"
        value={
          botPositions.length
            ? `${botPositions[0].side} @ ${price(botPositions[0].entry_price)}`
            : 'FLAT'
        }
        color={
          botPositions.length
            ? botPositions[0].side === 'LONG' ? THEME.profit : THEME.loss
            : undefined
        }
      />
      {bot.last_signal && (
        <Row
          label="Signal"
          value={
            bot.last_signal.action === 'HOLD'
              ? 'HOLD'
              : `${bot.last_signal.action} ${plainPct(bot.last_signal.confidence * 100, 0)}`
          }
        />
      )}
      {bot.last_heartbeat > 0 && (
        <Row label="Heartbeat" value={timeOfDay(bot.last_heartbeat)} />
      )}

      <div className="controls" style={{ marginTop: 10, padding: '8px 0 0 0', background: 'none' }}>
        <button
          className="button button--go"
          disabled={!offline}
          onClick={() => worldSocket.command(bot.id, 'start')}
        >
          START
        </button>
        <button
          className="button"
          disabled={offline || bot.status === 'PAUSED'}
          onClick={() => worldSocket.command(bot.id, 'pause')}
        >
          PAUSE
        </button>
        <button
          className="button"
          disabled={bot.status !== 'PAUSED'}
          onClick={() => worldSocket.command(bot.id, 'resume')}
        >
          RESUME
        </button>
        <button
          className="button button--danger"
          disabled={offline}
          onClick={() => worldSocket.command(bot.id, 'stop')}
        >
          STOP
        </button>
      </div>
    </div>
  )
}

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 8,
        fontSize: 11,
        padding: '3px 0',
        borderBottom: '1px solid rgba(120,150,220,0.08)',
      }}
    >
      <span style={{ color: 'var(--hud-muted)', textTransform: 'uppercase', fontSize: 10 }}>
        {label}
      </span>
      <span style={{ fontWeight: 700, color, fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  )
}

/** Cumulative realised P&L, drawn as inline SVG — no chart library needed. */
function Sparkline({ values }: { values: number[] }) {
  const width = 240
  const height = 48
  const min = Math.min(...values, 0)
  const max = Math.max(...values, 0)
  const span = max - min || 1

  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width
      const y = height - ((value - min) / span) * height
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  const zeroY = height - ((0 - min) / span) * height
  const last = values[values.length - 1] ?? 0

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <line
        x1={0} y1={zeroY} x2={width} y2={zeroY}
        stroke="rgba(147,160,189,0.35)" strokeWidth={1} strokeDasharray="3 3"
      />
      <polyline
        points={points}
        fill="none"
        stroke={last >= 0 ? THEME.profit : THEME.loss}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}
