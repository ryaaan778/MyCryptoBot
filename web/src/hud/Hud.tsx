/**
 * The HUD overlay.
 *
 * Deliberately restrained: the world is the spectacle, this is the instrument
 * panel. Everything here stays legible no matter what the environment is doing
 * behind it, and every control maps to a real backend command.
 */

import { useEffect, useMemo, useState } from 'react'

import type { BotState } from '../net/protocol'
import { worldSocket } from '../net/ws'
import { selectIsEmergency, useBots, useWorldStore } from '../state/store'
import { CAMERA_PRESETS, RISK_COLOR, THEME } from '../world/theme'
import {
  displaySymbol,
  money,
  pct,
  plainPct,
  price,
  quantity,
  signedMoney,
  timeOfDay,
} from '../world/format'
import { DataTables } from './tables/DataTables'

export function Hud() {
  const showTables = useWorldStore((s) => s.showTables)
  return (
    <div className="hud">
      <div>
        <Banners />
        <TopBar />
      </div>

      <div className="hud__bottom">
        <BotRoster />
        <CameraBar />
        <EventLog />
      </div>

      <BotPanel />
      {showTables && <DataTables />}
    </div>
  )
}

// --------------------------------------------------------------------- banners

function Banners() {
  const connection = useWorldStore((s) => s.connection)
  const system = useWorldStore((s) => s.system)
  const emergency = useWorldStore(selectIsEmergency)

  return (
    <>
      {emergency && (
        <div className="banner banner--emergency">
          ⚠ EMERGENCY STOP ACTIVE — ALL BOTS HALTED, BOOK FLATTENED
        </div>
      )}
      {connection !== 'live' && (
        <div className="banner banner--offline">
          {connection === 'connecting'
            ? 'CONNECTING TO THE TRADING ENGINE…'
            : connection === 'reconnecting'
              ? 'CONNECTION LOST — RECONNECTING. DISPLAYED DATA MAY BE STALE.'
              : 'DISCONNECTED FROM THE TRADING ENGINE'}
        </div>
      )}
      {system?.provider_degraded && (
        <div className="banner banner--degraded">
          ⚠ MARKET DATA DEGRADED — {system.provider_note.toUpperCase()}
        </div>
      )}
    </>
  )
}

// ---------------------------------------------------------------------- top bar

function TopBar() {
  const connection = useWorldStore((s) => s.connection)
  const system = useWorldStore((s) => s.system)
  const portfolio = useWorldStore((s) => s.portfolio)
  const risk = useWorldStore((s) => s.risk)
  const emergency = useWorldStore(selectIsEmergency)
  const toggleTables = useWorldStore((s) => s.toggleTables)
  const showTables = useWorldStore((s) => s.showTables)
  const showResearch = useWorldStore((s) => s.showResearch)
  const toggleResearch = useWorldStore((s) => s.toggleResearch)
  const setRenderMode = useWorldStore((s) => s.setRenderMode)
  const renderMode = useWorldStore((s) => s.renderMode)

  const pnlClass = (value: number) =>
    value > 0 ? 'value--up' : value < 0 ? 'value--down' : 'value--flat'

  return (
    <div className="topbar">
      <div className="topbar__brand">
        <div className="topbar__title">JOJO TRADING COMMAND CENTER</div>
        <div className="topbar__subtitle">
          {system ? `${system.provider.toUpperCase()} · ${Object.keys(CAMERA_PRESETS).length ? '' : ''}` : ''}
          MULTI-BOT AUTONOMOUS DESK
        </div>
      </div>

      <div className="stat">
        <span className="stat__label">Connection</span>
        <span
          className={`badge ${connection === 'live' ? 'badge--live' : 'badge--down'}`}
        >
          <span className={`dot ${connection === 'live' ? 'dot--pulse' : ''}`} />
          {connection.toUpperCase()}
        </span>
      </div>

      <div className="stat">
        <span className="stat__label">Environment</span>
        <span
          className={`badge ${system?.mode === 'LIVE' ? 'badge--real' : 'badge--paper'}`}
        >
          {system?.mode === 'LIVE' ? '● LIVE TRADING' : '◆ PAPER TRADING'}
        </span>
      </div>

      <div className="stat">
        <span className="stat__label">Equity</span>
        <span className="stat__value">
          {portfolio ? money(portfolio.equity) : '—'}
        </span>
      </div>

      <div className="stat">
        <span className="stat__label">Total P&amp;L</span>
        <span className={`stat__value ${pnlClass(portfolio?.total_pnl ?? 0)}`}>
          {portfolio ? signedMoney(portfolio.total_pnl) : '—'}
        </span>
      </div>

      <div className="stat">
        <span className="stat__label">Today</span>
        <span className={`stat__value ${pnlClass(portfolio?.today_pnl ?? 0)}`}>
          {portfolio ? signedMoney(portfolio.today_pnl) : '—'}
        </span>
      </div>

      <div className="stat">
        <span className="stat__label">Risk</span>
        <span
          className="stat__value"
          style={{ color: risk ? RISK_COLOR[risk.level] : undefined }}
        >
          {risk?.level ?? '—'}
        </span>
      </div>

      <div className="stat stat--grow">
        <span className="stat__label">Controls</span>
        <div className="row">
          <button className="iconbutton" onClick={toggleTables}>
            {showTables ? 'CLOSE TABLES' : 'DATA TABLES'}
          </button>
          <button className="iconbutton" onClick={toggleResearch}>
            {showResearch ? 'CLOSE RESEARCH' : 'RESEARCH'}
          </button>
          <button
            className="iconbutton"
            onClick={() => setRenderMode(renderMode === '3d' ? '2d' : '3d')}
          >
            {renderMode === '3d' ? '2D MODE' : '3D WORLD'}
          </button>
          <button
            className={`button ${emergency ? 'button--go' : 'button--danger'}`}
            style={{ flex: 'none', padding: '5px 12px' }}
            onClick={() => {
              if (emergency) worldSocket.resume()
              else if (window.confirm(
                'EMERGENCY STOP\n\nThis halts every bot and immediately flattens all open positions. Continue?',
              )) {
                worldSocket.emergencyStop()
              }
            }}
          >
            {emergency ? 'RESUME SYSTEM' : 'EMERGENCY STOP'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ bot roster

const STATUS_COLOR: Record<BotState['status'], string> = {
  TRADING: THEME.warning,
  ANALYZING: THEME.hologram,
  IDLE: THEME.profit,
  PAUSED: '#9aa4bb',
  OFFLINE: '#5b6478',
  HALTED: THEME.critical,
  // Research states share a hue so they read as one family, and are distinct
  // from every trading state so the two are never confused at a glance.
  RESEARCHING: THEME.research,
  TRAINING: THEME.research,
  VALIDATING: THEME.researchDeep,
}

function BotRoster() {
  const bots = useBots()
  const selected = useWorldStore((s) => s.selectedBot)
  const selectBot = useWorldStore((s) => s.selectBot)
  const focusCamera = useWorldStore((s) => s.focusCamera)

  return (
    <div className="panel roster">
      <div className="panel__header">
        <span>Bot Status</span>
        <span>{bots.filter((b) => b.status !== 'OFFLINE' && b.status !== 'HALTED').length}/{bots.length} ACTIVE</span>
      </div>
      {bots.map((bot) => (
        <button
          key={bot.id}
          className={`roster__item ${selected === bot.id ? 'roster__item--active' : ''}`}
          style={{ borderLeftColor: selected === bot.id ? bot.persona.accent : 'transparent' }}
          onClick={() => { selectBot(bot.id); focusCamera(bot.id) }}
        >
          <span
            className={`dot ${bot.status === 'TRADING' ? 'dot--pulse' : ''}`}
            style={{ color: STATUS_COLOR[bot.status] }}
          />
          <span>
            <div className="roster__name">{bot.name}</div>
            <div className="roster__meta">
              {displaySymbol(bot.symbol)} · {bot.status}
            </div>
          </span>
          <span
            className="roster__pnl"
            style={{
              color: bot.total_pnl > 0 ? THEME.profit : bot.total_pnl < 0 ? THEME.loss : undefined,
            }}
          >
            {signedMoney(bot.total_pnl, 0)}
          </span>
        </button>
      ))}
      {bots.length === 0 && <div className="empty">AWAITING ROSTER</div>}
    </div>
  )
}

// ----------------------------------------------------------------- camera bar

function CameraBar() {
  const cameraTarget = useWorldStore((s) => s.cameraTarget)
  const focusCamera = useWorldStore((s) => s.focusCamera)
  const selectBot = useWorldStore((s) => s.selectBot)
  const renderMode = useWorldStore((s) => s.renderMode)

  if (renderMode === '2d') return <div />

  return (
    <div className="camerabar">
      {CAMERA_PRESETS.map((preset) => (
        <button
          key={preset.id}
          className={`camerabar__button ${cameraTarget === preset.id ? 'camerabar__button--active' : ''}`}
          onClick={() => {
            focusCamera(preset.id)
            if (preset.id === preset.id.toLowerCase()) selectBot(preset.id)
          }}
        >
          {preset.label}
        </button>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ event log

function EventLog() {
  const events = useWorldStore((s) => s.events)
  const recent = useMemo(() => events.slice(-40).reverse(), [events])

  return (
    <div className="panel eventlog">
      <div className="panel__header">
        <span>JOJO Event Log</span>
        <span>{events.length} EVENTS</span>
      </div>
      <div className="eventlog__list">
        {recent.map((event) => (
          <div key={event.id} className={`eventlog__row eventlog__row--${event.severity}`}>
            <span className="eventlog__time">{timeOfDay(event.ts)}</span>
            <span>{event.message}</span>
          </div>
        ))}
        {recent.length === 0 && <div className="empty">NO EVENTS YET</div>}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ bot panel

function BotPanel() {
  const bot = useWorldStore((s) => (s.selectedBot ? s.bots[s.selectedBot] : null))
  const positions = useWorldStore((s) => s.positions)
  const trades = useWorldStore((s) => s.trades)
  const tickers = useWorldStore((s) => s.tickers)
  const selectBot = useWorldStore((s) => s.selectBot)
  const [busy, setBusy] = useState(false)

  useEffect(() => { setBusy(false) }, [bot?.id])

  if (!bot) return null

  const botPositions = Object.values(positions).filter((p) => p.bot_id === bot.id)
  const botTrades = trades.filter((t) => t.bot_id === bot.id).slice(-6).reverse()
  const ticker = tickers[bot.symbol]
  const offline = bot.status === 'OFFLINE' || bot.status === 'HALTED'

  const send = (action: string) => {
    setBusy(true)
    worldSocket.command(bot.id, action)
    window.setTimeout(() => setBusy(false), 700)
  }

  return (
    <div className="panel botpanel">
      <div className="panel__header">
        <div className="botpanel__title">
          <span className="botpanel__name" style={{ color: bot.persona.accent }}>
            {bot.name}
          </span>
          <span className="botpanel__stand">{bot.persona.title}</span>
        </div>
        <button className="iconbutton" onClick={() => selectBot(null)}>✕</button>
      </div>

      {bot.persona.quote && <div className="botpanel__quote">“{bot.persona.quote}”</div>}

      <Kv label="Status" value={bot.status} color={STATUS_COLOR[bot.status]} />
      <Kv label="Stand" value={bot.persona.stand} />
      <Kv label="Strategy" value={bot.strategy.replace(/_/g, ' ').toUpperCase()} />
      <Kv label="Symbol" value={`${displaySymbol(bot.symbol)} · ${bot.timeframe}`} />
      <Kv label="Leverage" value={`${bot.leverage.toFixed(0)}x`} />
      <Kv label="Allocation" value={plainPct(bot.allocation * 100, 0)} />

      <div className="panel__header" style={{ borderTop: '1px solid var(--hud-border)' }}>
        <span>Performance</span>
      </div>
      <Kv
        label="Total P&L"
        value={signedMoney(bot.total_pnl)}
        color={bot.total_pnl > 0 ? THEME.profit : bot.total_pnl < 0 ? THEME.loss : undefined}
      />
      <Kv label="Realized" value={signedMoney(bot.realized_pnl)} />
      <Kv label="Unrealized" value={signedMoney(bot.unrealized_pnl)} />
      <Kv label="Win Rate" value={plainPct(bot.win_rate, 1)} />
      <Kv label="Trades" value={`${bot.trades_won} won / ${bot.trades_total}`} />
      <Kv label="Exposure" value={money(bot.exposure)} />
      <Kv
        label="Risk"
        value={`${bot.risk_level} · ${plainPct(bot.risk_utilization * 100, 0)}`}
        color={RISK_COLOR[bot.risk_level]}
      />

      <div className="panel__header" style={{ borderTop: '1px solid var(--hud-border)' }}>
        <span>Current Signal</span>
      </div>
      {bot.last_signal ? (
        <>
          <Kv
            label="Action"
            value={bot.last_signal.action}
            color={
              bot.last_signal.action === 'LONG' ? THEME.profit
                : bot.last_signal.action === 'SHORT' ? THEME.loss
                  : bot.last_signal.action === 'CLOSE' ? THEME.warning : undefined
            }
          />
          <Kv label="Confidence" value={plainPct(bot.last_signal.confidence * 100, 0)} />
          <div className="kv" style={{ gridTemplateColumns: '1fr' }}>
            <span className="kv__key">Reason</span>
            <span style={{ fontSize: 11, lineHeight: 1.5, color: 'var(--hud-text)' }}>
              {bot.last_signal.reason}
            </span>
          </div>
        </>
      ) : (
        <div className="empty" style={{ padding: 18 }}>NO SIGNAL YET</div>
      )}

      <div className="panel__header" style={{ borderTop: '1px solid var(--hud-border)' }}>
        <span>Position</span>
      </div>
      {botPositions.length > 0 ? (
        botPositions.map((position) => (
          <div key={position.id}>
            <Kv
              label="Side"
              value={`${position.side} ${quantity(position.quantity)}`}
              color={position.side === 'LONG' ? THEME.profit : THEME.loss}
            />
            <Kv label="Entry" value={price(position.entry_price)} />
            <Kv label="Mark" value={price(ticker?.price ?? position.mark_price)} />
            <Kv label="Stop Loss" value={position.stop_loss ? price(position.stop_loss) : '—'} />
            <Kv label="Take Profit" value={position.take_profit ? price(position.take_profit) : '—'} />
            <Kv
              label="Unrealized"
              value={`${signedMoney(position.unrealized_pnl)} (${pct(position.unrealized_pnl_pct, 1)})`}
              color={position.unrealized_pnl >= 0 ? THEME.profit : THEME.loss}
            />
          </div>
        ))
      ) : (
        <div className="empty" style={{ padding: 18 }}>FLAT</div>
      )}

      {botTrades.length > 0 && (
        <>
          <div className="panel__header" style={{ borderTop: '1px solid var(--hud-border)' }}>
            <span>Recent Activity</span>
          </div>
          {botTrades.map((trade) => (
            <Kv
              key={trade.id}
              label={`${trade.side} · ${trade.reason.replace(/_/g, ' ')}`}
              value={signedMoney(trade.realized_pnl)}
              color={trade.realized_pnl >= 0 ? THEME.profit : THEME.loss}
            />
          ))}
        </>
      )}

      <div className="controls">
        <button className="button button--go" disabled={busy || !offline} onClick={() => send('start')}>
          START
        </button>
        <button className="button" disabled={busy || offline || bot.status === 'PAUSED'} onClick={() => send('pause')}>
          PAUSE
        </button>
        <button className="button" disabled={busy || bot.status !== 'PAUSED'} onClick={() => send('resume')}>
          RESUME
        </button>
        <button className="button button--danger" disabled={busy || offline} onClick={() => send('stop')}>
          STOP
        </button>
      </div>
    </div>
  )
}

function Kv({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="kv">
      <span className="kv__key">{label}</span>
      <span className="kv__value" style={color ? { color } : undefined}>{value}</span>
    </div>
  )
}
