/**
 * The conventional detailed view.
 *
 * The world is the primary interface, but anyone actually running a desk needs
 * sortable rows of positions, orders, trades and events. This is that view, and
 * it is also what the 2D fallback reuses.
 */

import { useEffect, useMemo, useState } from 'react'

import { api, type AuditEntry } from '../../net/rest'
import type { Order, Trade } from '../../net/protocol'
import { usePositions, useWorldStore } from '../../state/store'
import { THEME } from '../../world/theme'
import {
  displaySymbol,
  money,
  pct,
  price,
  quantity,
  signedMoney,
  timeOfDay,
} from '../../world/format'

type Tab = 'positions' | 'orders' | 'trades' | 'events' | 'audit'

/**
 * Tables accept a row cap because the 2D fallback shows all of them stacked on
 * one page — and that page exists precisely for machines that are struggling.
 * Rendering several hundred uncapped rows there is the heaviest thing the app
 * does, on the device least able to take it.
 */
export interface TableProps {
  limit?: number
}

function Truncated({ shown, total }: { shown: number; total: number }) {
  if (shown >= total) return null
  return (
    <div className="card__sub" style={{ padding: '8px 10px' }}>
      Showing the {shown} most recent of {total}. Open DATA TABLES for the full history.
    </div>
  )
}

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'positions', label: 'POSITIONS' },
  { id: 'orders', label: 'ORDERS' },
  { id: 'trades', label: 'TRADES' },
  { id: 'events', label: 'EVENT LOG' },
  { id: 'audit', label: 'AUDIT TRAIL' },
]

export function DataTables() {
  const [tab, setTab] = useState<Tab>('positions')
  const toggleTables = useWorldStore((s) => s.toggleTables)

  return (
    <div className="tables panel">
      <div className="tables__tabs">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            className={`camerabar__button ${tab === entry.id ? 'camerabar__button--active' : ''}`}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
          </button>
        ))}
        <div className="spacer" style={{ background: 'var(--hud-bg)' }} />
        <button className="camerabar__button" onClick={toggleTables}>✕ CLOSE</button>
      </div>
      <div className="tables__body">
        {tab === 'positions' && <PositionsTable />}
        {tab === 'orders' && <OrdersTable />}
        {tab === 'trades' && <TradesTable />}
        {tab === 'events' && <EventsTable />}
        {tab === 'audit' && <AuditTable />}
      </div>
    </div>
  )
}

export function PositionsTable({ limit }: TableProps = {}) {
  const all = usePositions()
  const bots = useWorldStore((s) => s.bots)
  const tickers = useWorldStore((s) => s.tickers)

  if (all.length === 0) return <div className="empty">NO OPEN POSITIONS</div>
  const positions = limit ? all.slice(0, limit) : all

  return (
    <>
    <table>
      <thead>
        <tr>
          <th>Bot</th><th>Symbol</th><th>Side</th>
          <th className="num">Qty</th><th className="num">Entry</th><th className="num">Mark</th>
          <th className="num">Stop</th><th className="num">Target</th>
          <th className="num">Unrealized</th><th className="num">%</th>
          <th className="num">Lev</th><th className="num">Margin</th><th>Opened</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((position) => (
          <tr key={position.id}>
            <td>{bots[position.bot_id]?.name ?? position.bot_id}</td>
            <td>{displaySymbol(position.symbol)}</td>
            <td style={{ color: position.side === 'LONG' ? THEME.profit : THEME.loss }}>
              {position.side}
            </td>
            <td className="num">{quantity(position.quantity)}</td>
            <td className="num">{price(position.entry_price)}</td>
            <td className="num">{price(tickers[position.symbol]?.price ?? position.mark_price)}</td>
            <td className="num">{position.stop_loss ? price(position.stop_loss) : '—'}</td>
            <td className="num">{position.take_profit ? price(position.take_profit) : '—'}</td>
            <td
              className="num"
              style={{ color: position.unrealized_pnl >= 0 ? THEME.profit : THEME.loss }}
            >
              {signedMoney(position.unrealized_pnl)}
            </td>
            <td
              className="num"
              style={{ color: position.unrealized_pnl >= 0 ? THEME.profit : THEME.loss }}
            >
              {pct(position.unrealized_pnl_pct, 2)}
            </td>
            <td className="num">{position.leverage.toFixed(0)}x</td>
            <td className="num">{money(position.margin)}</td>
            <td>{timeOfDay(position.opened_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <Truncated shown={positions.length} total={all.length} />
    </>
  )
}

export function OrdersTable({ limit }: TableProps = {}) {
  const bots = useWorldStore((s) => s.bots)
  const [all, setAll] = useState<Order[]>([])

  useEffect(() => {
    let alive = true
    const load = () => {
      api.orders(limit ?? 150).then((rows) => { if (alive) setAll(rows) }).catch(() => undefined)
    }
    load()
    const timer = window.setInterval(load, limit ? 8000 : 4000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [limit])

  if (all.length === 0) return <div className="empty">NO ORDERS YET</div>
  const orders = limit ? all.slice(0, limit) : all

  return (
    <>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Bot</th><th>Symbol</th><th>Side</th><th>Type</th>
          <th className="num">Qty</th><th className="num">Filled</th>
          <th className="num">Avg Price</th><th className="num">Fee</th>
          <th>Status</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {orders.map((order) => (
          <tr key={order.id}>
            <td>{timeOfDay(order.created_at)}</td>
            <td>{bots[order.bot_id]?.name ?? order.bot_id}</td>
            <td>{displaySymbol(order.symbol)}</td>
            <td style={{ color: order.side === 'BUY' ? THEME.profit : THEME.loss }}>
              {order.side}
            </td>
            <td>{order.type}</td>
            <td className="num">{quantity(order.quantity)}</td>
            <td className="num">{quantity(order.filled_quantity)}</td>
            <td className="num">
              {order.average_fill_price ? price(order.average_fill_price) : '—'}
            </td>
            <td className="num">{order.fee ? order.fee.toFixed(4) : '—'}</td>
            <td
              style={{
                color: order.status === 'FILLED' ? THEME.profit
                  : order.status === 'REJECTED' ? THEME.loss : undefined,
              }}
            >
              {order.status}
            </td>
            <td style={{ maxWidth: 320, whiteSpace: 'normal' }}>{order.reason}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <Truncated shown={orders.length} total={all.length} />
    </>
  )
}

export function TradesTable({ limit }: TableProps = {}) {
  const bots = useWorldStore((s) => s.bots)
  const live = useWorldStore((s) => s.trades)
  const [history, setHistory] = useState<Trade[]>([])

  useEffect(() => {
    let alive = true
    api.trades(limit ?? 200).then((r) => { if (alive) setHistory(r) }).catch(() => undefined)
    return () => { alive = false }
  }, [live.length, limit])

  const all = history.length ? history : [...live].reverse()
  if (all.length === 0) return <div className="empty">NO CLOSED TRADES YET</div>
  const rows = limit ? all.slice(0, limit) : all

  return (
    <>
    <table>
      <thead>
        <tr>
          <th>Closed</th><th>Bot</th><th>Symbol</th><th>Side</th>
          <th className="num">Qty</th><th className="num">Entry</th><th className="num">Exit</th>
          <th className="num">P&amp;L</th><th className="num">%</th>
          <th className="num">Fees</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((trade) => (
          <tr key={trade.id}>
            <td>{timeOfDay(trade.closed_at)}</td>
            <td>{bots[trade.bot_id]?.name ?? trade.bot_id}</td>
            <td>{displaySymbol(trade.symbol)}</td>
            <td style={{ color: trade.side === 'LONG' ? THEME.profit : THEME.loss }}>
              {trade.side}
            </td>
            <td className="num">{quantity(trade.quantity)}</td>
            <td className="num">{price(trade.entry_price)}</td>
            <td className="num">{price(trade.exit_price)}</td>
            <td
              className="num"
              style={{ color: trade.realized_pnl >= 0 ? THEME.profit : THEME.loss }}
            >
              {signedMoney(trade.realized_pnl)}
            </td>
            <td
              className="num"
              style={{ color: trade.realized_pnl >= 0 ? THEME.profit : THEME.loss }}
            >
              {pct(trade.realized_pnl_pct, 2)}
            </td>
            <td className="num">{trade.fees.toFixed(4)}</td>
            <td>{trade.reason.replace(/_/g, ' ')}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <Truncated shown={rows.length} total={all.length} />
    </>
  )
}

export function EventsTable({ limit }: TableProps = {}) {
  const events = useWorldStore((s) => s.events)
  const all = useMemo(() => [...events].reverse(), [events])
  const rows = useMemo(() => (limit ? all.slice(0, limit) : all), [all, limit])
  const bots = useWorldStore((s) => s.bots)

  if (all.length === 0) return <div className="empty">NO EVENTS YET</div>

  const severityColor: Record<string, string> = {
    success: THEME.profit,
    warning: THEME.warning,
    danger: THEME.loss,
    critical: THEME.critical,
    info: THEME.hologram,
  }

  return (
    <>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Severity</th><th>Type</th><th>Bot</th><th>Symbol</th><th>Message</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((event) => (
          <tr key={event.id}>
            <td>{timeOfDay(event.ts)}</td>
            <td style={{ color: severityColor[event.severity] }}>
              {event.severity.toUpperCase()}
            </td>
            <td>{event.type}</td>
            <td>{event.bot_id ? bots[event.bot_id]?.name ?? event.bot_id : '—'}</td>
            <td>{event.symbol ? displaySymbol(event.symbol) : '—'}</td>
            <td style={{ whiteSpace: 'normal' }}>{event.message}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <Truncated shown={rows.length} total={all.length} />
    </>
  )
}

export function AuditTable() {
  const [rows, setRows] = useState<AuditEntry[]>([])

  useEffect(() => {
    let alive = true
    const load = () => {
      api.audit(200).then((data) => { if (alive) setRows(data) }).catch(() => undefined)
    }
    load()
    const timer = window.setInterval(load, 6000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])

  if (rows.length === 0) return <div className="empty">AUDIT TRAIL EMPTY</div>

  return (
    <table>
      <thead>
        <tr>
          <th className="num">#</th><th>Time</th><th>Actor</th>
          <th>Action</th><th>Target</th><th>Detail</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((entry) => (
          <tr key={entry.seq}>
            <td className="num">{entry.seq}</td>
            <td>{timeOfDay(entry.ts)}</td>
            <td>{entry.actor}</td>
            <td>{entry.action}</td>
            <td>{entry.target ?? '—'}</td>
            <td style={{ whiteSpace: 'normal', maxWidth: 460 }}>
              {entry.detail ? JSON.stringify(entry.detail) : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
