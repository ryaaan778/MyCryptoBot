/**
 * Live world state.
 *
 * Trading data lands here directly from the socket, never gated behind an
 * animation frame. The 3D scene reads most of it through `getState()` inside
 * `useFrame` rather than by subscribing, so a 4 Hz P&L update animates the
 * world without triggering a React re-render of the whole scene graph.
 */

import { create } from 'zustand'
import { useShallow } from 'zustand/react/shallow'
import type {
  BotState,
  Candle,
  Frame,
  MarketTickData,
  Order,
  PnlTickData,
  Position,
  PortfolioState,
  PositionClosedData,
  RiskState,
  Signal,
  Snapshot,
  SystemState,
  Ticker,
  Trade,
  WorldEvent,
  ResearchState,
} from '../net/protocol'
import { pushEffect } from '../fx/effectQueue'

export type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'offline'
export type RenderMode = '3d' | '2d'
export type Quality = 'high' | 'medium' | 'low'

const MAX_EVENTS = 240
const MAX_TRADES = 120
const MAX_ORDERS = 120
const MAX_CANDLES = 220

export interface WorldStore {
  connection: ConnectionState
  system: SystemState | null
  portfolio: PortfolioState | null
  risk: RiskState | null
  bots: Record<string, BotState>
  botOrder: string[]
  positions: Record<string, Position>
  orders: Order[]
  trades: Trade[]
  events: WorldEvent[]
  tickers: Record<string, Ticker>
  candles: Record<string, Candle[]>
  research: ResearchState | null

  selectedBot: string | null
  cameraTarget: string
  showTables: boolean
  showHelp: boolean
  showResearch: boolean
  renderMode: RenderMode
  quality: Quality
  lastError: string | null

  applySnapshot: (snapshot: Snapshot) => void
  applyFrame: (frame: Frame) => void
  setConnection: (state: ConnectionState) => void
  selectBot: (botId: string | null) => void
  focusCamera: (target: string) => void
  toggleTables: () => void
  toggleHelp: () => void
  toggleResearch: () => void
  setResearch: (research: ResearchState) => void
  setRenderMode: (mode: RenderMode) => void
  setQuality: (quality: Quality) => void
}

function keyFor(symbol: string, timeframe: string): string {
  return `${symbol}|${timeframe}`
}

export const candleKey = keyFor

export const useWorldStore = create<WorldStore>((set, get) => ({
  connection: 'connecting',
  system: null,
  portfolio: null,
  risk: null,
  bots: {},
  botOrder: [],
  positions: {},
  orders: [],
  trades: [],
  events: [],
  tickers: {},
  candles: {},
  research: null,

  selectedBot: null,
  cameraTarget: 'WORLD',
  showTables: false,
  showResearch: false,
  showHelp: false,
  renderMode: '3d',
  quality: 'high',
  lastError: null,

  // ---------------------------------------------------------------- snapshot

  applySnapshot: (snapshot) => {
    const bots: Record<string, BotState> = {}
    for (const bot of snapshot.bots) bots[bot.id] = bot

    const positions: Record<string, Position> = {}
    for (const position of snapshot.positions) positions[position.id] = position

    set({
      system: snapshot.system,
      portfolio: snapshot.portfolio,
      risk: snapshot.risk,
      bots,
      botOrder: snapshot.bots.map((b) => b.id),
      positions,
      orders: snapshot.orders.slice(-MAX_ORDERS),
      trades: snapshot.trades.slice(-MAX_TRADES),
      events: snapshot.events.slice(-MAX_EVENTS),
      tickers: snapshot.tickers,
      candles: snapshot.candles,
    })
  },

  // ------------------------------------------------------------------ deltas

  applyFrame: (frame) => {
    const { type, data } = frame

    switch (type) {
      case 'snapshot':
        get().applySnapshot(data as Snapshot)
        return

      case 'bot.status': {
        const payload = data as { bot_id: string; status: BotState['status'] }
        const bot = get().bots[payload.bot_id]
        if (!bot) return
        set((s) => ({
          bots: { ...s.bots, [payload.bot_id]: { ...bot, status: payload.status } },
        }))
        return
      }

      case 'bot.signal': {
        const signal = data as Signal
        const bot = get().bots[signal.bot_id]
        if (!bot) return
        set((s) => ({
          bots: { ...s.bots, [signal.bot_id]: { ...bot, last_signal: signal } },
        }))
        pushEffect({ kind: 'signal', botId: signal.bot_id, action: signal.action })
        return
      }

      case 'bot.heartbeat': {
        const payload = data as { bot_id: string; ts: number }
        const bot = get().bots[payload.bot_id]
        if (!bot) return
        set((s) => ({
          bots: {
            ...s.bots,
            [payload.bot_id]: { ...bot, last_heartbeat: payload.ts },
          },
        }))
        return
      }

      case 'order.submitted': {
        const order = data as Order
        pushEffect({ kind: 'order', botId: order.bot_id, side: order.side })
        set((s) => ({ orders: [...s.orders, order].slice(-MAX_ORDERS) }))
        return
      }

      case 'order.filled': {
        const fill = data as { bot_id?: string; reduce_only?: boolean }
        if (fill.bot_id) {
          pushEffect({
            kind: 'fill',
            botId: fill.bot_id,
            closing: Boolean(fill.reduce_only),
          })
        }
        return
      }

      case 'position.opened': {
        const position = data as Position
        set((s) => ({ positions: { ...s.positions, [position.id]: position } }))
        return
      }

      case 'position.closed': {
        const payload = data as PositionClosedData
        set((s) => {
          const next = { ...s.positions }
          delete next[payload.position_id]
          return { positions: next }
        })
        return
      }

      case 'trade.closed': {
        const trade = data as Trade
        pushEffect({
          kind: trade.realized_pnl >= 0 ? 'profit' : 'loss',
          botId: trade.bot_id,
          magnitude: Math.abs(trade.realized_pnl_pct),
          reason: trade.reason,
        })
        set((s) => ({ trades: [...s.trades, trade].slice(-MAX_TRADES) }))
        return
      }

      case 'pnl.tick': {
        const payload = data as PnlTickData
        const positions: Record<string, Position> = {}
        for (const position of payload.positions) positions[position.id] = position

        set((s) => {
          const bots = { ...s.bots }
          for (const update of payload.bots) {
            const existing = bots[update.id]
            if (!existing) continue
            bots[update.id] = {
              ...existing,
              status: update.status,
              realized_pnl: update.realized_pnl,
              unrealized_pnl: update.unrealized_pnl,
              total_pnl: update.realized_pnl + update.unrealized_pnl,
              exposure: update.exposure,
              open_positions: update.open_positions,
              risk_level: update.risk_level,
              risk_utilization: update.risk_utilization,
            }
          }
          return {
            portfolio: payload.portfolio,
            risk: payload.risk,
            positions,
            bots,
          }
        })
        return
      }

      case 'risk.update': {
        const risk = data as RiskState
        const previous = get().risk
        if (risk.emergency_stop && !previous?.emergency_stop) {
          pushEffect({ kind: 'emergency' })
        }
        set({ risk })
        return
      }

      case 'market.tick': {
        const tick = data as MarketTickData
        set((s) => ({
          tickers: {
            ...s.tickers,
            [tick.symbol]: {
              symbol: tick.symbol,
              price: tick.price,
              bid: tick.bid,
              ask: tick.ask,
              volume_24h: s.tickers[tick.symbol]?.volume_24h ?? 0,
              change_24h_pct: tick.change_24h_pct,
              ts: tick.ts,
            },
          },
        }))
        return
      }

      case 'market.kline': {
        const payload = data as {
          symbol: string
          timeframe: string
          candle: Candle
        }
        const key = keyFor(payload.symbol, payload.timeframe)
        set((s) => {
          const series = s.candles[key] ?? []
          return {
            candles: {
              ...s.candles,
              [key]: [...series, payload.candle].slice(-MAX_CANDLES),
            },
          }
        })
        return
      }

      case 'event.log': {
        const event = data as WorldEvent
        set((s) => ({ events: [...s.events, event].slice(-MAX_EVENTS) }))
        return
      }

      case 'system.emergency_stop': {
        const payload = data as { active: boolean }
        if (payload.active) pushEffect({ kind: 'emergency' })
        set((s) => ({
          system: s.system ? { ...s.system, emergency_stop: payload.active } : s.system,
        }))
        return
      }

      case 'market.provider_degraded': {
        const payload = data as { note: string; degraded: boolean }
        set((s) => ({
          system: s.system
            ? { ...s.system, provider_degraded: payload.degraded, provider_note: payload.note }
            : s.system,
        }))
        return
      }

      case 'research': {
        // A slow side channel. It arrives once on connect and on request, and
        // must never interfere with the lifecycle frames above.
        set({ research: data as ResearchState })
        break
      }

      case 'error': {
        set({ lastError: (data as { message: string }).message })
        return
      }

      default:
        return
    }
  },

  // ------------------------------------------------------------------ ui

  setConnection: (connection) => set({ connection }),
  selectBot: (selectedBot) => set({ selectedBot }),
  focusCamera: (cameraTarget) => set({ cameraTarget }),
  toggleTables: () => set((s) => ({ showTables: !s.showTables })),
  toggleHelp: () => set((s) => ({ showHelp: !s.showHelp })),
  toggleResearch: () => set((s) => ({ showResearch: !s.showResearch })),
  setResearch: (research) => set({ research }),
  setRenderMode: (renderMode) => set({ renderMode }),
  setQuality: (quality) => set({ quality }),
}))

// ---------------------------------------------------------------- selectors

/**
 * Selectors that build a new array must be compared shallowly.
 *
 * zustand v5 sits on `useSyncExternalStore`, which compares snapshots by
 * identity: a selector returning a fresh `Object.values(...)` on every call
 * looks like a new snapshot each render and spins into an infinite update loop.
 * `useShallow` compares element-by-element, so an unchanged set of the same
 * objects settles.
 */
export const useBots = (): BotState[] =>
  useWorldStore(useShallow((s) => s.botOrder.map((id) => s.bots[id]).filter(Boolean)))

export const usePositions = (): Position[] =>
  useWorldStore(useShallow((s) => Object.values(s.positions)))

export const useBotPositions = (botId: string): Position[] =>
  useWorldStore(
    useShallow((s) => Object.values(s.positions).filter((p) => p.bot_id === botId)),
  )

/** Plain selectors — these return stable references or primitives. */
export const selectSelectedBot = (s: WorldStore): BotState | null =>
  s.selectedBot ? s.bots[s.selectedBot] ?? null : null

export const selectIsEmergency = (s: WorldStore): boolean =>
  Boolean(s.risk?.emergency_stop || s.system?.emergency_stop)
