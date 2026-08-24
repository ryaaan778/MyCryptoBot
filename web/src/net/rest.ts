/** REST client — cold reads, history, and the paths the 2D fallback uses. */

import type { Order, ResearchState, Snapshot, Trade, WorldEvent } from './protocol'

function base(): string {
  const override = new URLSearchParams(window.location.search).get('backend')
  return override ? override.replace(/\/$/, '') : ''
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${base()}${path}`, {
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${base()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

export interface HealthResponse {
  status: string
  server_time: number
  mode: string
  provider: string
  provider_degraded: boolean
  bots: number
}

export interface EquityPoint {
  ts: number
  equity: number
  realized_pnl: number
  unrealized_pnl: number
  exposure: number
  drawdown_pct: number
}

export interface AuditEntry {
  seq: number
  ts: number
  actor: string
  action: string
  target: string | null
  detail: unknown
}

export const api = {
  health: () => get<HealthResponse>('/api/health'),
  state: () => get<Snapshot>('/api/state'),
  trades: (limit = 100) => get<Trade[]>(`/api/trades?limit=${limit}`),
  orders: (limit = 100) => get<Order[]>(`/api/orders?limit=${limit}`),
  events: (limit = 200) => get<WorldEvent[]>(`/api/events?limit=${limit}`),
  equity: (limit = 400) => get<EquityPoint[]>(`/api/equity?limit=${limit}`),
  audit: (limit = 200) => get<AuditEntry[]>(`/api/audit?limit=${limit}`),
  research: () => get<ResearchState>('/api/research'),
  command: (botId: string, action: string) =>
    post<{ ok: boolean; message: string }>(`/api/bots/${botId}/command`, { action }),
  emergencyStop: () => post<{ ok: boolean }>('/api/system/emergency-stop'),
  resume: () => post<{ ok: boolean }>('/api/system/resume'),
}
