/**
 * Wire types — the TypeScript mirror of backend/models.py.
 *
 * Field names here are the field names on the wire; if one changes on the
 * backend it must change here too.
 */

export type BotStatus =
  | 'OFFLINE' | 'IDLE' | 'ANALYZING' | 'TRADING' | 'PAUSED' | 'HALTED'
  // Research states: the bot is off the desk working on a hypothesis, which the
  // world shows as visibly different from being idle.
  | 'RESEARCHING' | 'TRAINING' | 'VALIDATING'

export type RiskLevel = 'SAFE' | 'ELEVATED' | 'HIGH' | 'CRITICAL'
export type PositionSide = 'LONG' | 'SHORT'
export type OrderSide = 'BUY' | 'SELL'
export type SignalAction = 'LONG' | 'SHORT' | 'CLOSE' | 'HOLD'
export type ExecutionMode = 'PAPER' | 'LIVE'
export type EventSeverity = 'info' | 'success' | 'warning' | 'danger' | 'critical'

export type CloseReason =
  | 'SIGNAL' | 'STOP_LOSS' | 'TAKE_PROFIT'
  | 'LIQUIDATION' | 'MANUAL' | 'EMERGENCY_STOP'

export interface Candle {
  ts: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Ticker {
  symbol: string
  price: number
  bid: number
  ask: number
  volume_24h: number
  change_24h_pct: number
  ts: number
}

export interface Signal {
  bot_id: string
  symbol: string
  action: SignalAction
  confidence: number
  reason: string
  strategy: string
  indicators: Record<string, number>
  ts: number
}

export interface Persona {
  title: string
  stand: string
  quote: string
  palette: string[]
  accent: string
  district: [number, number]
  facing: number
}

export interface BotState {
  id: string
  name: string
  strategy: string
  symbol: string
  timeframe: string
  status: BotStatus
  allocation: number
  equity: number
  realized_pnl: number
  unrealized_pnl: number
  total_pnl: number
  exposure: number
  leverage: number
  trades_total: number
  trades_won: number
  win_rate: number
  open_positions: number
  risk_level: RiskLevel
  risk_utilization: number
  last_signal: Signal | null
  last_heartbeat: number
  error: string | null
  persona: Persona
}

export interface Position {
  id: string
  bot_id: string
  symbol: string
  side: PositionSide
  quantity: number
  entry_price: number
  mark_price: number
  leverage: number
  stop_loss: number | null
  take_profit: number | null
  unrealized_pnl: number
  unrealized_pnl_pct: number
  margin: number
  fees_paid: number
  opened_at: number
  updated_at: number
}

export interface Order {
  id: string
  bot_id: string
  symbol: string
  side: OrderSide
  type: 'MARKET' | 'LIMIT'
  quantity: number
  price: number | null
  filled_quantity: number
  average_fill_price: number
  status: string
  fee: number
  reduce_only: boolean
  reason: string
  created_at: number
  updated_at: number
}

export interface Trade {
  id: string
  bot_id: string
  symbol: string
  side: PositionSide
  quantity: number
  entry_price: number
  exit_price: number
  leverage: number
  realized_pnl: number
  realized_pnl_pct: number
  fees: number
  reason: CloseReason
  opened_at: number
  closed_at: number
}

export interface WorldEvent {
  id: string
  type: string
  bot_id: string | null
  symbol: string | null
  message: string
  severity: EventSeverity
  data: Record<string, unknown>
  ts: number
}

export interface RiskState {
  level: RiskLevel
  total_exposure: number
  exposure_pct: number
  max_exposure_pct: number
  current_drawdown_pct: number
  max_drawdown_pct: number
  drawdown_limit_pct: number
  open_positions: number
  max_open_positions: number
  utilization: number
  emergency_stop: boolean
  halted_bots: string[]
  breaches: string[]
  ts: number
}

export interface PortfolioState {
  starting_equity: number
  equity: number
  cash: number
  realized_pnl: number
  unrealized_pnl: number
  total_pnl: number
  total_pnl_pct: number
  today_pnl: number
  today_pnl_pct: number
  peak_equity: number
  open_positions: number
  total_exposure: number
  active_bots: number
  total_bots: number
  trades_total: number
  win_rate: number
  ts: number
}

export interface SystemState {
  mode: ExecutionMode
  provider: string
  provider_degraded: boolean
  provider_note: string
  running: boolean
  emergency_stop: boolean
  started_at: number
  server_time: number
}

export interface Snapshot {
  system: SystemState
  portfolio: PortfolioState
  risk: RiskState
  bots: BotState[]
  positions: Position[]
  orders: Order[]
  trades: Trade[]
  events: WorldEvent[]
  tickers: Record<string, Ticker>
  candles: Record<string, Candle[]>
}

/** A frame as it arrives on the socket. */
/**
 * One agent's research standing.
 *
 * Everything here is *research* status. A promoted champion is a champion
 * inside the research system and nothing more — it has not been deployed, and
 * the live-trading gate is untouched by any of it.
 */
export interface AgentResearch {
  agent: string
  champion_policy: string | null
  champion_version: number | null
  challenger_count: number
  experiments_total: number
  experiments_rejected: number
  current_experiment: string | null
  current_status: string | null
  hypothesis: string | null
  paper_allocation: number
  bias_drift: number
  last_updated: number
}

/**
 * `synthetic_only` is load-bearing rather than decorative. Simulator results
 * must never be shown as evidence about live markets, and a display surface is
 * exactly where that mistake gets made — so the flag travels with the numbers
 * and the panel is expected to say so out loud.
 */
export interface ResearchState {
  available: boolean
  agents: AgentResearch[]
  datasets: number
  policies: number
  experiments: number
  sources: string[]
  synthetic_only: boolean
  paper_capital_deployed: number
  updated_at: number
}

export interface Frame<T = unknown> {
  type: string
  data: T
}

export interface OrderFilledData {
  order_id: string
  bot_id?: string
  symbol?: string
  side?: OrderSide
  price: number
  quantity: number
  fee: number
  ts?: number
  reduce_only?: boolean
}

export interface PositionClosedData {
  position_id: string
  bot_id: string
  symbol: string
  reason: CloseReason
  exit_price: number
  realized_pnl: number
}

export interface MarketTickData {
  symbol: string
  price: number
  bid: number
  ask: number
  spread_bps: number
  change_24h_pct: number
  ts: number
}

export interface PnlTickData {
  portfolio: PortfolioState
  risk: RiskState
  positions: Position[]
  bots: Array<{
    id: string
    status: BotStatus
    realized_pnl: number
    unrealized_pnl: number
    exposure: number
    open_positions: number
    risk_level: RiskLevel
    risk_utilization: number
  }>
  ts: number
}

export const SYMBOL_DISPLAY = (symbol: string): string => symbol.replace('/', '')
