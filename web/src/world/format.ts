/** Number formatting shared by the world panels and the HUD. */

import { THEME } from './theme'

export function money(value: number, digits = 2): string {
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 10_000) return `$${(value / 1000).toFixed(1)}K`
  return `$${value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`
}

export function signedMoney(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : '-'}${money(Math.abs(value), digits)}`
}

export function pct(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}%`
}

export function plainPct(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`
}

export function price(value: number): string {
  if (value >= 1000) {
    return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }
  if (value >= 1) return value.toFixed(3)
  return value.toFixed(5)
}

export function quantity(value: number): string {
  if (value >= 1000) return value.toFixed(1)
  if (value >= 1) return value.toFixed(3)
  return value.toFixed(5)
}

export function pnlColor(value: number): string {
  if (value > 0) return THEME.profit
  if (value < 0) return THEME.loss
  return '#cdd5e6'
}

export function displaySymbol(symbol: string): string {
  return symbol.replace('/', '')
}

export function timeOfDay(ts: number): string {
  return new Date(ts).toLocaleTimeString('en-GB', { hour12: false })
}

export function relativeTime(ts: number): string {
  const seconds = Math.max(0, Math.round((Date.now() - ts) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  return `${Math.floor(seconds / 3600)}h ago`
}
