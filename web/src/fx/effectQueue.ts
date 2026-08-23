/**
 * The effect budget.
 *
 * Visual effects are *requests*, not commitments. They land in a bounded ring
 * and the renderer drains as many as its per-frame budget allows; anything that
 * doesn't fit is dropped oldest-first. Nothing here can block or delay a store
 * update, which is what keeps trading data ahead of decoration under load.
 */

import type { CloseReason, OrderSide, SignalAction } from '../net/protocol'

export type EffectRequest =
  | { kind: 'signal'; botId: string; action: SignalAction }
  | { kind: 'order'; botId: string; side: OrderSide }
  | { kind: 'fill'; botId: string; closing: boolean }
  | { kind: 'profit'; botId: string; magnitude: number; reason: CloseReason }
  | { kind: 'loss'; botId: string; magnitude: number; reason: CloseReason }
  | { kind: 'emergency' }

const MAX_QUEUED = 48

/** Effects consumed per frame. Keeps a burst of fills from stalling a frame. */
export const FRAME_BUDGET = 4

const queue: EffectRequest[] = []
let droppedCount = 0

export function pushEffect(effect: EffectRequest): void {
  if (queue.length >= MAX_QUEUED) {
    queue.shift()
    droppedCount += 1
  }
  queue.push(effect)
}

/** Drain up to `budget` effects. Called once per rendered frame. */
export function drainEffects(budget: number = FRAME_BUDGET): EffectRequest[] {
  if (queue.length === 0) return []
  return queue.splice(0, budget)
}

export function queuedEffects(): number {
  return queue.length
}

export function droppedEffects(): number {
  return droppedCount
}

export function clearEffects(): void {
  queue.length = 0
}
