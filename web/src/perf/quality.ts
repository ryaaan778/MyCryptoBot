/**
 * Performance tiering and the WebGL capability check.
 *
 * Trading visibility is never sacrificed to graphics: if the frame rate can't
 * hold up we drop quality, and if it still can't we fall back to the 2D
 * dashboard entirely rather than showing a world that stutters over live data.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'

import { useWorldStore, type Quality } from '../state/store'

/** Below this for a sustained period, drop a quality tier. */
const DEGRADE_FPS = 26
/** Below this even at the lowest tier, give up on 3D and use the 2D dashboard. */
const BAIL_FPS = 12

/**
 * Sampling is on a *time* window, not a frame count.
 *
 * Counting frames would make the check slowest exactly when it matters: at
 * 1 FPS a 90-frame window takes 90 seconds to produce a single sample, so a
 * machine that cannot render the world at all would sit at one frame per second
 * for minutes before anything reacted. Two seconds of wall clock always yields
 * a verdict in two seconds.
 */
const SAMPLE_SECONDS = 2
const BAIL_STRIKES = 2

export function detectWebGL(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    const context =
      canvas.getContext('webgl2') ??
      canvas.getContext('webgl') ??
      canvas.getContext('experimental-webgl')
    return Boolean(context)
  } catch {
    return false
  }
}

/** `?mode=2d` forces the fallback; `?mode=3d` forces the world. */
export function forcedRenderMode(): '2d' | '3d' | null {
  if (typeof window === 'undefined') return null
  const mode = new URLSearchParams(window.location.search).get('mode')
  if (mode === '2d') return '2d'
  if (mode === '3d') return '3d'
  return null
}

const NEXT_TIER: Record<Quality, Quality | null> = {
  high: 'medium',
  medium: 'low',
  low: null,
}

/**
 * Watches frame timing and degrades when the world can't keep up.
 * Mounted inside the Canvas.
 */
export function useQualityMonitor(): void {
  const frames = useRef(0)
  const elapsed = useRef(0)
  const strikes = useRef(0)

  useFrame((_, delta) => {
    // An explicit `?mode=3d` means the user asked for the world and should keep
    // it — but still let quality drop, so their choice runs as well as it can.
    const forced = forcedRenderMode()
    if (forced === '2d') return

    frames.current += 1
    elapsed.current += delta
    if (elapsed.current < SAMPLE_SECONDS) return

    const fps = frames.current / elapsed.current
    frames.current = 0
    elapsed.current = 0

    const store = useWorldStore.getState()

    if (fps < BAIL_FPS) {
      strikes.current += 1
      const atFloor = store.quality === 'low'
      if (atFloor && strikes.current >= BAIL_STRIKES && forced !== '3d') {
        // The world is unusable on this machine. Show the data instead —
        // trading visibility is never traded for graphics.
        store.setRenderMode('2d')
        return
      }
      // Still above the floor: drop a tier immediately rather than waiting.
      const next = NEXT_TIER[store.quality]
      if (next) store.setQuality(next)
      return
    }

    if (fps < DEGRADE_FPS) {
      const next = NEXT_TIER[store.quality]
      if (next) {
        store.setQuality(next)
        strikes.current = 0
      }
      return
    }

    strikes.current = 0
  })
}

/** A one-shot guess at a starting tier from what the device advertises. */
export function initialQuality(): Quality {
  if (typeof navigator === 'undefined') return 'high'
  const cores = navigator.hardwareConcurrency ?? 4
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8
  const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
  if (mobile || cores <= 2 || memory <= 2) return 'low'
  if (cores <= 4 || memory <= 4) return 'medium'
  return 'high'
}
