/**
 * World palette and layout constants.
 *
 * Colours are deliberately saturated and high-contrast — the world should look
 * theatrical, not like a muted corporate dashboard. Readability of the *data*
 * is preserved by the HUD, which uses its own restrained palette.
 */

import type { RiskLevel } from '../net/protocol'

export const THEME = {
  grass: '#3f7d4a',
  grassDeep: '#2c5c37',
  grassLight: '#579c5c',
  stone: '#8a8f9c',
  stoneDark: '#5b6070',
  stoneLight: '#b4b9c6',
  road: '#6d6a63',
  roadEdge: '#4d4a45',
  water: '#2f7fb8',
  waterDeep: '#1d5788',
  wood: '#7a4a28',
  woodDark: '#4e2f1a',
  sakura: '#f7a8d0',
  sakuraDeep: '#e07ab0',
  lantern: '#f0c04a',
  gold: '#f0c04a',
  royal: '#8b2f8f',
  night: '#1a1026',
  hologram: '#4fd6ff',
  // Research states get their own hue family — violet, unused elsewhere — so a
  // bot that is off the desk working on a hypothesis reads at a glance as
  // something other than idle, paused, or trading.
  research: '#a17bff',
  researchDeep: '#6f4fd8',
  profit: '#3ddc84',
  loss: '#ff5468',
  warning: '#ffb020',
  critical: '#ff2d55',
  white: '#f2f2f2',
} as const

export const RISK_COLOR: Record<RiskLevel, string> = {
  SAFE: THEME.profit,
  ELEVATED: THEME.warning,
  HIGH: '#ff7a3d',
  CRITICAL: THEME.critical,
}

/** World-space radius at which the bot districts sit around the headquarters. */
export const DISTRICT_RADIUS = 46

export const LAYOUT = {
  hq: [0, 0] as [number, number],
  marketRoom: [0, 40] as [number, number],
  tradingFloor: [0, 20] as [number, number],
  riskCenter: [-22, 12] as [number, number],
  // Large enough that the terrain edge sits outside the default view.
  groundSize: 230,
} as const

/** Camera presets — the named viewpoints in the HUD's camera bar. */
export interface CameraPreset {
  id: string
  label: string
  position: [number, number, number]
  target: [number, number, number]
}

// Each district preset sits *in front of* its character — derived from the
// district's position and its facing angle — so a bot is framed head-on with
// its data boards inside the frame, not clipped at the edges.
export const CAMERA_PRESETS: CameraPreset[] = [
  { id: 'WORLD', label: 'WORLD', position: [0, 78, 118], target: [0, 8, -4] },
  { id: 'JOJO', label: 'JOJO', position: [0, 22, 46], target: [0, 10, 0] },
  { id: 'jonathan', label: 'JONATHAN', position: [0, 15, -18], target: [0, 6, -44] },
  { id: 'joseph', label: 'JOSEPH', position: [-21, 15, -8], target: [-44, 6, -22] },
  { id: 'jotaro', label: 'JOTARO', position: [21, 15, -8], target: [44, 6, -22] },
  { id: 'jolyan', label: 'JOLYAN', position: [-21, 15, 13], target: [-44, 6, 26] },
  { id: 'kira', label: 'KIRA', position: [21, 15, 13], target: [44, 6, 26] },
  { id: 'MARKET', label: 'MARKET', position: [0, 18, 72], target: [0, 13, 40] },
  { id: 'FLOOR', label: 'TRADING FLOOR', position: [0, 16, 44], target: [0, 5, 20] },
  { id: 'RISK', label: 'RISK CENTER', position: [-30, 14, 34], target: [-16, 8, 10] },
]

export function presetFor(id: string): CameraPreset {
  return CAMERA_PRESETS.find((p) => p.id === id) ?? CAMERA_PRESETS[0]
}

/** Deterministic pseudo-random in [0,1) — keeps scatter stable across renders. */
export function hashRandom(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}
