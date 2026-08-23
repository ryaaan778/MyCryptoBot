/**
 * Pose system.
 *
 * Bot status drives posture directly — a character that is ANALYZING studies
 * the air in front of it, one that is TRADING strikes, one that is OFFLINE
 * slumps. The world is readable without the HUD because of this mapping.
 *
 * Poses are pure functions of time, so nothing here allocates per frame.
 */

import type { BotStatus, RiskLevel } from '../net/protocol'

export type PartName = 'torso' | 'head' | 'armL' | 'armR' | 'legL' | 'legR' | 'accessory'

export interface Transform {
  rx: number
  ry: number
  rz: number
  ox: number
  oy: number
  oz: number
}

export type PoseFrame = Record<PartName, Transform>

export type PoseName =
  | 'idle' | 'analyzing' | 'trading' | 'victory' | 'danger' | 'offline'

const ZERO: Transform = { rx: 0, ry: 0, rz: 0, ox: 0, oy: 0, oz: 0 }

function t(
  rx = 0, ry = 0, rz = 0, ox = 0, oy = 0, oz = 0,
): Transform {
  return { rx, ry, rz, ox, oy, oz }
}

/** Reused across frames so posing never allocates. */
const scratch: PoseFrame = {
  torso: { ...ZERO }, head: { ...ZERO },
  armL: { ...ZERO }, armR: { ...ZERO },
  legL: { ...ZERO }, legR: { ...ZERO },
  accessory: { ...ZERO },
}

function assign(part: PartName, value: Transform): void {
  const target = scratch[part]
  target.rx = value.rx
  target.ry = value.ry
  target.rz = value.rz
  target.ox = value.ox
  target.oy = value.oy
  target.oz = value.oz
}

export function poseForStatus(status: BotStatus, risk: RiskLevel): PoseName {
  if (status === 'OFFLINE' || status === 'HALTED') return 'offline'
  if (risk === 'CRITICAL' || risk === 'HIGH') return 'danger'
  if (status === 'TRADING') return 'trading'
  if (status === 'ANALYZING') return 'analyzing'
  if (status === 'PAUSED') return 'idle'
  return 'idle'
}

/**
 * Evaluate a pose at time `time`, phase-shifted by `phase` so five characters
 * standing together never move in lockstep.
 */
export function evaluatePose(name: PoseName, time: number, phase = 0): PoseFrame {
  const clock = time + phase

  switch (name) {
    case 'analyzing': {
      // Reading holograms: head down and turning, one hand raised to the chin.
      const scan = Math.sin(clock * 0.9)
      assign('torso', t(0.04, scan * 0.18, 0, 0, Math.sin(clock * 1.6) * 0.03))
      assign('head', t(0.22, scan * 0.42, 0))
      assign('armR', t(-1.5, 0, 0.35))
      assign('armL', t(-0.25, 0, -0.12))
      assign('legL', ZERO)
      assign('legR', ZERO)
      assign('accessory', t(0, clock * 0.8, 0))
      return scratch
    }

    case 'trading': {
      // The strike: fast alternating thrusts with the torso twisting into them.
      const rush = Math.sin(clock * 9)
      const counter = Math.sin(clock * 9 + Math.PI)
      assign('torso', t(0.12, rush * 0.3, 0, 0, Math.abs(rush) * 0.12))
      assign('head', t(0.1, rush * 0.2, 0))
      assign('armR', t(-2.1 + rush * 0.9, 0, 0.2))
      assign('armL', t(-2.1 + counter * 0.9, 0, -0.2))
      assign('legR', t(rush * 0.25, 0, 0))
      assign('legL', t(counter * 0.25, 0, 0))
      assign('accessory', t(0, clock * 3, 0))
      return scratch
    }

    case 'victory': {
      // Arms flung wide, chest to the sky — extravagant on purpose.
      const swell = Math.sin(clock * 2.2)
      assign('torso', t(-0.2, 0, 0, 0, 0.35 + swell * 0.18))
      assign('head', t(-0.45, Math.sin(clock * 1.3) * 0.25, 0))
      assign('armR', t(-0.5, 0, 1.35 + swell * 0.2))
      assign('armL', t(-0.5, 0, -1.35 - swell * 0.2))
      assign('legR', t(0.1, 0, 0.12))
      assign('legL', t(0.1, 0, -0.12))
      assign('accessory', t(0, clock * 4, 0))
      return scratch
    }

    case 'danger': {
      // Braced and trembling: the posture of a bot near its limits.
      const shake = Math.sin(clock * 22) * 0.05
      const brace = Math.sin(clock * 3.4)
      assign('torso', t(0.24 + shake, shake * 2, 0, shake * 0.3, -0.15))
      assign('head', t(0.3, shake * 3, shake))
      assign('armR', t(-0.9, 0, 0.75 + brace * 0.12))
      assign('armL', t(-0.9, 0, -0.75 - brace * 0.12))
      assign('legR', t(-0.12, 0, 0.2))
      assign('legL', t(-0.12, 0, -0.2))
      assign('accessory', t(0, clock * 6, shake * 4))
      return scratch
    }

    case 'offline': {
      // Dead still. The only pose with no time term at all.
      assign('torso', t(0.35, 0, 0, 0, -0.55))
      assign('head', t(0.7, 0, 0.1))
      assign('armR', t(0.25, 0, 0.1))
      assign('armL', t(0.25, 0, -0.1))
      assign('legR', t(-0.15, 0, 0.05))
      assign('legL', t(-0.15, 0, -0.05))
      assign('accessory', t(0, 0, 0))
      return scratch
    }

    case 'idle':
    default: {
      // Breathing, weight shifting — alive but at rest.
      const breath = Math.sin(clock * 1.5)
      const sway = Math.sin(clock * 0.7)
      assign('torso', t(0, sway * 0.09, 0, 0, breath * 0.09))
      assign('head', t(breath * 0.05, sway * 0.28, 0))
      assign('armR', t(breath * 0.12, 0, 0.13 + sway * 0.05))
      assign('armL', t(-breath * 0.12, 0, -0.13 - sway * 0.05))
      assign('legR', ZERO)
      assign('legL', ZERO)
      assign('accessory', t(0, clock * 0.5, 0))
      return scratch
    }
  }
}

/** Blend factor for easing between poses when status changes. */
export const POSE_BLEND_RATE = 6.5
