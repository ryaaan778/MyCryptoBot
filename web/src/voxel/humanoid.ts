/**
 * Shared humanoid body plan.
 *
 * Every character in the world is built from this skeleton and then diverges:
 * proportions, silhouette-defining headwear, coats, capes and accessories are
 * per-character. The parts are kept separate (rather than merged into one blob)
 * because each part is a rig bone — poses are group transforms, which is how
 * the exaggerated posing stays cheap.
 */

import { box, carve, merge, mirrorX, type Voxel, type VoxelModel, type VoxelPart } from './VoxelBuilder'

/** Palette slots. Every character palette follows this order. */
export const SKIN = 0
export const HAIR = 1
export const PRIMARY = 2
export const SECONDARY = 3
export const ACCENT = 4
export const DARK = 5
export const LIGHT = 6

export interface HumanoidSpec {
  name: string
  palette: string[]
  /** Half-width of the torso in voxels. Broader = heavier silhouette. */
  shoulder?: number
  torsoHeight?: number
  legHeight?: number
  headSize?: number
  /** Extra voxels for the character's defining headwear/hair. */
  hair?: (spec: Required<Pick<HumanoidSpec, 'headSize'>> & { headY: number }) => Voxel[]
  /** A coat, cape or skirt hanging from the torso. */
  coat?: (ctx: { shoulder: number; torsoY: number; torsoHeight: number }) => Voxel[]
  /** Anything held, worn at the neck, or floating — the character's signature. */
  accessory?: (ctx: { shoulder: number; torsoY: number; headY: number }) => Voxel[]
  /** Colour of the eyes; defaults to LIGHT. */
  eyeColor?: number
}

export interface BuiltCharacter extends VoxelModel {
  /** Height in voxels, so the world can place labels above the head. */
  height: number
  shoulder: number
}

export function buildHumanoid(spec: HumanoidSpec): BuiltCharacter {
  const shoulder = spec.shoulder ?? 4
  const torsoHeight = spec.torsoHeight ?? 10
  const legHeight = spec.legHeight ?? 10
  const headSize = spec.headSize ?? 6
  const eyeColor = spec.eyeColor ?? LIGHT

  const torsoY = legHeight
  const headY = torsoY + torsoHeight
  const depth = 4
  const halfDepth = -2

  // ---- legs (mirrored) ---------------------------------------------------
  const legWidth = Math.max(2, shoulder - 1)
  const rightLeg = merge(
    box(-shoulder, 1, halfDepth, legWidth, legHeight - 1, depth, PRIMARY),
    box(-shoulder, 0, halfDepth - 1, legWidth, 1, depth + 1, DARK),   // boot
  )
  const leftLeg = mirrorX(rightLeg, -1)

  // ---- torso -------------------------------------------------------------
  const torso = merge(
    box(-shoulder, torsoY, halfDepth, shoulder * 2, torsoHeight, depth, PRIMARY),
    // A contrasting collar reads as clothing rather than a painted box.
    box(-shoulder, torsoY + torsoHeight - 2, halfDepth, shoulder * 2, 2, depth, SECONDARY),
    box(-1, torsoY + torsoHeight - 4, halfDepth - 1, 2, 4, 1, ACCENT),  // chest emblem
  )

  // ---- arms (mirrored) ---------------------------------------------------
  const armLength = torsoHeight - 1
  const rightArm = merge(
    box(-shoulder - 3, torsoY + torsoHeight - armLength, halfDepth, 3, armLength, depth, PRIMARY),
    box(-shoulder - 3, torsoY + torsoHeight - armLength, halfDepth, 3, 2, depth, SECONDARY),  // cuff
    box(-shoulder - 3, torsoY + torsoHeight - armLength - 2, halfDepth, 3, 2, depth, SKIN),   // hand
  )
  const leftArm = mirrorX(rightArm, -1)

  // ---- head --------------------------------------------------------------
  const halfHead = Math.floor(headSize / 2)
  let head = box(-halfHead, headY, halfDepth, headSize, headSize + 1, depth, SKIN)
  // Carve eye sockets on the front face, then fill them with the eye colour.
  const eyeY = headY + Math.floor(headSize / 2) + 1
  head = carve(head, (x, y, z) =>
    z === halfDepth && y === eyeY && (x === -halfHead + 1 || x === halfHead - 2))
  head = merge(
    head,
    box(-halfHead + 1, eyeY, halfDepth, 1, 1, 1, eyeColor),
    box(halfHead - 2, eyeY, halfDepth, 1, 1, 1, eyeColor),
  )

  const hair = spec.hair ? spec.hair({ headSize, headY }) : defaultHair(headSize, headY)
  const coat = spec.coat ? spec.coat({ shoulder, torsoY, torsoHeight }) : []
  const accessory = spec.accessory ? spec.accessory({ shoulder, torsoY, headY }) : []

  const parts: VoxelPart[] = [
    { name: 'legR', voxels: rightLeg, pivot: [0, legHeight, 0] },
    { name: 'legL', voxels: leftLeg, pivot: [0, legHeight, 0] },
    { name: 'torso', voxels: merge(torso, coat), pivot: [0, torsoY, 0] },
    { name: 'armR', voxels: rightArm, pivot: [-shoulder, torsoY + torsoHeight - 1, 0] },
    { name: 'armL', voxels: leftArm, pivot: [shoulder, torsoY + torsoHeight - 1, 0] },
    { name: 'head', voxels: merge(head, hair), pivot: [0, headY, 0] },
  ]
  if (accessory.length) {
    parts.push({ name: 'accessory', voxels: accessory, pivot: [0, torsoY, 0] })
  }

  const height = parts.reduce(
    (max, part) => part.voxels.reduce((m, [, y]) => Math.max(m, y + 1), max),
    0,
  )

  return { name: spec.name, palette: spec.palette, parts, height, shoulder }
}

function defaultHair(headSize: number, headY: number): Voxel[] {
  const half = Math.floor(headSize / 2)
  return merge(
    box(-half, headY + headSize - 1, -2, headSize, 2, 4, HAIR),
    box(-half, headY + headSize - 3, -2, headSize, 2, 1, HAIR),
    box(-half, headY + headSize - 4, 1, headSize, 3, 1, HAIR),
  )
}

// --------------------------------------------------------------------------
// Reusable silhouette pieces
// --------------------------------------------------------------------------

/** A long coat that flares below the torso — reads as authority at any zoom. */
export function longCoat(
  shoulder: number, torsoY: number, torsoHeight: number,
  color: number, trim: number, length = 8,
): Voxel[] {
  const out: Voxel[] = []
  for (let i = 0; i < length; i++) {
    const flare = Math.floor(i / 3)
    const width = shoulder * 2 + flare * 2
    out.push(...box(-shoulder - flare, torsoY - i, -3, width, 1, 6, color))
  }
  out.push(...box(-shoulder - Math.floor(length / 3), torsoY - length + 1, -3,
    shoulder * 2 + Math.floor(length / 3) * 2, 1, 6, trim))
  // Lapels, so the front reads as an open coat.
  out.push(...box(-shoulder, torsoY + torsoHeight - 6, -3, 2, 6, 1, trim))
  out.push(...box(shoulder - 2, torsoY + torsoHeight - 6, -3, 2, 6, 1, trim))
  return out
}

/** A cape hanging behind the shoulders. */
export function cape(
  shoulder: number, torsoY: number, torsoHeight: number,
  color: number, length = 14,
): Voxel[] {
  const out: Voxel[] = []
  for (let i = 0; i < length; i++) {
    const flare = Math.floor(i / 4)
    out.push(...box(
      -shoulder - flare, torsoY + torsoHeight - 2 - i, 2,
      shoulder * 2 + flare * 2, 1, 1, color,
    ))
  }
  return out
}

/** A peaked cap whose brim dominates the silhouette. */
export function peakedCap(
  headSize: number, headY: number, crown: number, brim: number,
): Voxel[] {
  const half = Math.floor(headSize / 2)
  const top = headY + headSize
  return merge(
    box(-half - 1, top - 1, -3, headSize + 2, 3, 6, crown),
    box(-half - 2, top - 2, -4, headSize + 4, 1, 3, brim),   // forward brim
    box(-half - 1, top + 2, -2, headSize + 2, 1, 4, crown),
  )
}

/** A scarf or muffler around the neck, trailing to one side. */
export function scarf(shoulder: number, neckY: number, color: number): Voxel[] {
  return merge(
    box(-shoulder, neckY, -3, shoulder * 2, 2, 6, color),
    box(shoulder - 1, neckY - 6, -3, 2, 6, 2, color),
    box(shoulder, neckY - 9, -3, 2, 3, 2, color),
  )
}

/** Flowing hair strands, for a lighter, less blocky silhouette. */
export function strands(
  headSize: number, headY: number, color: number, length = 12,
): Voxel[] {
  const half = Math.floor(headSize / 2)
  const out: Voxel[] = []
  for (let i = 0; i < length; i++) {
    const sway = Math.round(Math.sin(i * 0.55) * 1.5)
    out.push(...box(-half - 1 + sway, headY + headSize - 2 - i, 1, 2, 1, 2, color))
    out.push(...box(half - 1 + sway, headY + headSize - 2 - i, 1, 2, 1, 2, color))
  }
  return out
}
