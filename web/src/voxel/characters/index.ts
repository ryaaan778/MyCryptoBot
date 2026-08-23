/**
 * The roster, as voxel geometry.
 *
 * Each character has to be recognisable from across the world at a glance, so
 * the differences are structural — headwear, coat length, shoulder width, hair
 * mass — not just recoloured copies of one body.
 */

import { box, merge, type Voxel } from '../VoxelBuilder'
import {
  ACCENT,
  DARK,
  HAIR,
  LIGHT,
  PRIMARY,
  SECONDARY,
  SKIN,
  buildHumanoid,
  cape,
  longCoat,
  peakedCap,
  scarf,
  strands,
  type BuiltCharacter,
} from '../humanoid'

/**
 * Palettes follow the slot order in humanoid.ts:
 * [skin, hair, primary, secondary, accent, dark, light]
 */

// ---------------------------------------------------------------- JOJO (master)

const JOJO = buildHumanoid({
  name: 'JOJO',
  palette: ['#e8b98a', '#f0c04a', '#3a2352', '#8b2f8f', '#f0c04a', '#1a1026', '#ffffff'],
  shoulder: 5,
  torsoHeight: 12,
  legHeight: 12,
  headSize: 6,
  coat: ({ shoulder, torsoY, torsoHeight }) =>
    merge(
      cape(shoulder, torsoY, torsoHeight, SECONDARY, 18),
      longCoat(shoulder, torsoY, torsoHeight, PRIMARY, ACCENT, 10),
    ),
  hair: ({ headSize, headY }) => {
    const half = Math.floor(headSize / 2)
    const top = headY + headSize
    // A crown: the one silhouette that reads instantly as "in charge".
    const crown: Voxel[] = merge(
      box(-half - 1, top - 1, -3, headSize + 2, 2, 6, ACCENT),
      box(-half, top + 1, -2, headSize, 1, 4, ACCENT),
    )
    for (let i = 0; i < 5; i++) {
      crown.push(...box(-half + i * 2 - 1, top + 2, -2, 1, 2, 1, ACCENT))
      crown.push(...box(-half + i * 2 - 1, top + 2, 1, 1, 2, 1, ACCENT))
    }
    return merge(
      box(-half, headY + headSize - 2, -2, headSize, 2, 4, HAIR),
      box(-half, headY + headSize - 4, 1, headSize, 3, 1, HAIR),
      crown,
    )
  },
  accessory: ({ headY }) =>
    // Orbiting authority marks, floated above the head by the renderer.
    merge(
      box(-1, headY + 12, -1, 2, 2, 2, ACCENT),
      box(-7, headY + 10, -1, 2, 2, 2, SECONDARY),
      box(5, headY + 10, -1, 2, 2, 2, SECONDARY),
    ),
})

// ------------------------------------------------------------------ JONATHAN

const JONATHAN = buildHumanoid({
  name: 'JONATHAN',
  palette: ['#e8b98a', '#2f4f8f', '#2f5fd0', '#1a2f5e', '#e8c56a', '#12203f', '#f2e7cf'],
  shoulder: 6,          // the broadest silhouette on the roster
  torsoHeight: 12,
  legHeight: 11,
  headSize: 6,
  coat: ({ shoulder, torsoY, torsoHeight }) =>
    longCoat(shoulder, torsoY, torsoHeight, SECONDARY, ACCENT, 9),
  hair: ({ headSize, headY }) => {
    const half = Math.floor(headSize / 2)
    return merge(
      box(-half, headY + headSize - 1, -2, headSize, 3, 4, HAIR),
      box(-half - 1, headY + headSize + 1, -2, headSize + 2, 2, 4, HAIR),
      box(-half, headY + headSize - 3, 1, headSize, 3, 2, HAIR),
    )
  },
  accessory: ({ shoulder, torsoY }) =>
    // Shoulder guards — armour reads as the steady, defensive one.
    merge(
      box(-shoulder - 3, torsoY + 10, -3, 4, 3, 6, ACCENT),
      box(shoulder - 1, torsoY + 10, -3, 4, 3, 6, ACCENT),
    ),
})

// -------------------------------------------------------------------- JOSEPH

const JOSEPH = buildHumanoid({
  name: 'JOSEPH',
  palette: ['#e8b98a', '#7a4a28', '#3fa86b', '#1e5c3c', '#e08a3c', '#123825', '#f4ead6'],
  shoulder: 5,
  torsoHeight: 11,
  legHeight: 11,
  headSize: 6,
  coat: ({ shoulder, torsoY, torsoHeight }) =>
    longCoat(shoulder, torsoY, torsoHeight, PRIMARY, ACCENT, 6),
  hair: ({ headSize, headY }) => {
    const half = Math.floor(headSize / 2)
    const top = headY + headSize
    return merge(
      box(-half, top - 2, -2, headSize, 2, 4, HAIR),
      // Wide-brimmed hat: the cunning one, always shading his eyes.
      box(-half - 3, top, -5, headSize + 6, 1, 10, SECONDARY),
      box(-half - 1, top + 1, -3, headSize + 2, 3, 6, SECONDARY),
      box(-half - 1, top + 1, -3, headSize + 2, 1, 6, ACCENT),
    )
  },
  accessory: ({ shoulder, torsoY, headY }) =>
    merge(
      scarf(shoulder, headY - 2, ACCENT),
      // Trailing marks that read as a thrown object caught mid-air.
      box(shoulder + 4, torsoY + 6, -1, 2, 2, 2, ACCENT),
      box(shoulder + 7, torsoY + 8, -1, 2, 2, 2, ACCENT),
    ),
})

// -------------------------------------------------------------------- JOTARO

const JOTARO = buildHumanoid({
  name: 'JOTARO',
  palette: ['#e0b183', '#1b1d2e', '#2e3350', '#1b1d2e', '#c9a227', '#0d0f1a', '#e6e8f5'],
  shoulder: 6,
  torsoHeight: 12,
  legHeight: 12,
  headSize: 6,
  coat: ({ shoulder, torsoY, torsoHeight }) =>
    longCoat(shoulder, torsoY, torsoHeight, PRIMARY, ACCENT, 12),
  hair: ({ headSize, headY }) =>
    // The cap and hair merge into one mass — the signature unbroken silhouette.
    merge(
      box(-Math.floor(headSize / 2), headY + headSize - 2, -2, headSize, 2, 5, HAIR),
      peakedCap(headSize, headY, SECONDARY, ACCENT),
      box(-Math.floor(headSize / 2) - 1, headY + headSize - 4, 2, headSize + 2, 4, 2, HAIR),
    ),
  accessory: ({ shoulder, torsoY }) => {
    // A chain across the chest, and gold shoulder studs.
    const chain: Voxel[] = []
    for (let i = 0; i < 7; i++) {
      chain.push(...box(-shoulder + i + 2, torsoY + 11 - Math.abs(3 - i), -3, 1, 1, 1, ACCENT))
    }
    return merge(
      chain,
      box(-shoulder - 1, torsoY + 11, -3, 2, 2, 6, ACCENT),
      box(shoulder - 1, torsoY + 11, -3, 2, 2, 6, ACCENT),
    )
  },
})

// -------------------------------------------------------------------- JOLYAN

const JOLYAN = buildHumanoid({
  name: 'JOLYAN',
  palette: ['#e8b98a', '#3ec9a7', '#3ec9a7', '#1d7a66', '#ffd9ec', '#14544a', '#f2f2f2'],
  shoulder: 4,          // the slightest build — quick and unbound
  torsoHeight: 10,
  legHeight: 12,
  headSize: 6,
  eyeColor: ACCENT,
  hair: ({ headSize, headY }) => {
    const half = Math.floor(headSize / 2)
    const top = headY + headSize
    // Twin buns plus long strands: an unmistakable outline from any angle.
    return merge(
      box(-half, top - 2, -2, headSize, 2, 4, HAIR),
      box(-half - 2, top, -1, 3, 3, 3, HAIR),
      box(half - 1, top, -1, 3, 3, 3, HAIR),
      box(-half, top + 1, -2, headSize, 1, 4, HAIR),
      strands(headSize, headY, HAIR, 14),
    )
  },
  coat: ({ shoulder, torsoY }) => {
    // Web-like threads trailing from the hips.
    const threads: Voxel[] = []
    for (let i = 0; i < 10; i++) {
      const sway = Math.round(Math.sin(i * 0.7) * 2)
      threads.push(...box(-shoulder + sway, torsoY - i, 2, 1, 1, 1, LIGHT))
      threads.push(...box(shoulder - 1 + sway, torsoY - i, 2, 1, 1, 1, LIGHT))
    }
    return threads
  },
  accessory: ({ torsoY }) =>
    merge(
      box(-1, torsoY + 6, -3, 2, 2, 1, ACCENT),
      box(-4, torsoY + 3, -3, 1, 1, 1, ACCENT),
      box(3, torsoY + 3, -3, 1, 1, 1, ACCENT),
    ),
})

// ---------------------------------------------------------------------- KIRA

const KIRA = buildHumanoid({
  name: 'KIRA',
  palette: ['#e8c3a0', '#d9c48a', '#d94f9c', '#5a1f47', '#2fb8a8', '#3a1030', '#f0e6ef'],
  shoulder: 5,
  torsoHeight: 11,
  legHeight: 11,
  headSize: 6,
  eyeColor: ACCENT,
  coat: ({ shoulder, torsoY, torsoHeight }) => {
    // A neat suit, not a coat — the quiet one who wants to look ordinary.
    const suit = box(-shoulder, torsoY, -3, shoulder * 2, torsoHeight, 6, PRIMARY)
    const shirt = box(-1, torsoY + torsoHeight - 7, -4, 2, 7, 1, LIGHT)
    const tie = box(-1, torsoY + torsoHeight - 7, -4, 1, 6, 1, ACCENT)
    return merge(suit, shirt, tie)
  },
  hair: ({ headSize, headY }) => {
    const half = Math.floor(headSize / 2)
    return merge(
      box(-half, headY + headSize - 1, -2, headSize, 2, 4, HAIR),
      box(-half, headY + headSize - 3, -3, headSize, 2, 1, HAIR),  // neat side part
      box(-half, headY + headSize - 3, 1, headSize, 2, 2, HAIR),
    )
  },
  accessory: ({ shoulder, torsoY }) =>
    // A single raised hand — the character's whole identity, held very still.
    merge(
      box(shoulder + 2, torsoY + 12, -2, 3, 4, 3, SKIN),
      box(shoulder + 2, torsoY + 16, -2, 3, 2, 1, SKIN),
      box(shoulder + 5, torsoY + 14, -2, 1, 3, 3, SKIN),
    ),
})

export const CHARACTERS: Record<string, BuiltCharacter> = {
  jojo: JOJO,
  jonathan: JONATHAN,
  joseph: JOSEPH,
  jotaro: JOTARO,
  jolyan: JOLYAN,
  kira: KIRA,
}

export function characterFor(botId: string): BuiltCharacter {
  return CHARACTERS[botId] ?? CHARACTERS.jonathan
}

export { type BuiltCharacter }
export { ACCENT, DARK, HAIR, LIGHT, PRIMARY, SECONDARY, SKIN }
