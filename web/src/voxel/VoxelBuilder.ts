/**
 * Voxel → geometry compiler.
 *
 * Models are authored as plain data (integer grid coordinates plus a palette
 * index). This compiles a model into a **single** BufferGeometry with baked
 * vertex colours, emitting only faces that touch air. A character built from
 * ~1200 voxels becomes one draw call instead of 1200 meshes, which is the
 * difference between a world that runs and a slideshow.
 */

import * as THREE from 'three'

/** `[x, y, z, paletteIndex]` on an integer grid. */
export type Voxel = [number, number, number, number]

export interface VoxelPart {
  name: string
  voxels: Voxel[]
  /** Rotation origin in grid units — a shoulder, a hip, the base of a neck. */
  pivot: [number, number, number]
}

export interface VoxelModel {
  name: string
  palette: string[]
  parts: VoxelPart[]
}

/** Face directions: normal, and the four corner offsets that wind it correctly. */
const FACES: Array<{
  dir: [number, number, number]
  corners: Array<[number, number, number]>
}> = [
  // +X
  { dir: [1, 0, 0], corners: [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]] },
  // -X
  { dir: [-1, 0, 0], corners: [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]] },
  // +Y
  { dir: [0, 1, 0], corners: [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]] },
  // -Y
  { dir: [0, -1, 0], corners: [[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]] },
  // +Z
  { dir: [0, 0, 1], corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] },
  // -Z
  { dir: [0, 0, -1], corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]] },
]

const key = (x: number, y: number, z: number): number =>
  // Grid coordinates stay well inside ±512, so pack them into one integer.
  ((x + 512) << 20) | ((y + 512) << 10) | (z + 512)

const colorCache = new Map<string, THREE.Color>()

function paletteColor(palette: string[], index: number): THREE.Color {
  const hex = palette[index] ?? palette[0] ?? '#ffffff'
  let color = colorCache.get(hex)
  if (!color) {
    color = new THREE.Color(hex).convertSRGBToLinear()
    colorCache.set(hex, color)
  }
  return color
}

export interface BuildOptions {
  /** World size of one voxel. */
  scale?: number
  /** Grid offset applied before scaling — used to centre a part on its pivot. */
  origin?: [number, number, number]
  /** Slightly darkens downward faces so forms read without heavy lighting. */
  shade?: boolean
}

/**
 * Compile voxels into one geometry, skipping faces hidden by a neighbour.
 */
export function buildVoxelGeometry(
  voxels: Voxel[],
  palette: string[],
  options: BuildOptions = {},
): THREE.BufferGeometry {
  const scale = options.scale ?? 1
  const [ox, oy, oz] = options.origin ?? [0, 0, 0]
  const shade = options.shade ?? true

  const occupied = new Set<number>()
  for (const [x, y, z] of voxels) occupied.add(key(x, y, z))

  // Count visible faces first so the typed arrays are allocated exactly once.
  let faceCount = 0
  for (const [x, y, z] of voxels) {
    for (const face of FACES) {
      if (!occupied.has(key(x + face.dir[0], y + face.dir[1], z + face.dir[2]))) {
        faceCount += 1
      }
    }
  }

  const vertexCount = faceCount * 6
  const positions = new Float32Array(vertexCount * 3)
  const normals = new Float32Array(vertexCount * 3)
  const colors = new Float32Array(vertexCount * 3)

  let p = 0
  let n = 0
  let c = 0

  for (const [x, y, z, colorIndex] of voxels) {
    const color = paletteColor(palette, colorIndex)

    for (const face of FACES) {
      const [dx, dy, dz] = face.dir
      if (occupied.has(key(x + dx, y + dy, z + dz))) continue

      // Ambient-style tint: top faces brightest, underside darkest.
      let tint = 1
      if (shade) {
        if (dy > 0) tint = 1.12
        else if (dy < 0) tint = 0.62
        else if (dx !== 0) tint = 0.86
        else tint = 0.94
      }

      const [a, b, cc, d] = face.corners
      const quad = [a, b, cc, a, cc, d]

      for (const corner of quad) {
        positions[p++] = (x + corner[0] - ox) * scale
        positions[p++] = (y + corner[1] - oy) * scale
        positions[p++] = (z + corner[2] - oz) * scale

        normals[n++] = dx
        normals[n++] = dy
        normals[n++] = dz

        colors[c++] = Math.min(1, color.r * tint)
        colors[c++] = Math.min(1, color.g * tint)
        colors[c++] = Math.min(1, color.b * tint)
      }
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.computeBoundingSphere()
  return geometry
}

// --------------------------------------------------------------------------
// Authoring primitives
// --------------------------------------------------------------------------

/** Emit a solid box of voxels. The workhorse of every model in the world. */
export function box(
  x: number, y: number, z: number,
  w: number, h: number, d: number,
  color: number,
): Voxel[] {
  const out: Voxel[] = []
  for (let ix = 0; ix < w; ix++) {
    for (let iy = 0; iy < h; iy++) {
      for (let iz = 0; iz < d; iz++) {
        out.push([x + ix, y + iy, z + iz, color])
      }
    }
  }
  return out
}

/** A hollow box — walls only, so interiors stay empty and cheap. */
export function shell(
  x: number, y: number, z: number,
  w: number, h: number, d: number,
  color: number,
  options: { floor?: boolean; roof?: boolean } = {},
): Voxel[] {
  const out: Voxel[] = []
  for (let ix = 0; ix < w; ix++) {
    for (let iy = 0; iy < h; iy++) {
      for (let iz = 0; iz < d; iz++) {
        const onWall = ix === 0 || ix === w - 1 || iz === 0 || iz === d - 1
        const onFloor = iy === 0 && options.floor !== false
        const onRoof = iy === h - 1 && options.roof !== false
        if (onWall || onFloor || onRoof) out.push([x + ix, y + iy, z + iz, color])
      }
    }
  }
  return out
}

/** A single voxel. */
export function vox(x: number, y: number, z: number, color: number): Voxel {
  return [x, y, z, color]
}

/** Mirror a set of voxels across the X axis about `axis` — for symmetric limbs. */
export function mirrorX(voxels: Voxel[], axis = 0): Voxel[] {
  return voxels.map(([x, y, z, c]) => [axis - x, y, z, c] as Voxel)
}

export function translate(voxels: Voxel[], dx: number, dy: number, dz: number): Voxel[] {
  return voxels.map(([x, y, z, c]) => [x + dx, y + dy, z + dz, c] as Voxel)
}

export function recolor(voxels: Voxel[], color: number): Voxel[] {
  return voxels.map(([x, y, z]) => [x, y, z, color] as Voxel)
}

/** Concatenate voxel groups. Later entries win where they overlap. */
export function merge(...groups: Voxel[][]): Voxel[] {
  const seen = new Map<number, Voxel>()
  for (const group of groups) {
    for (const voxel of group) {
      seen.set(key(voxel[0], voxel[1], voxel[2]), voxel)
    }
  }
  return Array.from(seen.values())
}

/** Remove any voxel matching a predicate — for carving windows, eyes, gaps. */
export function carve(
  voxels: Voxel[],
  predicate: (x: number, y: number, z: number) => boolean,
): Voxel[] {
  return voxels.filter(([x, y, z]) => !predicate(x, y, z))
}

/** Total voxel count — used by the perf overlay. */
export function countVoxels(model: VoxelModel): number {
  return model.parts.reduce((sum, part) => sum + part.voxels.length, 0)
}
