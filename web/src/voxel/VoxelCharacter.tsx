/**
 * Renders one character: a group of merged part-meshes driven by the pose system.
 *
 * Geometry is compiled once and cached by character id. Posing writes directly
 * into object transforms inside `useFrame`, so an animated roster costs no React
 * renders at all.
 */

import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import type { BotStatus, RiskLevel } from '../net/protocol'
import { buildVoxelGeometry, type VoxelPart } from './VoxelBuilder'
import { characterFor, type BuiltCharacter } from './characters'
import {
  POSE_BLEND_RATE,
  evaluatePose,
  poseForStatus,
  type PartName,
  type PoseName,
} from './poses'

/**
 * World units per voxel.
 *
 * Sized so a ~36-voxel character stands about 6 units tall — roughly the height
 * of a district building's wall. The characters are the subject of this world,
 * so they have to hold their own against the architecture rather than read as
 * figurines on a plinth.
 */
export const VOXEL_SCALE = 0.17

const geometryCache = new Map<string, THREE.BufferGeometry[]>()

function geometriesFor(id: string, character: BuiltCharacter): THREE.BufferGeometry[] {
  const cached = geometryCache.get(id)
  if (cached) return cached
  const built = character.parts.map((part: VoxelPart) =>
    buildVoxelGeometry(part.voxels, character.palette, {
      scale: VOXEL_SCALE,
      origin: part.pivot,
    }),
  )
  geometryCache.set(id, built)
  return built
}

interface PartRef {
  group: THREE.Group
  name: PartName
  pivot: [number, number, number]
}

export interface VoxelCharacterProps {
  botId: string
  status: BotStatus
  risk: RiskLevel
  /** Emissive tint — profit warms the character, loss cools it. */
  glow?: THREE.ColorRepresentation
  glowStrength?: number
  /** Desaturates and dims an offline character. */
  dimmed?: boolean
  phase?: number
  onClick?: () => void
  onPointerOver?: () => void
  onPointerOut?: () => void
}

export function VoxelCharacter({
  botId,
  status,
  risk,
  glow,
  glowStrength = 0,
  dimmed = false,
  phase = 0,
  onClick,
  onPointerOver,
  onPointerOut,
}: VoxelCharacterProps) {
  const character = useMemo(() => characterFor(botId), [botId])
  const geometries = useMemo(() => geometriesFor(botId, character), [botId, character])

  const material = useMemo(
    () =>
      new THREE.MeshLambertMaterial({
        vertexColors: true,
        emissive: new THREE.Color(0x000000),
      }),
    [],
  )

  const partRefs = useRef<PartRef[]>([])
  const currentPose = useRef<PoseName>('idle')
  const blend = useRef(1)
  const previousPose = useRef<PoseName>('idle')

  // Track pose changes so the transition eases instead of snapping.
  const targetPose = poseForStatus(status, risk)
  useLayoutEffect(() => {
    if (targetPose !== currentPose.current) {
      previousPose.current = currentPose.current
      currentPose.current = targetPose
      blend.current = 0
    }
  }, [targetPose])

  useLayoutEffect(() => {
    material.emissive.set(glow ?? 0x000000)
    material.emissiveIntensity = glowStrength
    material.opacity = dimmed ? 0.72 : 1
    material.transparent = dimmed
    material.color.setScalar(dimmed ? 0.55 : 1)
  }, [material, glow, glowStrength, dimmed])

  useLayoutEffect(() => () => material.dispose(), [material])

  useFrame((_, delta) => {
    const time = performance.now() / 1000
    blend.current = Math.min(1, blend.current + delta * POSE_BLEND_RATE)
    const k = blend.current

    const from = evaluatePose(previousPose.current, time, phase)
    // evaluatePose reuses one scratch object, so the outgoing pose must be
    // copied out before the incoming pose overwrites it.
    const fromCopy = partRefs.current.map((ref) => ({ ...from[ref.name] }))
    const to = evaluatePose(currentPose.current, time, phase)

    partRefs.current.forEach((ref, index) => {
      const a = fromCopy[index]
      const b = to[ref.name]
      if (!a || !b) return
      const group = ref.group
      group.rotation.set(
        a.rx + (b.rx - a.rx) * k,
        a.ry + (b.ry - a.ry) * k,
        a.rz + (b.rz - a.rz) * k,
      )
      // `ref.pivot` is already in world units, and pose offsets are world-space
      // nudges — so these simply add. (Re-scaling the pivot here would square
      // VOXEL_SCALE and collapse every limb toward the origin.)
      const [px, py, pz] = ref.pivot
      group.position.set(
        px + a.ox + (b.ox - a.ox) * k,
        py + a.oy + (b.oy - a.oy) * k,
        pz + a.oz + (b.oz - a.oz) * k,
      )
    })
  })

  return (
    <group
      // Models are authored facing -Z (eyes, lapels, brims and chest details
      // all sit on the low-Z face), but the world places characters expecting
      // +Z to be forward. This turn reconciles the two.
      rotation={[0, Math.PI, 0]}
      onClick={onClick ? (event) => { event.stopPropagation(); onClick() } : undefined}
      onPointerOver={onPointerOver ? (event) => { event.stopPropagation(); onPointerOver() } : undefined}
      onPointerOut={onPointerOut}
    >
      {character.parts.map((part, index) => (
        <group
          key={part.name}
          ref={(node) => {
            if (!node) return
            partRefs.current[index] = {
              group: node,
              name: part.name as PartName,
              pivot: [
                part.pivot[0] * VOXEL_SCALE,
                part.pivot[1] * VOXEL_SCALE,
                part.pivot[2] * VOXEL_SCALE,
              ],
            }
          }}
        >
          <mesh geometry={geometries[index]} material={material} castShadow receiveShadow />
        </group>
      ))}
    </group>
  )
}

/** Height of a character in world units — for placing labels above the head. */
export function characterHeight(botId: string): number {
  return characterFor(botId).height * VOXEL_SCALE
}
