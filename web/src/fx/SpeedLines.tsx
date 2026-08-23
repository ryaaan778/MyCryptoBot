/**
 * Dramatic overlays: radiating speed lines when a bot strikes, and the
 * full-world emergency state.
 *
 * Both are pure decoration and both are cheap — the speed lines are a single
 * instanced mesh that is invisible unless something is happening.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { useWorldStore } from '../state/store'
import { THEME } from '../world/theme'

const dummy = new THREE.Object3D()
const LINE_COUNT = 44

/** Radiating manga-style speed lines around a point of action. */
export function SpeedLines({
  position, active, color = '#ffffff', radius = 7,
}: {
  position: [number, number, number]
  active: boolean
  color?: string
  radius?: number
}) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const intensity = useRef(0)

  const lines = useMemo(
    () =>
      Array.from({ length: LINE_COUNT }, (_, i) => {
        const angle = (i / LINE_COUNT) * Math.PI * 2
        return {
          angle,
          length: 1.6 + ((i * 37) % 11) / 6,
          offset: ((i * 53) % 17) / 17,
        }
      }),
    [],
  )

  useFrame(({ clock }, delta) => {
    const instanced = mesh.current
    if (!instanced) return

    // Fade in while active, out when not — no popping.
    const target = active ? 1 : 0
    intensity.current += (target - intensity.current) * Math.min(1, delta * 6)
    if (intensity.current < 0.01) {
      instanced.visible = false
      return
    }
    instanced.visible = true

    const time = clock.elapsedTime
    lines.forEach((line, index) => {
      const pulse = (time * 2.2 + line.offset) % 1
      const distance = radius * (0.45 + pulse * 0.85)
      dummy.position.set(
        Math.cos(line.angle) * distance,
        Math.sin(line.angle * 1.7) * 0.9,
        Math.sin(line.angle) * distance,
      )
      dummy.rotation.set(0, -line.angle, 0)
      dummy.scale.set(
        line.length * intensity.current * (1 - pulse * 0.5),
        0.08,
        0.08,
      )
      dummy.updateMatrix()
      instanced.setMatrixAt(index, dummy.matrix)
    })
    instanced.instanceMatrix.needsUpdate = true

    const material = instanced.material as THREE.MeshBasicMaterial
    material.opacity = 0.55 * intensity.current
  })

  return (
    <group position={position}>
      <instancedMesh ref={mesh} args={[undefined, undefined, LINE_COUNT]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color={color} transparent opacity={0} depthWrite={false} toneMapped={false} />
      </instancedMesh>
    </group>
  )
}

/** Speed lines follow whichever bots are actively striking. */
export function ActionLines() {
  const bots = useWorldStore((s) => s.bots)
  const botOrder = useWorldStore((s) => s.botOrder)

  return (
    <group>
      {botOrder.map((id) => {
        const bot = bots[id]
        if (!bot) return null
        return (
          <SpeedLines
            key={id}
            position={[bot.persona.district[0], 5, bot.persona.district[1] + 5]}
            active={bot.status === 'TRADING'}
            color={bot.persona.accent}
            radius={6}
          />
        )
      })}
    </group>
  )
}

/**
 * Emergency state: the world's light turns red and a warning pillar rises over
 * the headquarters. This is the loudest thing the world can do, reserved for
 * the one condition that warrants it.
 */
export function EmergencyState() {
  const emergency = useWorldStore((s) => Boolean(s.risk?.emergency_stop))
  const light = useRef<THREE.PointLight>(null)
  const pillar = useRef<THREE.Mesh>(null)
  const level = useRef(0)

  useFrame(({ clock }, delta) => {
    const target = emergency ? 1 : 0
    level.current += (target - level.current) * Math.min(1, delta * 3)

    const time = clock.elapsedTime
    if (light.current) {
      light.current.intensity = level.current * (60 + Math.sin(time * 7) * 40)
    }
    if (pillar.current) {
      pillar.current.visible = level.current > 0.02
      pillar.current.scale.set(level.current, 1, level.current)
      const material = pillar.current.material as THREE.MeshBasicMaterial
      material.opacity = level.current * (0.25 + Math.sin(time * 5) * 0.12)
      pillar.current.rotation.y = time * 0.4
    }
  })

  return (
    <group>
      <pointLight ref={light} position={[0, 30, 0]} color={THEME.critical} intensity={0} distance={200} />
      <mesh ref={pillar} position={[0, 40, 0]} visible={false}>
        <cylinderGeometry args={[9, 14, 80, 12, 1, true]} />
        <meshBasicMaterial
          color={THEME.critical}
          transparent
          opacity={0}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}
