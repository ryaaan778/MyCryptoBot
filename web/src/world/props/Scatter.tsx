/**
 * Environment scatter: grass tufts, flowers, rocks, petals, lanterns.
 *
 * All of it goes through `InstancedMesh` — one draw call per prop type no
 * matter how many are placed. Positions are derived from a deterministic hash
 * so the world looks hand-placed but costs nothing to generate and never
 * shifts between renders.
 */

import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { LAYOUT, THEME, hashRandom } from '../theme'

const dummy = new THREE.Object3D()

interface ScatterProps {
  count: number
  seed: number
  color: string
  size: [number, number, number]
  /** Keep-out radius around the world centre and each district. */
  avoid?: Array<[number, number, number]>
  yBase?: number
  jitterScale?: number
}

function useScatterPositions(
  count: number,
  seed: number,
  avoid: Array<[number, number, number]>,
): Array<[number, number, number]> {
  return useMemo(() => {
    const half = LAYOUT.groundSize / 2
    const out: Array<[number, number, number]> = []
    let attempt = 0
    while (out.length < count && attempt < count * 6) {
      const s = seed + attempt * 3
      attempt += 1
      const x = (hashRandom(s) - 0.5) * half * 2
      const z = (hashRandom(s + 0.5) - 0.5) * half * 2
      const blocked = avoid.some(([ax, az, r]) => (x - ax) ** 2 + (z - az) ** 2 < r * r)
      if (blocked) continue
      out.push([x, 0, z])
    }
    return out
  }, [count, seed, avoid])
}

export function Scatter({
  count, seed, color, size, avoid = [], yBase = 0, jitterScale = 0.4,
}: ScatterProps) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const positions = useScatterPositions(count, seed, avoid)

  useLayoutEffect(() => {
    const instanced = mesh.current
    if (!instanced) return
    positions.forEach(([x, , z], index) => {
      const scale = 1 + (hashRandom(seed + index * 7) - 0.5) * jitterScale
      dummy.position.set(x, yBase + (size[1] * scale) / 2, z)
      dummy.rotation.set(0, hashRandom(seed + index * 11) * Math.PI * 2, 0)
      dummy.scale.setScalar(scale)
      dummy.updateMatrix()
      instanced.setMatrixAt(index, dummy.matrix)
    })
    instanced.instanceMatrix.needsUpdate = true
    instanced.computeBoundingSphere()
  }, [positions, seed, size, yBase, jitterScale])

  if (positions.length === 0) return null

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, positions.length]}
      castShadow={false}
      receiveShadow
      frustumCulled
    >
      <boxGeometry args={size} />
      <meshLambertMaterial color={color} />
    </instancedMesh>
  )
}

/** Falling cherry petals — the world's ambient motion. */
export function Petals({ count = 220 }: { count?: number }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const seeds = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => ({
        x: (hashRandom(i * 1.7) - 0.5) * LAYOUT.groundSize,
        z: (hashRandom(i * 2.3) - 0.5) * LAYOUT.groundSize,
        y: hashRandom(i * 3.1) * 30 + 4,
        speed: 0.6 + hashRandom(i * 4.7) * 1.1,
        drift: hashRandom(i * 5.3) * Math.PI * 2,
        spin: 0.5 + hashRandom(i * 6.1),
      })),
    [count],
  )

  useFrame(({ clock }) => {
    const instanced = mesh.current
    if (!instanced) return
    const time = clock.elapsedTime
    for (let i = 0; i < seeds.length; i++) {
      const petal = seeds[i]
      // Wrap through a 34-unit column so petals fall forever.
      const y = 34 - ((petal.y + time * petal.speed) % 34)
      dummy.position.set(
        petal.x + Math.sin(time * 0.5 + petal.drift) * 2.4,
        y,
        petal.z + Math.cos(time * 0.4 + petal.drift) * 2.4,
      )
      dummy.rotation.set(time * petal.spin, time * petal.spin * 0.7, 0)
      dummy.scale.setScalar(0.9)
      dummy.updateMatrix()
      instanced.setMatrixAt(i, dummy.matrix)
    }
    instanced.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, count]} frustumCulled={false}>
      <boxGeometry args={[0.22, 0.06, 0.16]} />
      <meshBasicMaterial color={THEME.sakura} transparent opacity={0.9} />
    </instancedMesh>
  )
}

/** A cherry tree: trunk plus a blossom canopy of stacked boxes. */
export function SakuraTree({
  position, scale = 1, seed = 0,
}: {
  position: [number, number, number]
  scale?: number
  seed?: number
}) {
  const blossoms = useMemo(() => {
    const out: Array<{ pos: [number, number, number]; size: number; deep: boolean }> = []
    for (let i = 0; i < 14; i++) {
      const angle = hashRandom(seed + i) * Math.PI * 2
      const radius = 0.6 + hashRandom(seed + i + 0.3) * 1.9
      out.push({
        pos: [
          Math.cos(angle) * radius,
          3.2 + hashRandom(seed + i + 0.7) * 1.8,
          Math.sin(angle) * radius,
        ],
        size: 0.9 + hashRandom(seed + i + 0.9) * 0.9,
        deep: hashRandom(seed + i + 1.3) > 0.65,
      })
    }
    return out
  }, [seed])

  return (
    <group position={position} scale={scale}>
      <mesh position={[0, 1.6, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.5, 3.2, 0.5]} />
        <meshLambertMaterial color={THEME.woodDark} />
      </mesh>
      <mesh position={[0.5, 2.6, 0.2]} rotation={[0, 0, -0.5]} castShadow>
        <boxGeometry args={[0.3, 1.4, 0.3]} />
        <meshLambertMaterial color={THEME.woodDark} />
      </mesh>
      {blossoms.map((blossom, index) => (
        <mesh key={index} position={blossom.pos} castShadow>
          <boxGeometry args={[blossom.size, blossom.size * 0.7, blossom.size]} />
          <meshLambertMaterial color={blossom.deep ? THEME.sakuraDeep : THEME.sakura} />
        </mesh>
      ))}
    </group>
  )
}

/** A stone lantern — the recurring light motif along every path. */
export function StoneLantern({
  position, lit = true,
}: {
  position: [number, number, number]
  lit?: boolean
}) {
  return (
    <group position={position}>
      <mesh position={[0, 0.2, 0]} receiveShadow>
        <boxGeometry args={[0.9, 0.4, 0.9]} />
        <meshLambertMaterial color={THEME.stoneDark} />
      </mesh>
      <mesh position={[0, 0.9, 0]} castShadow>
        <boxGeometry args={[0.4, 1.2, 0.4]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>
      <mesh position={[0, 1.75, 0]} castShadow>
        <boxGeometry args={[0.75, 0.7, 0.75]} />
        <meshLambertMaterial
          color={lit ? THEME.lantern : THEME.stoneDark}
          emissive={lit ? THEME.lantern : '#000000'}
          emissiveIntensity={lit ? 0.85 : 0}
        />
      </mesh>
      <mesh position={[0, 2.25, 0]} castShadow>
        <boxGeometry args={[1.1, 0.25, 1.1]} />
        <meshLambertMaterial color={THEME.stoneLight} />
      </mesh>
      {lit && <pointLight position={[0, 1.75, 0]} color={THEME.lantern} intensity={4} distance={9} />}
    </group>
  )
}

/** A torii gate marking the entrance to a district. */
export function Torii({
  position, rotation = 0, color = '#c2452f', scale = 1,
}: {
  position: [number, number, number]
  rotation?: number
  color?: string
  scale?: number
}) {
  return (
    <group position={position} rotation={[0, rotation, 0]} scale={scale}>
      <mesh position={[-2.4, 3, 0]} castShadow>
        <boxGeometry args={[0.55, 6, 0.55]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[2.4, 3, 0]} castShadow>
        <boxGeometry args={[0.55, 6, 0.55]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0, 6.3, 0]} castShadow>
        <boxGeometry args={[6.6, 0.6, 0.8]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0, 5.4, 0]} castShadow>
        <boxGeometry args={[5.4, 0.4, 0.6]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0, 6.85, 0]} castShadow>
        <boxGeometry args={[7.2, 0.3, 1]} />
        <meshLambertMaterial color={THEME.night} />
      </mesh>
    </group>
  )
}

/** A dramatic statue on a plinth — one per district. */
export function Statue({
  position, color, accent, rotation = 0,
}: {
  position: [number, number, number]
  color: string
  accent: string
  rotation?: number
}) {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh position={[0, 0.5, 0]} receiveShadow castShadow>
        <boxGeometry args={[3.2, 1, 3.2]} />
        <meshLambertMaterial color={THEME.stoneDark} />
      </mesh>
      <mesh position={[0, 1.3, 0]} castShadow>
        <boxGeometry args={[2.4, 0.6, 2.4]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>
      {/* An abstract figure mid-pose: torso, raised arm, head. */}
      <mesh position={[0, 3, 0]} castShadow>
        <boxGeometry args={[1.1, 2.8, 0.9]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0.95, 4.4, 0]} rotation={[0, 0, -0.7]} castShadow>
        <boxGeometry args={[0.5, 2.2, 0.5]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[-0.8, 3.2, 0]} rotation={[0, 0, 0.45]} castShadow>
        <boxGeometry args={[0.45, 1.8, 0.45]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0, 4.9, 0]} castShadow>
        <boxGeometry args={[0.95, 1, 0.95]} />
        <meshLambertMaterial color={color} />
      </mesh>
      <mesh position={[0, 5.6, 0]} castShadow>
        <boxGeometry args={[1.2, 0.35, 1.2]} />
        <meshLambertMaterial color={accent} emissive={accent} emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}
