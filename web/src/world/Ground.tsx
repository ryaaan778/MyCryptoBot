/**
 * The land itself: terrain, radial roads, water, bridges, and the ambient
 * scatter that makes the world worth zooming into.
 *
 * When the desk hits an emergency stop the whole ground palette shifts red —
 * the environment is a status indicator, not scenery.
 */

import { useMemo } from 'react'

import { useWorldStore } from '../state/store'
import { DISTRICT_RADIUS, LAYOUT, THEME, hashRandom } from './theme'
import { Petals, SakuraTree, Scatter, StoneLantern, Torii } from './props/Scatter'

/** District centres, so scatter and roads keep clear of the built areas. */
const DISTRICT_POINTS: Array<[number, number]> = [
  [0, -46], [-44, -22], [44, -22], [-44, 26], [44, 26],
]

export function Ground() {
  const emergency = useWorldStore((s) => Boolean(s.risk?.emergency_stop))

  const keepOut = useMemo(
    () =>
      [
        [0, 0, 18] as [number, number, number],
        [LAYOUT.marketRoom[0], LAYOUT.marketRoom[1], 16] as [number, number, number],
        [LAYOUT.tradingFloor[0], LAYOUT.tradingFloor[1], 12] as [number, number, number],
        ...DISTRICT_POINTS.map(([x, z]) => [x, z, 13] as [number, number, number]),
      ],
    [],
  )

  const roads = useMemo(
    () =>
      DISTRICT_POINTS.map(([x, z]) => {
        const angle = Math.atan2(z, x)
        const length = Math.hypot(x, z)
        return {
          position: [x / 2, 0.02, z / 2] as [number, number, number],
          rotation: -angle,
          length: length + 6,
        }
      }),
    [],
  )

  const trees = useMemo(() => {
    const out: Array<{ pos: [number, number, number]; scale: number; seed: number }> = []
    for (let i = 0; i < 34; i++) {
      const angle = hashRandom(i * 2.7) * Math.PI * 2
      const radius = 24 + hashRandom(i * 3.9) * 46
      const x = Math.cos(angle) * radius
      const z = Math.sin(angle) * radius
      const nearDistrict = DISTRICT_POINTS.some(
        ([dx, dz]) => (x - dx) ** 2 + (z - dz) ** 2 < 150,
      )
      if (nearDistrict) continue
      out.push({
        pos: [x, 0, z],
        scale: 0.85 + hashRandom(i * 5.1) * 0.7,
        seed: i * 13,
      })
    }
    return out
  }, [])

  const lanterns = useMemo(() => {
    const out: Array<[number, number, number]> = []
    for (const [x, z] of DISTRICT_POINTS) {
      const steps = 4
      for (let i = 1; i <= steps; i++) {
        const t = i / (steps + 1)
        const px = x * t
        const pz = z * t
        // A lantern either side of every approach road.
        const angle = Math.atan2(z, x) + Math.PI / 2
        out.push([px + Math.cos(angle) * 4.2, 0, pz + Math.sin(angle) * 4.2])
        out.push([px - Math.cos(angle) * 4.2, 0, pz - Math.sin(angle) * 4.2])
      }
    }
    return out
  }, [])

  const groundColor = emergency ? '#5c2430' : THEME.grass

  return (
    <group>
      {/* Terrain */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[LAYOUT.groundSize, LAYOUT.groundSize]} />
        <meshLambertMaterial color={groundColor} />
      </mesh>

      {/* A raised, darker plateau under the headquarters */}
      <mesh position={[0, 0.25, 0]} receiveShadow castShadow>
        <boxGeometry args={[34, 0.5, 34]} />
        <meshLambertMaterial color={emergency ? '#4a1d28' : THEME.grassDeep} />
      </mesh>
      <mesh position={[0, 0.55, 0]} receiveShadow>
        <boxGeometry args={[26, 0.2, 26]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>

      {/* Radial roads out to every district */}
      {roads.map((road, index) => (
        <group key={index} position={road.position} rotation={[0, road.rotation, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
            <planeGeometry args={[road.length, 6]} />
            <meshLambertMaterial color={THEME.road} />
          </mesh>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 3.2]} receiveShadow>
            <planeGeometry args={[road.length, 0.5]} />
            <meshLambertMaterial color={THEME.roadEdge} />
          </mesh>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, -3.2]} receiveShadow>
            <planeGeometry args={[road.length, 0.5]} />
            <meshLambertMaterial color={THEME.roadEdge} />
          </mesh>
        </group>
      ))}

      {/* A moat pond with a bridge, east of the plateau */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[26, 0.05, 8]} receiveShadow>
        <planeGeometry args={[22, 16]} />
        <meshLambertMaterial color={THEME.water} transparent opacity={0.85} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[26, 0.04, 8]}>
        <planeGeometry args={[24, 18]} />
        <meshLambertMaterial color={THEME.waterDeep} />
      </mesh>
      <Bridge position={[26, 0, 8]} />

      {/* A second pond to the west, beside the risk center */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-34, 0.05, -4]} receiveShadow>
        <planeGeometry args={[16, 12]} />
        <meshLambertMaterial color={THEME.water} transparent opacity={0.85} />
      </mesh>

      {/* Torii gates on each approach */}
      {DISTRICT_POINTS.map(([x, z], index) => {
        const angle = Math.atan2(z, x)
        const gateRadius = DISTRICT_RADIUS - 17
        return (
          <Torii
            key={index}
            position={[
              Math.cos(angle) * gateRadius,
              0,
              Math.sin(angle) * gateRadius,
            ]}
            rotation={-angle + Math.PI / 2}
            color={emergency ? '#7a2030' : '#c2452f'}
          />
        )
      })}

      {trees.map((tree, index) => (
        <SakuraTree key={index} position={tree.pos} scale={tree.scale} seed={tree.seed} />
      ))}

      {lanterns.map((position, index) => (
        <StoneLantern key={index} position={position} lit={!emergency} />
      ))}

      {/* Instanced ground detail */}
      <Scatter count={520} seed={11} color={THEME.grassLight} size={[0.3, 0.5, 0.3]} avoid={keepOut} />
      <Scatter count={180} seed={29} color={THEME.sakura} size={[0.26, 0.34, 0.26]} avoid={keepOut} />
      <Scatter count={130} seed={47} color={THEME.lantern} size={[0.24, 0.3, 0.24]} avoid={keepOut} />
      <Scatter count={90} seed={71} color={THEME.stone} size={[0.7, 0.5, 0.7]} avoid={keepOut} />
      <Scatter count={60} seed={97} color={THEME.stoneDark} size={[1.1, 0.7, 1.1]} avoid={keepOut} />

      <Petals count={emergency ? 60 : 220} />
    </group>
  )
}

function Bridge({ position }: { position: [number, number, number] }) {
  const planks = useMemo(() => Array.from({ length: 11 }, (_, i) => i), [])
  return (
    <group position={position}>
      {planks.map((i) => {
        const t = (i / (planks.length - 1)) * 2 - 1
        return (
          <mesh key={i} position={[t * 10, 1.4 - t * t * 1.1, 0]} castShadow receiveShadow>
            <boxGeometry args={[2, 0.3, 4.6]} />
            <meshLambertMaterial color={THEME.wood} />
          </mesh>
        )
      })}
      {[-2.2, 2.2].map((z) => (
        <group key={z}>
          {planks.map((i) => {
            const t = (i / (planks.length - 1)) * 2 - 1
            return (
              <mesh key={i} position={[t * 10, 2.3 - t * t * 1.1, z]} castShadow>
                <boxGeometry args={[0.3, 1.5, 0.3]} />
                <meshLambertMaterial color={THEME.woodDark} />
              </mesh>
            )
          })}
        </group>
      ))}
    </group>
  )
}
