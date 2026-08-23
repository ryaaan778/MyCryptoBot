/**
 * The Trading Floor — where order flow becomes visible.
 *
 * Every submitted order launches a voxel projectile from its bot's district
 * toward the market display; a fill detonates it. All of it is pooled: one
 * InstancedMesh for projectiles, one for impact debris, with a fixed cap. Under
 * a burst of fills the pool simply reuses its oldest slot rather than growing.
 */

import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { useWorldStore } from '../state/store'
import { drainEffects, type EffectRequest } from '../fx/effectQueue'
import { HoloLabel } from './HoloPanel'
import { LAYOUT, THEME } from './theme'

const dummy = new THREE.Object3D()
const MAX_PROJECTILES = 64
const MAX_DEBRIS = 220

interface Projectile {
  active: boolean
  from: THREE.Vector3
  to: THREE.Vector3
  progress: number
  speed: number
  color: THREE.Color
  arc: number
  scale: number
}

interface Debris {
  active: boolean
  position: THREE.Vector3
  velocity: THREE.Vector3
  life: number
  maxLife: number
  color: THREE.Color
  scale: number
}

const MARKET_POINT = new THREE.Vector3(LAYOUT.marketRoom[0], 13, LAYOUT.marketRoom[1])

export function TradingFloor() {
  const bots = useWorldStore((s) => s.bots)
  const botOrder = useWorldStore((s) => s.botOrder)

  const projectiles = useRef<Projectile[]>(
    Array.from({ length: MAX_PROJECTILES }, () => ({
      active: false,
      from: new THREE.Vector3(),
      to: new THREE.Vector3(),
      progress: 0,
      speed: 1,
      color: new THREE.Color(),
      arc: 6,
      scale: 1,
    })),
  )
  const debris = useRef<Debris[]>(
    Array.from({ length: MAX_DEBRIS }, () => ({
      active: false,
      position: new THREE.Vector3(),
      velocity: new THREE.Vector3(),
      life: 0,
      maxLife: 1,
      color: new THREE.Color(),
      scale: 1,
    })),
  )

  const projectileMesh = useRef<THREE.InstancedMesh>(null)
  const debrisMesh = useRef<THREE.InstancedMesh>(null)
  const nextProjectile = useRef(0)
  const nextDebris = useRef(0)

  /** District origin for a bot, in world space. */
  const originFor = useMemo(() => {
    const map = new Map<string, THREE.Vector3>()
    for (const id of botOrder) {
      const bot = bots[id]
      if (!bot) continue
      map.set(id, new THREE.Vector3(bot.persona.district[0], 6, bot.persona.district[1]))
    }
    return map
  }, [bots, botOrder])

  const spawnProjectile = (botId: string, color: THREE.Color, outbound: boolean) => {
    const origin = originFor.get(botId)
    if (!origin) return
    const slot = projectiles.current[nextProjectile.current]
    nextProjectile.current = (nextProjectile.current + 1) % MAX_PROJECTILES
    slot.active = true
    slot.progress = 0
    slot.speed = 0.85 + Math.random() * 0.5
    slot.color.copy(color)
    slot.arc = 7 + Math.random() * 4
    slot.scale = 0.5 + Math.random() * 0.25
    if (outbound) {
      slot.from.copy(origin)
      slot.to.copy(MARKET_POINT)
    } else {
      slot.from.copy(MARKET_POINT)
      slot.to.copy(origin)
    }
  }

  const spawnBurst = (
    at: THREE.Vector3, color: THREE.Color, count: number, power: number,
  ) => {
    for (let i = 0; i < count; i++) {
      const slot = debris.current[nextDebris.current]
      nextDebris.current = (nextDebris.current + 1) % MAX_DEBRIS
      slot.active = true
      slot.position.copy(at)
      slot.velocity.set(
        (Math.random() - 0.5) * power,
        Math.random() * power * 0.9 + 1.2,
        (Math.random() - 0.5) * power,
      )
      slot.maxLife = 0.9 + Math.random() * 0.7
      slot.life = slot.maxLife
      slot.color.copy(color)
      slot.scale = 0.22 + Math.random() * 0.3
    }
  }

  const handleEffect = (effect: EffectRequest) => {
    switch (effect.kind) {
      case 'order':
        spawnProjectile(
          effect.botId,
          new THREE.Color(effect.side === 'BUY' ? THEME.profit : THEME.loss),
          true,
        )
        return
      case 'fill': {
        const origin = originFor.get(effect.botId)
        if (origin) {
          spawnBurst(
            MARKET_POINT.clone(),
            new THREE.Color(effect.closing ? THEME.warning : THEME.hologram),
            10, 5,
          )
          spawnProjectile(effect.botId, new THREE.Color(THEME.hologram), false)
        }
        return
      }
      case 'profit': {
        const origin = originFor.get(effect.botId)
        if (origin) {
          const count = Math.min(34, 12 + Math.round(effect.magnitude))
          spawnBurst(origin.clone().setY(4), new THREE.Color(THEME.gold), count, 7)
        }
        return
      }
      case 'loss': {
        const origin = originFor.get(effect.botId)
        if (origin) {
          spawnBurst(origin.clone().setY(4), new THREE.Color(THEME.loss), 14, 4)
        }
        return
      }
      case 'signal': {
        const origin = originFor.get(effect.botId)
        if (origin) {
          const color =
            effect.action === 'LONG' ? THEME.profit
              : effect.action === 'SHORT' ? THEME.loss
                : THEME.warning
          spawnBurst(origin.clone().setY(5), new THREE.Color(color), 8, 3)
        }
        return
      }
      case 'emergency':
        spawnBurst(new THREE.Vector3(0, 8, 0), new THREE.Color(THEME.critical), 40, 12)
        return
    }
  }

  useLayoutEffect(() => {
    const mesh = projectileMesh.current
    if (mesh) mesh.count = MAX_PROJECTILES
    const dmesh = debrisMesh.current
    if (dmesh) dmesh.count = MAX_DEBRIS
  }, [])

  useFrame((_, delta) => {
    // Drain the bounded effect queue. Anything beyond the frame budget waits
    // for the next frame, or is dropped if the queue overflows.
    for (const effect of drainEffects()) handleEffect(effect)

    const pMesh = projectileMesh.current
    if (pMesh) {
      projectiles.current.forEach((slot, index) => {
        if (!slot.active) {
          dummy.position.set(0, -1000, 0)
          dummy.scale.setScalar(0.001)
        } else {
          slot.progress += delta * slot.speed
          if (slot.progress >= 1) {
            slot.active = false
            spawnBurst(slot.to.clone(), slot.color, 6, 4)
            dummy.position.set(0, -1000, 0)
            dummy.scale.setScalar(0.001)
          } else {
            const t = slot.progress
            dummy.position.lerpVectors(slot.from, slot.to, t)
            // Parabolic arc so orders visibly fly rather than slide.
            dummy.position.y += Math.sin(t * Math.PI) * slot.arc
            dummy.rotation.set(t * 9, t * 7, 0)
            dummy.scale.setScalar(slot.scale)
          }
        }
        dummy.updateMatrix()
        pMesh.setMatrixAt(index, dummy.matrix)
        pMesh.setColorAt(index, slot.color)
      })
      pMesh.instanceMatrix.needsUpdate = true
      if (pMesh.instanceColor) pMesh.instanceColor.needsUpdate = true
    }

    const dMesh = debrisMesh.current
    if (dMesh) {
      debris.current.forEach((slot, index) => {
        if (!slot.active) {
          dummy.position.set(0, -1000, 0)
          dummy.scale.setScalar(0.001)
        } else {
          slot.life -= delta
          if (slot.life <= 0) {
            slot.active = false
            dummy.position.set(0, -1000, 0)
            dummy.scale.setScalar(0.001)
          } else {
            slot.velocity.y -= delta * 9.2
            slot.position.addScaledVector(slot.velocity, delta)
            if (slot.position.y < 0.2) {
              slot.position.y = 0.2
              slot.velocity.y *= -0.35
              slot.velocity.x *= 0.7
              slot.velocity.z *= 0.7
            }
            dummy.position.copy(slot.position)
            dummy.rotation.set(slot.life * 6, slot.life * 4, slot.life * 5)
            dummy.scale.setScalar(slot.scale * (slot.life / slot.maxLife))
          }
        }
        dummy.updateMatrix()
        dMesh.setMatrixAt(index, dummy.matrix)
        dMesh.setColorAt(index, slot.color)
      })
      dMesh.instanceMatrix.needsUpdate = true
      if (dMesh.instanceColor) dMesh.instanceColor.needsUpdate = true
    }
  })

  const [fx, fz] = LAYOUT.tradingFloor

  return (
    <group>
      {/* The floor itself */}
      <group position={[fx, 0, fz]}>
        <mesh position={[0, 0.22, 0]} receiveShadow>
          <boxGeometry args={[22, 0.44, 14]} />
          <meshLambertMaterial color={THEME.stoneDark} />
        </mesh>
        <mesh position={[0, 0.48, 0]} receiveShadow>
          <boxGeometry args={[18, 0.1, 10]} />
          <meshLambertMaterial color={THEME.hologram} emissive={THEME.hologram} emissiveIntensity={0.18} />
        </mesh>
        <HoloLabel position={[0, 2.4, 0]} text="TRADING FLOOR" color={THEME.hologram} scale={0.8} />
        <OrderFlowStalls />
      </group>

      <instancedMesh ref={projectileMesh} args={[undefined, undefined, MAX_PROJECTILES]} frustumCulled={false}>
        <boxGeometry args={[0.9, 0.9, 0.9]} />
        <meshBasicMaterial vertexColors toneMapped={false} />
      </instancedMesh>

      <instancedMesh ref={debrisMesh} args={[undefined, undefined, MAX_DEBRIS]} frustumCulled={false}>
        <boxGeometry args={[0.9, 0.9, 0.9]} />
        <meshBasicMaterial vertexColors toneMapped={false} />
      </instancedMesh>
    </group>
  )
}

/** Market stalls flanking the floor — small world detail worth zooming into. */
function OrderFlowStalls() {
  const stalls = useMemo(
    () => [
      { x: -8, color: THEME.profit },
      { x: -3, color: THEME.lantern },
      { x: 3, color: THEME.sakura },
      { x: 8, color: THEME.loss },
    ],
    [],
  )
  return (
    <group>
      {stalls.map((stall, index) => (
        <group key={index} position={[stall.x, 0.5, -5.5]}>
          <mesh position={[0, 1, 0]} castShadow>
            <boxGeometry args={[3, 0.3, 2.4]} />
            <meshLambertMaterial color={THEME.wood} />
          </mesh>
          {[-1.3, 1.3].map((lx) => (
            <mesh key={lx} position={[lx, 0.5, 0]} castShadow>
              <boxGeometry args={[0.2, 1, 0.2]} />
              <meshLambertMaterial color={THEME.woodDark} />
            </mesh>
          ))}
          <mesh position={[0, 2, 0]} castShadow>
            <boxGeometry args={[3.4, 0.9, 2.8]} />
            <meshLambertMaterial color={stall.color} />
          </mesh>
        </group>
      ))}
    </group>
  )
}
