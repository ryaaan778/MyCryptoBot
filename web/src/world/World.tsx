/**
 * The world scene graph, plus the lighting and quality tiering that keep it
 * running on modest hardware.
 */

import { Canvas } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'

import { useWorldStore } from '../state/store'
import { CameraRig } from '../camera/CameraRig'
import { ActionLines, EmergencyState } from '../fx/SpeedLines'
import { BotDistrict } from './BotDistrict'
import { Ground } from './Ground'
import { Headquarters } from './Headquarters'
import { MarketRoom } from './MarketRoom'
import { TradingFloor } from './TradingFloor'
import { THEME } from './theme'
import { useQualityMonitor } from '../perf/quality'

export function World() {
  const quality = useWorldStore((s) => s.quality)
  const showTables = useWorldStore((s) => s.showTables)

  const dpr = useMemo<[number, number]>(() => {
    if (quality === 'low') return [0.6, 1]
    if (quality === 'medium') return [0.8, 1.35]
    return [1, 1.9]
  }, [quality])

  return (
    <Canvas
      // The tables overlay covers the world completely, so rendering behind it
      // is pure waste — and on a slow machine that wasted frame time is what
      // makes the tables feel unresponsive.
      frameloop={showTables ? 'never' : 'always'}
      shadows={quality === 'high'}
      dpr={dpr}
      camera={{ position: [0, 78, 118], fov: 52, near: 0.5, far: 700 }}
      gl={{
        antialias: quality !== 'low',
        powerPreference: 'high-performance',
        alpha: false,
      }}
      onCreated={({ gl, scene }) => {
        gl.setClearColor(new THREE.Color('#0b1030'))
        scene.fog = new THREE.Fog('#0b1030', 170, 380)
      }}
    >
      <Suspense fallback={null}>
        <SceneContents />
      </Suspense>
    </Canvas>
  )
}

function SceneContents() {
  const bots = useWorldStore((s) => s.bots)
  const botOrder = useWorldStore((s) => s.botOrder)
  const selectBot = useWorldStore((s) => s.selectBot)
  const quality = useWorldStore((s) => s.quality)

  useQualityMonitor()

  return (
    <>
      <WorldLighting />
      <CameraRig />

      {/* Clicking empty space deselects. */}
      <mesh
        position={[0, -1, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        onClick={() => selectBot(null)}
      >
        <planeGeometry args={[400, 400]} />
        <meshBasicMaterial visible={false} />
      </mesh>

      <Ground />
      <Headquarters />
      <MarketRoom />
      <TradingFloor />

      {botOrder.map((id, index) => {
        const bot = bots[id]
        if (!bot) return null
        return <BotDistrict key={id} bot={bot} index={index} />
      })}

      {quality !== 'low' && <ActionLines />}
      <EmergencyState />
      <Sky />
    </>
  )
}

function WorldLighting() {
  const emergency = useWorldStore((s) => Boolean(s.risk?.emergency_stop))
  const quality = useWorldStore((s) => s.quality)
  const key = useRef<THREE.DirectionalLight>(null)

  useFrame((_, delta) => {
    if (!key.current) return
    // The key light bleeds toward red when the desk is halted.
    const target = emergency ? new THREE.Color('#ff6a6a') : new THREE.Color('#fff4e0')
    key.current.color.lerp(target, Math.min(1, delta * 2.5))
  })

  return (
    <>
      <ambientLight intensity={emergency ? 0.45 : 0.85} color={emergency ? '#61304a' : '#a8b8e8'} />
      <hemisphereLight
        intensity={emergency ? 0.4 : 0.7}
        color={emergency ? '#803048' : '#cfe0ff'}
        groundColor={THEME.grassDeep}
      />
      <directionalLight
        ref={key}
        position={[48, 70, 32]}
        intensity={emergency ? 1.0 : 1.5}
        castShadow={quality === 'high'}
        shadow-mapSize-width={quality === 'high' ? 2048 : 1024}
        shadow-mapSize-height={quality === 'high' ? 2048 : 1024}
        shadow-camera-left={-90}
        shadow-camera-right={90}
        shadow-camera-top={90}
        shadow-camera-bottom={-90}
        shadow-camera-far={220}
        shadow-bias={-0.0007}
      />
      {/* Rim and fill: dark-palette characters (JOTARO especially) would
          otherwise disappear against the night sky. */}
      <directionalLight position={[-40, 30, -50]} intensity={0.55} color="#7f9fff" />
      <directionalLight position={[0, 18, 60]} intensity={0.35} color="#ffe8d0" />
    </>
  )
}

/** A star dome and a moon — the world reads as a permanent dramatic night. */
function Sky() {
  const stars = useMemo(() => {
    const positions = new Float32Array(900 * 3)
    for (let i = 0; i < 900; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(Math.random() * 0.9)
      const radius = 250
      positions[i * 3] = Math.sin(phi) * Math.cos(theta) * radius
      positions[i * 3 + 1] = Math.cos(phi) * radius + 30
      positions[i * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius
    }
    return positions
  }, [])

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(stars, 3))
    return g
  }, [stars])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <group>
      <points geometry={geometry}>
        <pointsMaterial size={1.4} color="#ffffff" sizeAttenuation transparent opacity={0.75} />
      </points>
      <mesh position={[-120, 110, -180]}>
        <boxGeometry args={[26, 26, 4]} />
        <meshBasicMaterial color="#fff3cf" toneMapped={false} />
      </mesh>
    </group>
  )
}

/** Reports the live renderer stats the HUD's perf readout shows. */
export function useRendererInfo(): () => { calls: number; triangles: number } {
  const { gl } = useThree()
  return () => ({
    calls: gl.info.render.calls,
    triangles: gl.info.render.triangles,
  })
}
