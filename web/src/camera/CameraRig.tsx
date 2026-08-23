/**
 * Camera: orbit, zoom, pan, plus eased flights to named presets.
 *
 * Clicking a character sets `cameraTarget` in the store; this rig notices and
 * flies there. Any manual orbit input cancels the flight immediately, so the
 * camera never fights the user.
 */

import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

import { useWorldStore } from '../state/store'
import { presetFor } from '../world/theme'

const FLIGHT_SPEED = 1.9

export function CameraRig() {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera } = useThree()
  const cameraTarget = useWorldStore((s) => s.cameraTarget)

  const flying = useRef(false)
  const fromPosition = useRef(new THREE.Vector3())
  const toPosition = useRef(new THREE.Vector3())
  const fromTarget = useRef(new THREE.Vector3())
  const toTarget = useRef(new THREE.Vector3())
  const progress = useRef(1)

  useEffect(() => {
    const preset = presetFor(cameraTarget)
    fromPosition.current.copy(camera.position)
    toPosition.current.set(...preset.position)
    fromTarget.current.copy(controls.current?.target ?? new THREE.Vector3())
    toTarget.current.set(...preset.target)
    progress.current = 0
    flying.current = true
  }, [cameraTarget, camera])

  useFrame((_, delta) => {
    if (!flying.current || !controls.current) return
    progress.current = Math.min(1, progress.current + delta * FLIGHT_SPEED)
    // Smoothstep: eases out of the old view and into the new one.
    const t = progress.current * progress.current * (3 - 2 * progress.current)

    camera.position.lerpVectors(fromPosition.current, toPosition.current, t)
    controls.current.target.lerpVectors(fromTarget.current, toTarget.current, t)
    controls.current.update()

    if (progress.current >= 1) flying.current = false
  })

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enablePan
      enableZoom
      enableDamping
      dampingFactor={0.08}
      minDistance={6}
      maxDistance={190}
      maxPolarAngle={Math.PI / 2.08}
      target={[0, 6, 0]}
      onStart={() => { flying.current = false }}
    />
  )
}
