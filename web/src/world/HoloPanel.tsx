/**
 * Holographic data panels.
 *
 * Trading numbers have to be *readable* inside the world, not decorative. These
 * draw into a canvas texture and map it onto a plane: crisp text at any zoom,
 * no font files to fetch, and no DOM overlay cost per panel.
 *
 * Redraws are throttled and only happen when the rendered content actually
 * changes, so a wall of live panels costs almost nothing per frame.
 */

import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { THEME } from './theme'

export interface PanelRow {
  label: string
  value: string
  /** CSS colour for the value; defaults to white. */
  color?: string
  /** Renders the row larger — for the headline number on a panel. */
  emphasis?: boolean
}

export interface HoloPanelProps {
  position: [number, number, number]
  rotation?: [number, number, number]
  title?: string
  rows: PanelRow[]
  width?: number
  accent?: string
  /** World-space height; width follows from the aspect of the canvas. */
  scale?: number
  opacity?: number
  billboard?: boolean
}

const CANVAS_WIDTH = 512
const LINE_HEIGHT = 46
const HEADER_HEIGHT = 62
const PADDING = 22

function drawPanel(
  canvas: HTMLCanvasElement,
  title: string | undefined,
  rows: PanelRow[],
  accent: string,
): void {
  const height =
    PADDING * 2 + (title ? HEADER_HEIGHT : 0) +
    rows.reduce((sum, row) => sum + (row.emphasis ? LINE_HEIGHT * 1.6 : LINE_HEIGHT), 0)
  canvas.width = CANVAS_WIDTH
  canvas.height = Math.max(96, Math.round(height))

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Panel body
  ctx.fillStyle = 'rgba(8, 12, 26, 0.82)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  // Accent frame — the corner ticks read as a holographic projection.
  ctx.strokeStyle = accent
  ctx.lineWidth = 3
  ctx.strokeRect(1.5, 1.5, canvas.width - 3, canvas.height - 3)
  ctx.lineWidth = 6
  const tick = 26
  ctx.beginPath()
  ctx.moveTo(0, tick); ctx.lineTo(0, 0); ctx.lineTo(tick, 0)
  ctx.moveTo(canvas.width - tick, 0); ctx.lineTo(canvas.width, 0); ctx.lineTo(canvas.width, tick)
  ctx.moveTo(0, canvas.height - tick); ctx.lineTo(0, canvas.height); ctx.lineTo(tick, canvas.height)
  ctx.moveTo(canvas.width - tick, canvas.height); ctx.lineTo(canvas.width, canvas.height)
  ctx.lineTo(canvas.width, canvas.height - tick)
  ctx.stroke()

  let y = PADDING

  if (title) {
    ctx.fillStyle = accent
    ctx.font = 'bold 34px ui-monospace, "SF Mono", Menlo, Consolas, monospace'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText(title.toUpperCase(), PADDING, y)
    y += HEADER_HEIGHT - 18
    ctx.fillStyle = accent
    ctx.globalAlpha = 0.45
    ctx.fillRect(PADDING, y, canvas.width - PADDING * 2, 2)
    ctx.globalAlpha = 1
    y += 16
  }

  for (const row of rows) {
    const size = row.emphasis ? 44 : 28
    const rowHeight = row.emphasis ? LINE_HEIGHT * 1.6 : LINE_HEIGHT

    ctx.font = `500 ${row.emphasis ? 22 : 22}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`
    ctx.fillStyle = 'rgba(220, 228, 245, 0.62)'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText(row.label.toUpperCase(), PADDING, y + (row.emphasis ? 4 : 6))

    ctx.font = `bold ${size}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`
    ctx.fillStyle = row.color ?? '#f2f6ff'
    ctx.textAlign = 'right'
    ctx.fillText(row.value, canvas.width - PADDING, y + (row.emphasis ? 22 : 2))

    y += rowHeight
  }
}

export function HoloPanel({
  position,
  rotation = [0, 0, 0],
  title,
  rows,
  accent = THEME.hologram,
  scale = 3,
  opacity = 0.96,
  billboard = false,
}: HoloPanelProps) {
  const canvas = useMemo(() => document.createElement('canvas'), [])
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas)
    t.colorSpace = THREE.SRGBColorSpace
    t.minFilter = THREE.LinearFilter
    t.magFilter = THREE.LinearFilter
    return t
  }, [canvas])

  const group = useRef<THREE.Group>(null)
  // Signature of the rendered content — redraw only when it actually changes.
  const signature = `${title ?? ''}|${accent}|${rows
    .map((r) => `${r.label}=${r.value}=${r.color ?? ''}=${r.emphasis ?? false}`)
    .join(';')}`
  const lastSignature = useRef('')

  useEffect(() => {
    if (signature === lastSignature.current) return
    lastSignature.current = signature
    drawPanel(canvas, title, rows, accent)
    texture.needsUpdate = true
    // `rows` is intentionally excluded: `signature` already captures its content,
    // and the array identity changes every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, canvas, texture, title, accent])

  useEffect(() => () => texture.dispose(), [texture])

  useFrame(({ camera }) => {
    if (!billboard || !group.current) return
    group.current.quaternion.copy(camera.quaternion)
  })

  const aspect = canvas.height > 0 ? canvas.width / canvas.height : 2
  const height = scale
  const width = height * aspect

  return (
    <group ref={group} position={position} rotation={billboard ? [0, 0, 0] : rotation}>
      <mesh>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial
          map={texture}
          transparent
          opacity={opacity}
          depthWrite={false}
          side={THREE.DoubleSide}
          toneMapped={false}
        />
      </mesh>
      {/* A faint backing glow so panels read against bright sky. */}
      <mesh position={[0, 0, -0.02]}>
        <planeGeometry args={[width * 1.04, height * 1.06]} />
        <meshBasicMaterial
          color={accent}
          transparent
          opacity={0.13}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  )
}

/** A single bold line of world text — names, P&L callouts, district signs. */
export function HoloLabel({
  position,
  text,
  color = '#ffffff',
  scale = 1.4,
  billboard = true,
  outline = true,
}: {
  position: [number, number, number]
  text: string
  color?: string
  scale?: number
  billboard?: boolean
  outline?: boolean
}) {
  const canvas = useMemo(() => document.createElement('canvas'), [])
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas)
    t.colorSpace = THREE.SRGBColorSpace
    t.minFilter = THREE.LinearFilter
    return t
  }, [canvas])
  const group = useRef<THREE.Group>(null)
  const last = useRef('')

  useEffect(() => {
    const signature = `${text}|${color}|${outline}`
    if (signature === last.current) return
    last.current = signature

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const font = 'bold 84px ui-monospace, "SF Mono", Menlo, Consolas, monospace'
    ctx.font = font
    const metrics = ctx.measureText(text)
    canvas.width = Math.max(64, Math.ceil(metrics.width) + 48)
    canvas.height = 128

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.font = font
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    if (outline) {
      ctx.lineWidth = 12
      ctx.strokeStyle = 'rgba(6, 8, 18, 0.92)'
      ctx.strokeText(text, canvas.width / 2, canvas.height / 2)
    }
    ctx.fillStyle = color
    ctx.fillText(text, canvas.width / 2, canvas.height / 2)
    texture.needsUpdate = true
  }, [text, color, outline, canvas, texture])

  useEffect(() => () => texture.dispose(), [texture])

  useFrame(({ camera }) => {
    if (billboard && group.current) group.current.quaternion.copy(camera.quaternion)
  })

  const aspect = canvas.height > 0 ? canvas.width / canvas.height : 4
  return (
    <group ref={group} position={position}>
      <mesh>
        <planeGeometry args={[scale * aspect, scale]} />
        <meshBasicMaterial map={texture} transparent depthWrite={false} toneMapped={false} />
      </mesh>
    </group>
  )
}
