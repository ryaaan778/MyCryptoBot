/**
 * The Market Room — a holographic candlestick chart rendered as voxel geometry.
 *
 * Candles are drawn with two InstancedMeshes (bodies and wicks), so a 120-bar
 * chart that updates live is two draw calls. The symbol on display follows the
 * selected bot, defaulting to the first series the backend sent.
 */

import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import type { Candle } from '../net/protocol'
import { candleKey, useWorldStore } from '../state/store'
import { HoloLabel, HoloPanel, type PanelRow } from './HoloPanel'
import { LAYOUT, THEME } from './theme'
import { displaySymbol, pct, plainPct, pnlColor, price } from './format'

const dummy = new THREE.Object3D()
const CHART_WIDTH = 30
const CHART_HEIGHT = 12
const MAX_BARS = 120

export function MarketRoom() {
  const bots = useWorldStore((s) => s.bots)
  const selectedBot = useWorldStore((s) => s.selectedBot)
  const candles = useWorldStore((s) => s.candles)
  const tickers = useWorldStore((s) => s.tickers)
  const focusCamera = useWorldStore((s) => s.focusCamera)

  // Show the selected bot's market, or fall back to the first available series.
  const active = selectedBot ? bots[selectedBot] : null
  const key = active
    ? candleKey(active.symbol, active.timeframe)
    : Object.keys(candles)[0]
  const symbol = active?.symbol ?? key?.split('|')[0] ?? ''
  const timeframe = active?.timeframe ?? key?.split('|')[1] ?? ''
  const series = (key ? candles[key] : undefined) ?? []
  const ticker = tickers[symbol]

  const [x, z] = LAYOUT.marketRoom

  const rows: PanelRow[] = ticker
    ? [
        { label: 'Price', value: price(ticker.price), emphasis: true },
        { label: 'Bid', value: price(ticker.bid), color: THEME.profit },
        { label: 'Ask', value: price(ticker.ask), color: THEME.loss },
        {
          label: 'Spread',
          value: `${(((ticker.ask - ticker.bid) / ticker.price) * 10_000).toFixed(2)} bps`,
        },
        {
          label: '24h Change',
          value: pct(ticker.change_24h_pct),
          color: pnlColor(ticker.change_24h_pct),
        },
      ]
    : [{ label: 'Market', value: 'AWAITING DATA' }]

  return (
    <group position={[x, 0, z]} onClick={() => focusCamera('MARKET')}>
      <MarketHall />

      <group position={[0, 12, 0]}>
        <CandleChart candles={series} />
        <HoloLabel
          position={[0, CHART_HEIGHT / 2 + 2.4, 0]}
          text={`${displaySymbol(symbol)} · ${timeframe.toUpperCase()}`}
          color={THEME.hologram}
          scale={1.3}
          billboard={false}
        />
        <PriceLine candles={series} ticker={ticker?.price} />
      </group>

      <HoloPanel position={[-19, 11, 0]} rotation={[0, 0.7, 0]} title="Market" rows={rows} accent={THEME.hologram} scale={3.4} />
      <SignalBoard />
      <VolumeColumns candles={series} />
    </group>
  )
}

/** The open pavilion the chart floats inside. */
function MarketHall() {
  return (
    <group>
      <mesh position={[0, 0.3, 0]} receiveShadow>
        <boxGeometry args={[42, 0.6, 26]} />
        <meshLambertMaterial color={THEME.stoneDark} />
      </mesh>
      <mesh position={[0, 0.65, 0]} receiveShadow>
        <boxGeometry args={[36, 0.2, 20]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>

      {[[-18, -11], [18, -11], [-18, 11], [18, 11]].map(([px, pz], index) => (
        <mesh key={index} position={[px, 10, pz]} castShadow>
          <boxGeometry args={[1.6, 20, 1.6]} />
          <meshLambertMaterial color={THEME.stoneLight} />
        </mesh>
      ))}

      {/* An open frame rather than a roof: a solid slab here would occlude the
          whole centre of the world from the default overhead view. */}
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * 18, 20.6, 0]} castShadow>
          <boxGeometry args={[2.4, 1.2, 26]} />
          <meshLambertMaterial color={THEME.royal} />
        </mesh>
      ))}
      {[-1, 1].map((side) => (
        <mesh key={`z${side}`} position={[0, 20.6, side * 11]} castShadow>
          <boxGeometry args={[42, 1.2, 2.4]} />
          <meshLambertMaterial color={THEME.night} />
        </mesh>
      ))}
      <pointLight position={[0, 16, 0]} color={THEME.hologram} intensity={22} distance={46} />
    </group>
  )
}

/** Candle bodies and wicks, instanced. */
function CandleChart({ candles }: { candles: Candle[] }) {
  const bodies = useRef<THREE.InstancedMesh>(null)
  const wicks = useRef<THREE.InstancedMesh>(null)

  const visible = useMemo(() => candles.slice(-MAX_BARS), [candles])

  const bounds = useMemo(() => {
    if (visible.length === 0) return { low: 0, high: 1 }
    let low = Infinity
    let high = -Infinity
    for (const candle of visible) {
      low = Math.min(low, candle.low)
      high = Math.max(high, candle.high)
    }
    const pad = (high - low) * 0.08 || 1
    return { low: low - pad, high: high + pad }
  }, [visible])

  useLayoutEffect(() => {
    const bodyMesh = bodies.current
    const wickMesh = wicks.current
    if (!bodyMesh || !wickMesh || visible.length === 0) return

    const span = bounds.high - bounds.low || 1
    const barWidth = CHART_WIDTH / Math.max(visible.length, 1)
    const toY = (value: number) => ((value - bounds.low) / span - 0.5) * CHART_HEIGHT

    const up = new THREE.Color(THEME.profit)
    const down = new THREE.Color(THEME.loss)

    visible.forEach((candle, index) => {
      const cx = -CHART_WIDTH / 2 + barWidth * (index + 0.5)
      const openY = toY(candle.open)
      const closeY = toY(candle.close)
      const bullish = candle.close >= candle.open
      const bodyHeight = Math.max(Math.abs(closeY - openY), 0.06)

      dummy.position.set(cx, (openY + closeY) / 2, 0)
      dummy.scale.set(barWidth * 0.68, bodyHeight, barWidth * 0.68)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      bodyMesh.setMatrixAt(index, dummy.matrix)
      bodyMesh.setColorAt(index, bullish ? up : down)

      const highY = toY(candle.high)
      const lowY = toY(candle.low)
      dummy.position.set(cx, (highY + lowY) / 2, 0)
      dummy.scale.set(barWidth * 0.16, Math.max(highY - lowY, 0.06), barWidth * 0.16)
      dummy.updateMatrix()
      wickMesh.setMatrixAt(index, dummy.matrix)
      wickMesh.setColorAt(index, bullish ? up : down)
    })

    bodyMesh.count = visible.length
    wickMesh.count = visible.length
    bodyMesh.instanceMatrix.needsUpdate = true
    wickMesh.instanceMatrix.needsUpdate = true
    if (bodyMesh.instanceColor) bodyMesh.instanceColor.needsUpdate = true
    if (wickMesh.instanceColor) wickMesh.instanceColor.needsUpdate = true
    bodyMesh.computeBoundingSphere()
  }, [visible, bounds])

  if (visible.length === 0) {
    return <HoloLabel position={[0, 0, 0]} text="AWAITING MARKET DATA" color={THEME.hologram} scale={1} />
  }

  return (
    <group>
      {/* Chart backdrop */}
      <mesh position={[0, 0, -0.6]}>
        <planeGeometry args={[CHART_WIDTH + 3, CHART_HEIGHT + 3]} />
        <meshBasicMaterial color="#060a18" transparent opacity={0.55} depthWrite={false} />
      </mesh>
      {[-0.5, -0.25, 0, 0.25, 0.5].map((fraction) => (
        <mesh key={fraction} position={[0, fraction * CHART_HEIGHT, -0.5]}>
          <planeGeometry args={[CHART_WIDTH + 2, 0.03]} />
          <meshBasicMaterial color={THEME.hologram} transparent opacity={0.2} depthWrite={false} />
        </mesh>
      ))}

      <instancedMesh ref={bodies} args={[undefined, undefined, MAX_BARS]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshLambertMaterial vertexColors emissive="#ffffff" emissiveIntensity={0.16} />
      </instancedMesh>
      <instancedMesh ref={wicks} args={[undefined, undefined, MAX_BARS]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshLambertMaterial vertexColors />
      </instancedMesh>
    </group>
  )
}

/** A live marker riding the current price across the chart. */
function PriceLine({ candles, ticker }: { candles: Candle[]; ticker?: number }) {
  const marker = useRef<THREE.Mesh>(null)

  const bounds = useMemo(() => {
    const visible = candles.slice(-MAX_BARS)
    if (visible.length === 0) return null
    let low = Infinity
    let high = -Infinity
    for (const candle of visible) {
      low = Math.min(low, candle.low)
      high = Math.max(high, candle.high)
    }
    const pad = (high - low) * 0.08 || 1
    return { low: low - pad, high: high + pad }
  }, [candles])

  useFrame(({ clock }) => {
    if (!marker.current || !bounds || ticker === undefined) return
    const span = bounds.high - bounds.low || 1
    const y = ((ticker - bounds.low) / span - 0.5) * CHART_HEIGHT
    marker.current.position.y = y
    const material = marker.current.material as THREE.MeshBasicMaterial
    material.opacity = 0.55 + Math.sin(clock.elapsedTime * 4) * 0.25
  })

  if (!bounds || ticker === undefined) return null

  return (
    <mesh ref={marker} position={[0, 0, 0.3]}>
      <planeGeometry args={[CHART_WIDTH + 2, 0.09]} />
      <meshBasicMaterial color={THEME.lantern} transparent opacity={0.7} depthWrite={false} />
    </mesh>
  )
}

/** Volume as animated voxel columns beneath the chart. */
function VolumeColumns({ candles }: { candles: Candle[] }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const visible = useMemo(() => candles.slice(-40), [candles])

  useLayoutEffect(() => {
    const instanced = mesh.current
    if (!instanced || visible.length === 0) return
    const peak = Math.max(...visible.map((c) => c.volume)) || 1
    const barWidth = 24 / visible.length

    visible.forEach((candle, index) => {
      const height = Math.max(0.2, (candle.volume / peak) * 4.5)
      dummy.position.set(-12 + barWidth * (index + 0.5), 1 + height / 2, 9)
      dummy.scale.set(barWidth * 0.7, height, 0.8)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      instanced.setMatrixAt(index, dummy.matrix)
      instanced.setColorAt(
        index,
        new THREE.Color(candle.close >= candle.open ? THEME.profit : THEME.loss),
      )
    })
    instanced.count = visible.length
    instanced.instanceMatrix.needsUpdate = true
    if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true
  }, [visible])

  if (visible.length === 0) return null

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, 40]} frustumCulled={false}>
      <boxGeometry args={[1, 1, 1]} />
      <meshLambertMaterial vertexColors transparent opacity={0.75} />
    </instancedMesh>
  )
}

/** Every bot's current read on the market, side by side. */
function SignalBoard() {
  const bots = useWorldStore((s) => s.bots)
  const botOrder = useWorldStore((s) => s.botOrder)

  const rows: PanelRow[] = botOrder.map((id) => {
    const bot = bots[id]
    const signal = bot?.last_signal
    const action = signal && signal.action !== 'HOLD' ? signal.action : 'HOLD'
    return {
      label: bot?.name ?? id,
      value: action === 'HOLD' ? '—' : `${action} ${(signal!.confidence * 100).toFixed(0)}%`,
      color:
        action === 'LONG' ? THEME.profit
          : action === 'SHORT' ? THEME.loss
            : action === 'CLOSE' ? THEME.warning
              : '#8b95aa',
    }
  })

  if (rows.length === 0) return null

  return (
    <HoloPanel
      position={[19, 11, 0]}
      rotation={[0, -0.7, 0]}
      title="Bot Signals"
      rows={rows}
      accent={THEME.hologram}
      scale={3.4}
    />
  )
}

export { CHART_HEIGHT, CHART_WIDTH, plainPct }
