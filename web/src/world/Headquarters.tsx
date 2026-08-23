/**
 * JOJO'S TRADING HEADQUARTERS — the centre of the world.
 *
 * JOJO stands on the command dais with the desk-wide numbers floating around
 * him: equity, total and today's P&L, exposure, drawdown, open positions,
 * active bots. A giant ticker orbits the roof carrying live prices.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { useWorldStore } from '../state/store'
import { VoxelCharacter } from '../voxel/VoxelCharacter'
import { HoloLabel, HoloPanel, type PanelRow } from './HoloPanel'
import { RISK_COLOR, THEME } from './theme'
import { displaySymbol, money, pct, plainPct, pnlColor, price, signedMoney } from './format'

export function Headquarters() {
  const portfolio = useWorldStore((s) => s.portfolio)
  const risk = useWorldStore((s) => s.risk)
  const system = useWorldStore((s) => s.system)
  const bots = useWorldStore((s) => s.bots)
  const focusCamera = useWorldStore((s) => s.focusCamera)
  const selectBot = useWorldStore((s) => s.selectBot)

  const emergency = Boolean(risk?.emergency_stop)
  const accent = emergency ? THEME.critical : THEME.gold
  const activeBots = Object.values(bots).filter(
    (b) => b.status !== 'OFFLINE' && b.status !== 'HALTED',
  ).length

  const commandRows: PanelRow[] = [
    {
      label: 'Total Equity',
      value: portfolio ? money(portfolio.equity) : '—',
      color: '#ffffff',
      emphasis: true,
    },
    {
      label: 'Total P&L',
      value: portfolio ? signedMoney(portfolio.total_pnl) : '—',
      color: pnlColor(portfolio?.total_pnl ?? 0),
    },
    {
      label: "Today's P&L",
      value: portfolio ? signedMoney(portfolio.today_pnl) : '—',
      color: pnlColor(portfolio?.today_pnl ?? 0),
    },
    {
      label: 'Return',
      value: portfolio ? pct(portfolio.total_pnl_pct) : '—',
      color: pnlColor(portfolio?.total_pnl_pct ?? 0),
    },
  ]

  const positionRows: PanelRow[] = [
    {
      label: 'Open Positions',
      value: risk ? `${risk.open_positions}/${risk.max_open_positions}` : '—',
    },
    {
      label: 'Total Exposure',
      value: portfolio ? money(portfolio.total_exposure) : '—',
    },
    {
      label: 'Exposure',
      value: risk ? plainPct(risk.exposure_pct, 0) : '—',
      color: (risk?.exposure_pct ?? 0) > risk?.max_exposure_pct! * 0.75
        ? THEME.warning : '#f2f6ff',
    },
    {
      label: 'Drawdown',
      value: risk ? plainPct(risk.current_drawdown_pct, 2) : '—',
      color: (risk?.current_drawdown_pct ?? 0) > (risk?.drawdown_limit_pct ?? 25) * 0.6
        ? THEME.loss : '#f2f6ff',
    },
  ]

  const statusRows: PanelRow[] = [
    {
      label: 'State',
      value: emergency ? 'EMERGENCY' : 'COORDINATING',
      color: emergency ? THEME.critical : THEME.profit,
      emphasis: true,
    },
    {
      label: 'Active Bots',
      value: `${activeBots} / ${Object.keys(bots).length}`,
      color: activeBots === 0 ? THEME.loss : '#f2f6ff',
    },
    {
      label: 'Risk',
      value: risk?.level ?? '—',
      color: risk ? RISK_COLOR[risk.level] : '#f2f6ff',
    },
    {
      label: 'Execution',
      value: system?.mode ?? '—',
      color: system?.mode === 'LIVE' ? THEME.critical : THEME.hologram,
    },
  ]

  return (
    <group>
      <HQBuilding emergency={emergency} />

      {/* JOJO on the command dais */}
      <group position={[0, 3.3, 0]}>
        <VoxelCharacter
          botId="jojo"
          status={emergency ? 'HALTED' : 'IDLE'}
          risk={risk?.level ?? 'SAFE'}
          glow={accent}
          glowStrength={emergency ? 0.22 : 0.08}
          onClick={() => { focusCamera('JOJO'); selectBot(null) }}
        />
        <HoloLabel position={[0, 8.2, 0]} text="JOJO" color={accent} scale={1.5} />
        <HoloLabel
          position={[0, 7.4, 0]}
          text={emergency ? 'EMERGENCY STOP' : 'THE MASTER'}
          color={emergency ? THEME.critical : '#e8dcc0'}
          scale={0.55}
        />
        <OrbitingMarks accent={accent} paused={emergency} />
      </group>

      {/* Command boards around the dais */}
      <HoloPanel position={[-13.5, 9.6, 5]} rotation={[0, 0.6, 0]} title="Command" rows={commandRows} accent={accent} scale={3.4} />
      <HoloPanel position={[13.5, 9.6, 5]} rotation={[0, -0.6, 0]} title="Exposure" rows={positionRows} accent={accent} scale={3.4} />
      <HoloPanel position={[0, 11.4, -9]} rotation={[0, 0, 0]} title="JOJO" rows={statusRows} accent={accent} scale={3.4} />

      <GiantTicker emergency={emergency} />
      <RiskMeter />
    </group>
  )
}

/** The building itself: a tiered pagoda over a stone command hall. */
function HQBuilding({ emergency }: { emergency: boolean }) {
  const roof = emergency ? '#7a2030' : THEME.royal
  const trim = emergency ? THEME.critical : THEME.gold

  return (
    <group>
      {/* Command dais */}
      <mesh position={[0, 1.2, 0]} receiveShadow castShadow>
        <boxGeometry args={[14, 1.6, 14]} />
        <meshLambertMaterial color={THEME.stoneLight} />
      </mesh>
      <mesh position={[0, 2.3, 0]} receiveShadow castShadow>
        <boxGeometry args={[10, 0.8, 10]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>
      <mesh position={[0, 3, 0]} receiveShadow castShadow>
        <boxGeometry args={[6, 0.6, 6]} />
        <meshLambertMaterial color={trim} emissive={trim} emissiveIntensity={0.25} />
      </mesh>

      {/* Pillars */}
      {[[-6, -6], [6, -6], [-6, 6], [6, 6]].map(([x, z], index) => (
        <group key={index}>
          <mesh position={[x, 7, z]} castShadow>
            <boxGeometry args={[1.4, 10, 1.4]} />
            <meshLambertMaterial color={THEME.stoneLight} />
          </mesh>
          <mesh position={[x, 12.2, z]} castShadow>
            <boxGeometry args={[2, 0.8, 2]} />
            <meshLambertMaterial color={trim} />
          </mesh>
        </group>
      ))}

      {/* Tiered roof */}
      {[
        { y: 13.2, size: 20, thickness: 0.9 },
        { y: 15.4, size: 15, thickness: 0.8 },
        { y: 17.4, size: 10, thickness: 0.7 },
      ].map((tier, index) => (
        <group key={index}>
          <mesh position={[0, tier.y, 0]} castShadow receiveShadow>
            <boxGeometry args={[tier.size, tier.thickness, tier.size]} />
            <meshLambertMaterial color={roof} />
          </mesh>
          <mesh position={[0, tier.y + tier.thickness / 2 + 0.15, 0]} castShadow>
            <boxGeometry args={[tier.size - 1.4, 0.3, tier.size - 1.4]} />
            <meshLambertMaterial color={trim} />
          </mesh>
        </group>
      ))}

      {/* Spire */}
      <mesh position={[0, 19.6, 0]} castShadow>
        <boxGeometry args={[1.2, 4, 1.2]} />
        <meshLambertMaterial color={trim} emissive={trim} emissiveIntensity={0.6} />
      </mesh>
      <mesh position={[0, 22, 0]} castShadow>
        <boxGeometry args={[2.4, 0.5, 2.4]} />
        <meshLambertMaterial color={trim} emissive={trim} emissiveIntensity={0.8} />
      </mesh>
      <pointLight position={[0, 21, 0]} color={trim} intensity={18} distance={40} />

      {/* Steps down to the plaza */}
      {[0, 1, 2].map((i) => (
        <mesh key={i} position={[0, 0.45 - i * 0.2, 8 + i * 1.4]} receiveShadow>
          <boxGeometry args={[10 - i, 0.4, 1.6]} />
          <meshLambertMaterial color={THEME.stoneLight} />
        </mesh>
      ))}
    </group>
  )
}

/** Marks that orbit JOJO — the visual signature of orchestration. */
function OrbitingMarks({ accent, paused }: { accent: string; paused: boolean }) {
  const group = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!group.current || paused) return
    group.current.rotation.y = clock.elapsedTime * 0.55
  })
  const marks = useMemo(() => [0, 1, 2, 3, 4].map((i) => (i / 5) * Math.PI * 2), [])

  return (
    <group ref={group} position={[0, 5.4, 0]}>
      {marks.map((angle, index) => (
        <mesh key={index} position={[Math.cos(angle) * 2.2, Math.sin(angle * 2) * 0.3, Math.sin(angle) * 2.2]}>
          <boxGeometry args={[0.28, 0.28, 0.28]} />
          <meshBasicMaterial color={accent} toneMapped={false} />
        </mesh>
      ))}
    </group>
  )
}

/** The market ticker orbiting the headquarters roof. */
function GiantTicker({ emergency }: { emergency: boolean }) {
  const tickers = useWorldStore((s) => s.tickers)
  const group = useRef<THREE.Group>(null)

  useFrame(({ clock }) => {
    if (group.current) group.current.rotation.y = clock.elapsedTime * 0.12
  })

  const entries = Object.values(tickers)
  if (entries.length === 0) return null

  return (
    <group ref={group} position={[0, 24.5, 0]}>
      {entries.map((ticker, index) => {
        const angle = (index / entries.length) * Math.PI * 2
        const radius = 13
        return (
          <group
            key={ticker.symbol}
            position={[Math.cos(angle) * radius, 0, Math.sin(angle) * radius]}
            rotation={[0, -angle + Math.PI / 2, 0]}
          >
            <HoloPanel
              position={[0, 0, 0]}
              title={displaySymbol(ticker.symbol)}
              rows={[
                { label: 'Price', value: price(ticker.price), emphasis: true },
                {
                  label: '24h',
                  value: pct(ticker.change_24h_pct),
                  color: pnlColor(ticker.change_24h_pct),
                },
              ]}
              accent={emergency ? THEME.critical : THEME.hologram}
              scale={2.6}
            />
          </group>
        )
      })}
    </group>
  )
}

/** The giant risk meter beside JOJO — one glance tells you how hot the desk is. */
function RiskMeter() {
  const risk = useWorldStore((s) => s.risk)
  const bar = useRef<THREE.Mesh>(null)
  const utilization = risk?.utilization ?? 0
  const color = risk ? RISK_COLOR[risk.level] : THEME.profit
  const segments = 12
  const lit = Math.round(utilization * segments)

  useFrame(({ clock }) => {
    if (!bar.current) return
    // The top of the meter pulses when the desk is in danger.
    const material = bar.current.material as THREE.MeshLambertMaterial
    const danger = risk?.level === 'CRITICAL' || risk?.level === 'HIGH'
    material.emissiveIntensity = danger
      ? 0.6 + Math.sin(clock.elapsedTime * 8) * 0.4
      : 0.35
  })

  return (
    <group position={[-16, 0, 10]}>
      <mesh position={[0, 0.6, 0]} receiveShadow castShadow>
        <boxGeometry args={[3.4, 1.2, 3.4]} />
        <meshLambertMaterial color={THEME.stoneDark} />
      </mesh>
      <mesh position={[0, 1.4, 0]} castShadow>
        <boxGeometry args={[2.2, 0.6, 2.2]} />
        <meshLambertMaterial color={THEME.stone} />
      </mesh>

      {/* Segmented column */}
      {Array.from({ length: segments }, (_, i) => (
        <mesh key={i} position={[0, 2.2 + i * 0.85, 0]} castShadow>
          <boxGeometry args={[1.5, 0.66, 1.5]} />
          <meshLambertMaterial
            color={i < lit ? color : '#2a3040'}
            emissive={i < lit ? color : '#000000'}
            emissiveIntensity={i < lit ? 0.5 : 0}
          />
        </mesh>
      ))}

      <mesh ref={bar} position={[0, 2.2 + segments * 0.85, 0]} castShadow>
        <boxGeometry args={[2, 0.7, 2]} />
        <meshLambertMaterial color={color} emissive={color} emissiveIntensity={0.35} />
      </mesh>

      <HoloLabel position={[0, 2.2 + segments * 0.85 + 1.6, 0]} text="RISK" color={color} scale={1} />
      <HoloPanel
        position={[0, 4.6, 2.6]}
        title="Risk Center"
        rows={[
          { label: 'Level', value: risk?.level ?? '—', color, emphasis: true },
          { label: 'Utilization', value: plainPct((risk?.utilization ?? 0) * 100, 0), color },
          {
            label: 'Drawdown Limit',
            value: risk ? plainPct(risk.drawdown_limit_pct, 0) : '—',
          },
        ]}
        accent={color}
        scale={2.6}
      />
    </group>
  )
}
