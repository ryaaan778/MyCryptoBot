/**
 * One bot's district: their building, their statue, their character, and the
 * live numbers for their book.
 *
 * The district *is* the status display. A profitable bot's ground glows warm
 * and gold motes rise from it; a losing one desaturates; a high-risk one gains
 * a pulsing warning ring; an offline one goes dark and its lanterns die.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import type { BotState, Position } from '../net/protocol'
import { useWorldStore } from '../state/store'
import { VoxelCharacter } from '../voxel/VoxelCharacter'
import { HoloLabel, HoloPanel, type PanelRow } from './HoloPanel'
import { RISK_COLOR, THEME } from './theme'
import {
  displaySymbol,
  money,
  pct,
  plainPct,
  pnlColor,
  price,
  quantity,
  signedMoney,
} from './format'
import { StoneLantern } from './props/Scatter'

export interface BotDistrictProps {
  bot: BotState
  index: number
}

export function BotDistrict({ bot, index }: BotDistrictProps) {
  const positions = useWorldStore((s) => s.positions)
  const tickers = useWorldStore((s) => s.tickers)
  const selectBot = useWorldStore((s) => s.selectBot)
  const focusCamera = useWorldStore((s) => s.focusCamera)
  const selected = useWorldStore((s) => s.selectedBot === bot.id)

  const [x, z] = bot.persona.district
  const accent = bot.persona.accent || THEME.hologram
  const primary = bot.persona.palette[0] ?? THEME.stone
  const secondary = bot.persona.palette[3] ?? THEME.stoneDark

  const offline = bot.status === 'OFFLINE' || bot.status === 'HALTED'
  const profitable = bot.total_pnl > 0
  const losing = bot.total_pnl < 0
  const danger = bot.risk_level === 'HIGH' || bot.risk_level === 'CRITICAL'

  const botPositions = useMemo(
    () => Object.values(positions).filter((p) => p.bot_id === bot.id),
    [positions, bot.id],
  )
  const ticker = tickers[bot.symbol]

  const statusRows: PanelRow[] = [
    {
      label: 'P&L',
      value: signedMoney(bot.total_pnl),
      color: pnlColor(bot.total_pnl),
      emphasis: true,
    },
    { label: 'Strategy', value: bot.strategy.replace(/_/g, ' ').toUpperCase() },
    { label: 'Symbol', value: `${displaySymbol(bot.symbol)} ${bot.timeframe}` },
    {
      label: 'Status',
      value: bot.status,
      color: offline ? THEME.loss : bot.status === 'TRADING' ? THEME.warning : THEME.profit,
    },
  ]

  const bookRows: PanelRow[] = [
    {
      label: 'Position',
      value: botPositions.length
        ? `${botPositions[0].side} ${quantity(botPositions[0].quantity)}`
        : 'FLAT',
      color: botPositions.length
        ? botPositions[0].side === 'LONG' ? THEME.profit : THEME.loss
        : '#9aa4bb',
    },
    {
      label: 'Entry',
      value: botPositions.length ? price(botPositions[0].entry_price) : '—',
    },
    {
      label: 'Mark',
      value: ticker ? price(ticker.price) : '—',
    },
    {
      label: 'Unrealized',
      value: signedMoney(bot.unrealized_pnl),
      color: pnlColor(bot.unrealized_pnl),
    },
  ]

  const statsRows: PanelRow[] = [
    { label: 'Win Rate', value: plainPct(bot.win_rate, 1) },
    { label: 'Trades', value: `${bot.trades_won}/${bot.trades_total}` },
    { label: 'Exposure', value: money(bot.exposure) },
    {
      label: 'Risk',
      value: bot.risk_level,
      color: RISK_COLOR[bot.risk_level],
    },
    { label: 'Leverage', value: `${bot.leverage.toFixed(0)}x` },
  ]

  return (
    <group position={[x, 0, z]} rotation={[0, bot.persona.facing, 0]}>
      <DistrictGround
        accent={accent}
        profitable={profitable}
        losing={losing}
        offline={offline}
        selected={selected}
      />

      <DistrictBuilding primary={primary} secondary={secondary} accent={accent} dim={offline} />

      {/* The character on their plinth */}
      <group position={[0, 2.1, 6]}>
        <mesh position={[0, -0.9, 0]} receiveShadow castShadow>
          <boxGeometry args={[5.4, 1.8, 5.4]} />
          <meshLambertMaterial color={offline ? '#33383f' : THEME.stone} />
        </mesh>
        <mesh position={[0, 0.05, 0]} receiveShadow>
          <boxGeometry args={[4.4, 0.2, 4.4]} />
          <meshLambertMaterial
            color={offline ? '#22262c' : accent}
            emissive={offline ? '#000000' : accent}
            emissiveIntensity={offline ? 0 : 0.35}
          />
        </mesh>

        <VoxelCharacter
          botId={bot.id}
          status={bot.status}
          risk={bot.risk_level}
          glow={profitable ? THEME.gold : danger ? THEME.critical : accent}
          // Kept low on purpose: emissive is added uniformly to every voxel,
          // so a strong glow drowns the character's own palette. Profit and
          // danger are carried by the district colour, aura and particles.
          glowStrength={offline ? 0 : profitable ? 0.12 : danger ? 0.16 : 0.04}
          dimmed={offline}
          phase={index * 1.7}
          onClick={() => { selectBot(bot.id); focusCamera(bot.id) }}
        />

        <HoloLabel
          position={[0, 7.4, 0]}
          text={bot.name}
          color={offline ? '#7d8798' : accent}
          scale={1.15}
        />
        <HoloLabel
          position={[0, 6.6, 0]}
          text={bot.persona.title}
          color={offline ? '#5d6675' : '#d8dfee'}
          scale={0.45}
        />
        {bot.total_pnl !== 0 && (
          <FloatingPnl value={bot.total_pnl} />
        )}
      </group>

      {/* Live boards */}
      <HoloPanel position={[-10.5, 7.2, 2]} rotation={[0, 0.6, 0]} title={bot.name} rows={statusRows} accent={accent} scale={2.8} />
      <HoloPanel position={[10.5, 7.2, 2]} rotation={[0, -0.6, 0]} title="Book" rows={bookRows} accent={accent} scale={2.8} />
      <HoloPanel position={[0, 9.6, -6.5]} title="Performance" rows={statsRows} accent={accent} scale={2.8} />

      {bot.last_signal && bot.last_signal.action !== 'HOLD' && !offline && (
        <HoloPanel
          position={[0, 3.2, 12]}
          title="Signal"
          rows={[
            {
              label: bot.last_signal.action,
              value: `${(bot.last_signal.confidence * 100).toFixed(0)}%`,
              color:
                bot.last_signal.action === 'LONG'
                  ? THEME.profit
                  : bot.last_signal.action === 'SHORT'
                    ? THEME.loss
                    : THEME.warning,
              emphasis: true,
            },
          ]}
          accent={accent}
          scale={1.9}
          billboard
        />
      )}

      {danger && !offline && <WarningRing color={RISK_COLOR[bot.risk_level]} />}
      {botPositions.length > 0 && <PositionMarkers positions={botPositions} />}

      <StoneLantern position={[-8, 0, 10]} lit={!offline} />
      <StoneLantern position={[8, 0, 10]} lit={!offline} />
    </group>
  )
}

/** District plaza — its colour is the bot's P&L. */
function DistrictGround({
  accent, profitable, losing, offline, selected,
}: {
  accent: string
  profitable: boolean
  losing: boolean
  offline: boolean
  selected: boolean
}) {
  const ring = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (!ring.current || !selected) return
    const material = ring.current.material as THREE.MeshBasicMaterial
    material.opacity = 0.35 + Math.sin(clock.elapsedTime * 3) * 0.2
  })

  const plaza = offline ? '#2b2f36' : profitable ? '#4d7a52' : losing ? '#6b4048' : THEME.grassDeep

  return (
    <group>
      <mesh position={[0, 0.18, 0]} receiveShadow>
        <boxGeometry args={[28, 0.36, 28]} />
        <meshLambertMaterial color={plaza} />
      </mesh>
      <mesh position={[0, 0.4, 0]} receiveShadow>
        <boxGeometry args={[21, 0.12, 21]} />
        <meshLambertMaterial color={offline ? '#3a3f47' : THEME.stone} />
      </mesh>
      <mesh position={[0, 0.48, 0]} receiveShadow>
        <boxGeometry args={[10, 0.1, 10]} />
        <meshLambertMaterial
          color={offline ? '#2e3238' : accent}
          emissive={offline ? '#000000' : accent}
          emissiveIntensity={offline ? 0 : 0.22}
        />
      </mesh>
      {selected && (
        <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.56, 0]}>
          <ringGeometry args={[13, 14.4, 48]} />
          <meshBasicMaterial color={accent} transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  )
}

/** Each district gets its own architecture so it reads from a distance. */
function DistrictBuilding({
  primary, secondary, accent, dim,
}: {
  primary: string
  secondary: string
  accent: string
  dim: boolean
}) {
  const wall = dim ? '#3c4048' : primary
  const roof = dim ? '#2a2d33' : secondary

  return (
    <group position={[0, 0, -6]}>
      <mesh position={[0, 3.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[12, 6, 9]} />
        <meshLambertMaterial color={wall} />
      </mesh>
      <mesh position={[0, 6.6, 0]} castShadow>
        <boxGeometry args={[14, 0.9, 11]} />
        <meshLambertMaterial color={roof} />
      </mesh>
      <mesh position={[0, 7.6, 0]} castShadow>
        <boxGeometry args={[10, 0.8, 8]} />
        <meshLambertMaterial color={roof} />
      </mesh>
      <mesh position={[0, 8.3, 0]} castShadow>
        <boxGeometry args={[1, 1.4, 1]} />
        <meshLambertMaterial
          color={dim ? '#3a3f47' : accent}
          emissive={dim ? '#000000' : accent}
          emissiveIntensity={dim ? 0 : 0.7}
        />
      </mesh>

      {/* Windows: dark when the bot is offline */}
      {[-4, 0, 4].map((wx) => (
        <mesh key={wx} position={[wx, 3.6, 4.6]}>
          <boxGeometry args={[2.4, 2.4, 0.2]} />
          <meshLambertMaterial
            color={dim ? '#1c1f24' : THEME.lantern}
            emissive={dim ? '#000000' : THEME.lantern}
            emissiveIntensity={dim ? 0 : 0.55}
          />
        </mesh>
      ))}

      {/* Support beams */}
      {[-5.6, 5.6].map((bx) => (
        <mesh key={bx} position={[bx, 3.2, 4.4]} castShadow>
          <boxGeometry args={[0.7, 6.4, 0.7]} />
          <meshLambertMaterial color={THEME.woodDark} />
        </mesh>
      ))}
    </group>
  )
}

/** Rising P&L number above the character. */
function FloatingPnl({ value }: { value: number }) {
  const group = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (group.current) {
      group.current.position.y = 5.6 + Math.sin(clock.elapsedTime * 1.4) * 0.16
    }
  })
  return (
    <group ref={group}>
      <HoloLabel position={[0, 0, 0]} text={signedMoney(value)} color={pnlColor(value)} scale={0.75} />
    </group>
  )
}

/** Pulsing hazard ring for a bot near its limits. */
function WarningRing({ color }: { color: string }) {
  const inner = useRef<THREE.Mesh>(null)
  const outer = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (inner.current) {
      const scale = 1 + (t % 1.6) / 1.6
      inner.current.scale.setScalar(scale)
      const material = inner.current.material as THREE.MeshBasicMaterial
      material.opacity = 0.55 * (1 - (t % 1.6) / 1.6)
    }
    if (outer.current) {
      const material = outer.current.material as THREE.MeshBasicMaterial
      material.opacity = 0.28 + Math.sin(t * 6) * 0.16
    }
  })

  return (
    <group position={[0, 0.62, 6]}>
      <mesh ref={inner} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3, 3.5, 40]} />
        <meshBasicMaterial color={color} transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={outer} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[5.4, 6.2, 44]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      <pointLight position={[0, 2, 0]} color={color} intensity={9} distance={16} />
    </group>
  )
}

/** A column per open position, height scaled by unrealised P&L. */
function PositionMarkers({ positions }: { positions: Position[] }) {
  return (
    <group position={[0, 0.6, 13]}>
      {positions.map((position, index) => {
        const magnitude = Math.min(6, Math.abs(position.unrealized_pnl_pct) / 10 + 0.6)
        const color = pnlColor(position.unrealized_pnl)
        return (
          <group key={position.id} position={[(index - (positions.length - 1) / 2) * 2.6, 0, 0]}>
            <mesh position={[0, magnitude / 2, 0]} castShadow>
              <boxGeometry args={[1.2, magnitude, 1.2]} />
              <meshLambertMaterial color={color} emissive={color} emissiveIntensity={0.4} />
            </mesh>
            <HoloLabel
              position={[0, magnitude + 0.7, 0]}
              text={`${position.side} ${pct(position.unrealized_pnl_pct, 1)}`}
              color={color}
              scale={0.42}
            />
          </group>
        )
      })}
    </group>
  )
}
