/**
 * Application shell: opens the socket, decides between the world and the 2D
 * fallback, and renders the HUD over whichever won.
 */

import { useEffect, useMemo, useState } from 'react'

import { worldSocket } from './net/ws'
import { useWorldStore } from './state/store'
import { World } from './world/World'
import { Hud } from './hud/Hud'
import { Fallback2D } from './fallback/Fallback2D'
import { detectWebGL, forcedRenderMode, initialQuality } from './perf/quality'
import './hud/hud.css'

export function App() {
  const renderMode = useWorldStore((s) => s.renderMode)
  const setRenderMode = useWorldStore((s) => s.setRenderMode)
  const setQuality = useWorldStore((s) => s.setQuality)
  const connection = useWorldStore((s) => s.connection)
  const system = useWorldStore((s) => s.system)

  const [webglAvailable] = useState(detectWebGL)
  const forced = useMemo(forcedRenderMode, [])

  // Decide the initial render mode once, before the first paint of the world.
  useEffect(() => {
    if (forced === '2d') {
      setRenderMode('2d')
      return
    }
    if (!webglAvailable) {
      setRenderMode('2d')
      return
    }
    if (forced === '3d') setRenderMode('3d')
    setQuality(initialQuality())
  }, [forced, webglAvailable, setRenderMode, setQuality])

  useEffect(() => {
    worldSocket.connect()
    return () => worldSocket.disconnect()
  }, [])

  // Keyboard shortcuts for the things a desk operator reaches for constantly.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return
      const store = useWorldStore.getState()
      if (event.key === 'Escape') store.selectBot(null)
      if (event.key === 't' || event.key === 'T') store.toggleTables()
      if (event.key === 'w' || event.key === 'W') store.focusCamera('WORLD')
      if (event.key === 'm' || event.key === 'M') store.focusCamera('MARKET')
      if (event.key === 'j' || event.key === 'J') store.focusCamera('JOJO')
      if (event.key >= '1' && event.key <= '5') {
        const id = store.botOrder[Number(event.key) - 1]
        if (id) { store.selectBot(id); store.focusCamera(id) }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const fallbackReason = !webglAvailable
    ? 'WebGL is unavailable in this browser, so the voxel world cannot render. Every trading control remains available here.'
    : forced === '2d'
      ? 'Simplified view requested via ?mode=2d. All trading data and controls are live.'
      : 'The voxel world could not hold a usable frame rate, so it was set aside. Trading visibility is never traded for graphics.'

  const booting = connection === 'connecting' && !system

  return (
    <div className="app">
      {renderMode === '3d' ? (
        <>
          <div className="canvas-layer">
            <World />
          </div>
          <Hud />
        </>
      ) : (
        <Fallback2D reason={fallbackReason} />
      )}

      {booting && (
        <div className="loading">
          <div className="loading__title">JOJO</div>
          <div className="loading__sub">CONNECTING TO THE TRADING ENGINE…</div>
        </div>
      )}
    </div>
  )
}
