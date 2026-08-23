/**
 * Reconnecting WebSocket client.
 *
 * A dropped connection is a loss of trading visibility, so reconnect is
 * aggressive at first and backs off only as failures persist. On every
 * reconnect the server re-sends a full snapshot, so no resync logic is needed
 * here — the store simply overwrites.
 */

import type { Frame } from './protocol'
import { useWorldStore } from '../state/store'

const INITIAL_BACKOFF_MS = 400
const MAX_BACKOFF_MS = 10_000
const HEARTBEAT_MS = 20_000
/** No frame at all for this long means the link is dead even if the socket says otherwise. */
const STALE_MS = 45_000

export function websocketUrl(): string {
  const override = new URLSearchParams(window.location.search).get('backend')
  if (override) return override.replace(/^http/, 'ws').replace(/\/$/, '') + '/ws'
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}

export class WorldSocket {
  private socket: WebSocket | null = null
  private backoff = INITIAL_BACKOFF_MS
  private heartbeatTimer: number | null = null
  private staleTimer: number | null = null
  private lastFrameAt = 0
  private closedByUs = false

  connect(): void {
    this.closedByUs = false
    const store = useWorldStore.getState()
    store.setConnection(this.backoff === INITIAL_BACKOFF_MS ? 'connecting' : 'reconnecting')

    let socket: WebSocket
    try {
      socket = new WebSocket(websocketUrl())
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = socket

    socket.onopen = () => {
      this.backoff = INITIAL_BACKOFF_MS
      this.lastFrameAt = Date.now()
      useWorldStore.getState().setConnection('live')
      this.startTimers()
    }

    socket.onmessage = (event) => {
      this.lastFrameAt = Date.now()
      let frame: Frame
      try {
        frame = JSON.parse(event.data as string) as Frame
      } catch {
        return
      }
      // Applied synchronously: trading state must never wait on a render.
      useWorldStore.getState().applyFrame(frame)
    }

    socket.onerror = () => {
      // onclose always follows; reconnect is handled there.
    }

    socket.onclose = () => {
      this.stopTimers()
      this.socket = null
      if (this.closedByUs) {
        useWorldStore.getState().setConnection('offline')
        return
      }
      useWorldStore.getState().setConnection('reconnecting')
      this.scheduleReconnect()
    }
  }

  private startTimers(): void {
    this.stopTimers()
    this.heartbeatTimer = window.setInterval(() => {
      this.send({ type: 'ping', data: {} })
    }, HEARTBEAT_MS)
    this.staleTimer = window.setInterval(() => {
      if (Date.now() - this.lastFrameAt > STALE_MS) {
        // Silent socket: force a reconnect rather than show stale prices.
        this.socket?.close()
      }
    }, STALE_MS / 2)
  }

  private stopTimers(): void {
    if (this.heartbeatTimer !== null) window.clearInterval(this.heartbeatTimer)
    if (this.staleTimer !== null) window.clearInterval(this.staleTimer)
    this.heartbeatTimer = null
    this.staleTimer = null
  }

  private scheduleReconnect(): void {
    const delay = this.backoff
    this.backoff = Math.min(this.backoff * 2, MAX_BACKOFF_MS)
    window.setTimeout(() => {
      if (!this.closedByUs) this.connect()
    }, delay)
  }

  send(message: { type: string; data?: unknown }): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false
    this.socket.send(JSON.stringify({ data: {}, ...message }))
    return true
  }

  command(botId: string, action: string): boolean {
    return this.send({ type: `bot.${action}`, data: { bot_id: botId } })
  }

  emergencyStop(): boolean {
    return this.send({ type: 'system.emergency_stop' })
  }

  resume(): boolean {
    return this.send({ type: 'system.resume' })
  }

  requestSnapshot(): boolean {
    return this.send({ type: 'snapshot.request' })
  }

  disconnect(): void {
    this.closedByUs = true
    this.stopTimers()
    this.socket?.close()
    this.socket = null
  }
}

export const worldSocket = new WorldSocket()
