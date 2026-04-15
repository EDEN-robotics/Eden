// Supabase broadcast bridge connecting Chat → Simulator.
// Every time EDEN emits a non-"none" [ACTION] tag, Chat broadcasts on this
// channel; the /sim page subscribes and feeds it to the physics engine.

import { supabase } from './supabase'

export const SIM_CHANNEL = 'eden-action-bus'
export const SIM_EVENT = 'eden-action'

// Sentinel channel — callers should hold onto the returned object and call
// cleanup() on unmount.
export function openSimBusSender() {
  const channel = supabase.channel(SIM_CHANNEL, { config: { broadcast: { self: false } } })
  let subscribed = false
  channel.subscribe((status) => {
    if (status === 'SUBSCRIBED') subscribed = true
  })
  return {
    broadcast(action, meta = {}) {
      if (!subscribed) return false
      channel.send({
        type: 'broadcast',
        event: SIM_EVENT,
        payload: { action, meta, ts: Date.now() },
      })
      return true
    },
    close() { supabase.removeChannel(channel) },
  }
}

export function openSimBusReceiver(onAction) {
  const channel = supabase.channel(SIM_CHANNEL, { config: { broadcast: { self: false } } })
  channel.on('broadcast', { event: SIM_EVENT }, ({ payload }) => {
    if (payload?.action) onAction(payload)
  })
  channel.subscribe()
  return () => supabase.removeChannel(channel)
}

// Parser: natural language + ROS-2 style /cmd_vel → { linear, angular, duration, raw }
// Returns null if nothing parseable.
export function parseAction(raw) {
  if (!raw) return null
  const s = String(raw).trim().toLowerCase()

  // /cmd_vel linear.x=0.3 angular.z=0.5  (optionally duration=3)
  if (s.startsWith('/cmd_vel')) {
    const lx = parseFloat((s.match(/linear\.x\s*=\s*(-?[\d.]+)/) || [])[1])
    const az = parseFloat((s.match(/angular\.z\s*=\s*(-?[\d.]+)/) || [])[1])
    const dur = parseFloat((s.match(/duration\s*=\s*(-?[\d.]+)/) || [])[1])
    return {
      linear: isFinite(lx) ? lx : 0,
      angular: isFinite(az) ? az : 0,
      duration: isFinite(dur) ? dur : 2.0,
      raw,
    }
  }

  // forward / move forward / go forward
  if (/\b(forward|ahead|straight)\b/.test(s)) {
    const speed = extractNumber(s, 'at', 'speed') ?? 0.35
    return { linear: speed, angular: 0, duration: extractNumber(s, 'for', 's', 'seconds') ?? 2.5, raw }
  }
  if (/\b(backward|back|reverse)\b/.test(s)) {
    const speed = extractNumber(s, 'at', 'speed') ?? 0.3
    return { linear: -speed, angular: 0, duration: extractNumber(s, 'for', 's') ?? 2.0, raw }
  }

  // turn left [N deg]
  const turnLeft = s.match(/turn\s+left(?:\s+(\d+))?/)
  if (turnLeft) {
    const deg = turnLeft[1] ? parseFloat(turnLeft[1]) : 90
    const angular = 0.9 // rad/s
    const duration = Math.abs((deg * Math.PI) / 180) / angular
    return { linear: 0, angular, duration, raw }
  }
  const turnRight = s.match(/turn\s+right(?:\s+(\d+))?/)
  if (turnRight) {
    const deg = turnRight[1] ? parseFloat(turnRight[1]) : 90
    const angular = -0.9
    const duration = Math.abs((deg * Math.PI) / 180) / Math.abs(angular)
    return { linear: 0, angular, duration, raw }
  }

  // spin / rotate
  if (/\b(spin|rotate)\b/.test(s)) {
    return { linear: 0, angular: 1.2, duration: 3.0, raw }
  }

  // stop / halt
  if (/\b(stop|halt|freeze|brake)\b/.test(s)) {
    return { linear: 0, angular: 0, duration: 0.1, raw, stop: true }
  }

  // patrol — circle pattern
  if (/\b(patrol|orbit|circle)\b/.test(s)) {
    return { linear: 0.25, angular: 0.5, duration: 6.0, raw }
  }

  // fallback — nothing matched
  return null
}

function extractNumber(s, ...markers) {
  for (const m of markers) {
    const re = new RegExp(`${m}\\s+(-?[\\d.]+)`, 'i')
    const match = s.match(re)
    if (match) return parseFloat(match[1])
  }
  return null
}
