// Bridge connecting Chat → Simulator across tabs.
//
// Uses TWO transports:
//  1. window localStorage 'storage' events — reliable cross-tab within
//     the same browser. This is the primary path for the demo flow.
//  2. Supabase broadcast — covers the cross-device case (two people on
//     different laptops).
//
// Both carry the same payload shape: { action, meta, ts }.

import { supabase } from './supabase'

export const SIM_CHANNEL = 'eden-action-bus'
export const SIM_EVENT = 'eden-action'
const LS_KEY = 'eden:action:latest'

// ───── Sender ─────
export function openSimBusSender() {
  const channel = supabase.channel(SIM_CHANNEL, { config: { broadcast: { self: false } } })
  let subscribed = false
  const pending = []

  channel.subscribe((status) => {
    console.log('[simBridge:sender] supabase status →', status)
    if (status === 'SUBSCRIBED') {
      subscribed = true
      // flush any queued messages
      while (pending.length) {
        const p = pending.shift()
        channel.send({ type: 'broadcast', event: SIM_EVENT, payload: p })
      }
    }
  })

  return {
    broadcast(action, meta = {}) {
      const payload = { action, meta, ts: Date.now(), nonce: Math.random().toString(36).slice(2, 10) }
      console.log('[simBridge:sender] dispatch', payload)

      // Primary transport: localStorage (cross-tab, synchronous, reliable)
      try {
        window.localStorage.setItem(LS_KEY, JSON.stringify(payload))
      } catch (err) {
        console.warn('[simBridge:sender] localStorage failed:', err)
      }

      // Secondary transport: supabase broadcast (cross-device)
      if (subscribed) {
        channel.send({ type: 'broadcast', event: SIM_EVENT, payload })
      } else {
        pending.push(payload)
      }
      return true
    },
    close() { supabase.removeChannel(channel) },
  }
}

// ───── Receiver ─────
export function openSimBusReceiver(onAction) {
  const seen = new Set() // dedup by nonce (both transports may fire)

  function deliver(payload) {
    if (!payload?.action) return
    const id = payload.nonce || `${payload.ts}:${payload.action}`
    if (seen.has(id)) return
    seen.add(id)
    // Garbage-collect old ids
    if (seen.size > 200) {
      const arr = Array.from(seen)
      arr.slice(0, 100).forEach((k) => seen.delete(k))
    }
    console.log('[simBridge:receiver] deliver', payload)
    onAction(payload)
  }

  // Primary: storage events
  function onStorage(e) {
    if (e.key !== LS_KEY || !e.newValue) return
    try { deliver(JSON.parse(e.newValue)) } catch {}
  }
  window.addEventListener('storage', onStorage)

  // Secondary: supabase broadcast
  const channel = supabase.channel(SIM_CHANNEL, { config: { broadcast: { self: false } } })
  channel.on('broadcast', { event: SIM_EVENT }, ({ payload }) => {
    console.log('[simBridge:receiver] supabase payload', payload)
    deliver(payload)
  })
  channel.subscribe((status) => {
    console.log('[simBridge:receiver] supabase status →', status)
  })

  return () => {
    window.removeEventListener('storage', onStorage)
    supabase.removeChannel(channel)
  }
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

  // Looser matches for whatever EDEN improvises in [ACTION]
  if (/\b(drive|move|go|head|navigate|proceed|advance|approach)\b/.test(s)) {
    if (/\b(back|reverse)\b/.test(s)) {
      return { linear: -0.3, angular: 0, duration: 2.0, raw }
    }
    if (/\bleft\b/.test(s)) {
      return { linear: 0.25, angular: 0.45, duration: 2.5, raw }
    }
    if (/\bright\b/.test(s)) {
      return { linear: 0.25, angular: -0.45, duration: 2.5, raw }
    }
    return { linear: 0.35, angular: 0, duration: 2.5, raw }
  }
  if (/\b(look|scan|inspect|observe)\b/.test(s)) {
    return { linear: 0, angular: 0.6, duration: 3.0, raw }
  }
  if (/\b(come|approach|here)\b/.test(s)) {
    return { linear: 0.3, angular: 0, duration: 2.0, raw }
  }
  if (/\bwait\b/.test(s)) {
    return { linear: 0, angular: 0, duration: 0.1, stop: true, raw }
  }

  // Last-resort: any directional verb with a number → treat number as duration
  const durMatch = s.match(/(-?[\d.]+)/)
  if (durMatch && /\b(second|sec|s)\b/.test(s)) {
    return { linear: 0.3, angular: 0, duration: parseFloat(durMatch[1]), raw }
  }

  console.warn('[simBridge] unmapped action:', raw)
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
