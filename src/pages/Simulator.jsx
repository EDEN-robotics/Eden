import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, ArrowLeft, Power, RotateCcw, Play, Pause, Wifi, Send } from 'lucide-react'
import { openSimBusReceiver, parseAction } from '../lib/simBridge'
import "@fontsource/jetbrains-mono"

// ───── World constants ─────
const WORLD = { w: 10, h: 7 } // meters
const ROBOT_RADIUS = 0.22     // meters
const OBSTACLES = [
  { x: 2.5, y: 2.0, w: 0.6, h: 0.6, label: 'crate' },
  { x: 6.0, y: 1.5, w: 1.2, h: 0.4, label: 'wall' },
  { x: 7.5, y: 4.5, w: 0.8, h: 0.8, label: 'box' },
  { x: 3.5, y: 5.0, w: 0.4, h: 1.2, label: 'pillar' },
  { x: 5.5, y: 3.5, w: 0.3, h: 0.3, label: 'marker' },
]

// Pixels per meter — computed dynamically from canvas size
function ppm(canvasW) { return canvasW / WORLD.w }

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)) }

// ───── Main component ─────
export default function Simulator() {
  const canvasRef = useRef(null)
  const stateRef = useRef({
    x: 1.0, y: WORLD.h / 2, heading: 0, // rad (0 = +x)
    linVel: 0, angVel: 0,                // current velocity
    cmdUntil: 0,                          // timestamp when current cmd expires
    trail: [],                            // [{x,y,ts}]
    lastTick: performance.now(),
    collisions: 0,
  })
  const [telemetry, setTelemetry] = useState({ x: 1, y: 3.5, heading: 0, linVel: 0, angVel: 0 })
  const [log, setLog] = useState([]) // [{ source, action, parsed, ts }]
  const [running, setRunning] = useState(true)
  const [manualInput, setManualInput] = useState('')
  const [connected, setConnected] = useState(false)

  // Subscribe to the action bus
  useEffect(() => {
    const close = openSimBusReceiver((payload) => {
      applyAction(payload.action, 'eden')
    })
    setConnected(true)
    return () => { close(); setConnected(false) }
  }, [])

  function applyAction(rawAction, source) {
    const parsed = parseAction(rawAction)
    const logEntry = { source, action: rawAction, parsed, ts: Date.now() }
    setLog((prev) => [logEntry, ...prev].slice(0, 40))
    console.log('[sim] applyAction', logEntry)

    if (!parsed) {
      // Unmapped — nudge forward a bit so the demo doesn't look dead
      const s = stateRef.current
      s.linVel = 0.15
      s.angVel = 0
      s.cmdUntil = performance.now() + 800
      return
    }
    const s = stateRef.current
    if (parsed.stop) {
      s.linVel = 0; s.angVel = 0; s.cmdUntil = performance.now() + 100
    } else {
      s.linVel = parsed.linear
      s.angVel = parsed.angular
      s.cmdUntil = performance.now() + parsed.duration * 1000
    }
  }

  // Keyboard manual control
  useEffect(() => {
    function onKey(e) {
      if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return
      const map = {
        ArrowUp:    'forward',
        ArrowDown:  'backward',
        ArrowLeft:  'turn left 30',
        ArrowRight: 'turn right 30',
        Space:      'stop',
        KeyR:       null,
      }
      const action = map[e.code] ?? (e.key === ' ' ? 'stop' : null)
      if (action) { e.preventDefault(); applyAction(action, 'manual') }
      if (e.code === 'KeyR') {
        const s = stateRef.current
        s.x = 1.0; s.y = WORLD.h / 2; s.heading = 0
        s.linVel = 0; s.angVel = 0; s.cmdUntil = 0; s.trail = []; s.collisions = 0
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Physics + render loop
  useEffect(() => {
    let raf = 0
    function tick(now) {
      const s = stateRef.current
      const dt = Math.min(0.05, (now - s.lastTick) / 1000) // cap dt for tab-switch
      s.lastTick = now

      if (running) {
        // expire command
        if (now > s.cmdUntil) { s.linVel = 0; s.angVel = 0 }

        // integrate
        const nextX = s.x + s.linVel * Math.cos(s.heading) * dt
        const nextY = s.y + s.linVel * Math.sin(s.heading) * dt
        const nextHeading = s.heading + s.angVel * dt

        // collision check against obstacles (bbox vs circle)
        const collided = OBSTACLES.some((o) => circleIntersectsRect(nextX, nextY, ROBOT_RADIUS, o))
          || nextX < ROBOT_RADIUS || nextX > WORLD.w - ROBOT_RADIUS
          || nextY < ROBOT_RADIUS || nextY > WORLD.h - ROBOT_RADIUS
        if (!collided) {
          s.x = nextX; s.y = nextY
        } else {
          s.linVel = 0 // halt on collision
          s.collisions += 1
        }
        s.heading = nextHeading

        // trail
        if (Math.abs(s.linVel) > 0.02 || Math.abs(s.angVel) > 0.05) {
          s.trail.push({ x: s.x, y: s.y, ts: now })
          if (s.trail.length > 600) s.trail.shift()
        }
      }

      render(canvasRef.current, s, now)
      setTelemetry({ x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running])

  function handleManualSubmit(e) {
    e?.preventDefault()
    if (!manualInput.trim()) return
    applyAction(manualInput.trim(), 'manual')
    setManualInput('')
  }

  const latest = log[0]

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-4 px-6 py-4 border-b border-white/10 bg-black/80 backdrop-blur">
        <Link to="/" className="flex items-center gap-2 text-xs text-white/50 hover:text-white transition-colors">
          <ArrowLeft size={12} /> Home
        </Link>
        <div className="h-4 w-px bg-white/10" />
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-rose-400" />
          <span className="text-sm font-semibold">EDEN Simulator</span>
          <span className="text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-400/30">
            Action Layer
          </span>
        </div>
        <div className="flex-1" />
        <div className={`flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-widest ${connected ? 'text-emerald-300' : 'text-white/30'}`}>
          <Wifi size={11} />
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-white/30'}`} />
          {connected ? 'Bus connected' : 'Disconnected'}
        </div>
        <Link to="/chat" className="text-xs text-cyan-300 hover:text-cyan-200 font-mono uppercase tracking-widest transition-colors">
          Open Chat →
        </Link>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_340px] overflow-hidden">
        {/* World canvas */}
        <div className="relative flex items-center justify-center p-4 bg-gradient-to-br from-black via-[#0a0f14] to-black overflow-hidden">
          <div className="relative w-full max-w-5xl aspect-[10/7] rounded-xl border border-white/10 overflow-hidden shadow-[0_0_60px_rgba(0,200,255,0.05)]">
            <canvas
              ref={canvasRef}
              width={1400}
              height={980}
              className="w-full h-full block"
            />

            {/* Overlay HUD */}
            <div className="absolute top-3 left-3 flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest">
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-white/70">sim · gazebo-lite</span>
              </div>
              <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-white/50">
                {WORLD.w}m × {WORLD.h}m
              </div>
            </div>

            <div className="absolute bottom-3 left-3 text-[10px] font-mono text-white/40">
              <span>↑/↓ drive · ←/→ turn · space stop · R reset</span>
            </div>

            {/* Current action banner */}
            <AnimatePresence>
              {latest && stateRef.current.cmdUntil > performance.now() && (
                <motion.div
                  key={latest.ts}
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="absolute top-3 right-3 max-w-xs px-3 py-2 rounded-lg bg-rose-500/15 border border-rose-400/30 backdrop-blur"
                >
                  <div className="text-[9px] font-mono uppercase tracking-widest text-rose-300 mb-1">executing · {latest.source}</div>
                  <code className="text-xs font-mono text-white break-all">{latest.action}</code>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Right rail: telemetry + log + manual */}
        <aside className="border-l border-white/10 bg-black/60 flex flex-col overflow-hidden">
          {/* Telemetry */}
          <div className="px-4 py-4 border-b border-white/10">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-300">Telemetry</span>
              <button
                onClick={() => setRunning((r) => !r)}
                className="flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest px-2 py-1 rounded border border-white/10 hover:border-white/30 text-white/60 hover:text-white"
              >
                {running ? <><Pause size={10} /> pause</> : <><Play size={10} /> resume</>}
              </button>
            </div>
            <TelemetryRow label="x" value={telemetry.x.toFixed(2)} unit="m" />
            <TelemetryRow label="y" value={telemetry.y.toFixed(2)} unit="m" />
            <TelemetryRow label="heading" value={((telemetry.heading * 180 / Math.PI) % 360).toFixed(1)} unit="°" />
            <TelemetryRow label="linear_vel" value={telemetry.linVel.toFixed(3)} unit="m/s" highlight={Math.abs(telemetry.linVel) > 0.01} />
            <TelemetryRow label="angular_vel" value={telemetry.angVel.toFixed(3)} unit="rad/s" highlight={Math.abs(telemetry.angVel) > 0.01} />
            <TelemetryRow label="collisions" value={stateRef.current.collisions} unit="" />
          </div>

          {/* Action log */}
          <div className="flex-1 overflow-y-auto eden-chat-scroll px-4 py-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-300">Action bus</span>
              <span className="text-[10px] font-mono text-white/30">{log.length} msgs</span>
            </div>
            {log.length === 0 ? (
              <p className="text-xs text-white/30 font-mono">Awaiting actions from /chat or manual input…</p>
            ) : (
              <div className="space-y-2">
                {log.map((l, i) => (
                  <motion.div
                    key={l.ts}
                    initial={{ opacity: 0, x: 6 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="rounded-md border border-white/5 bg-white/[0.02] px-2.5 py-2"
                  >
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className={`text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${l.source === 'eden' ? 'bg-cyan-500/15 text-cyan-300' : 'bg-amber-500/15 text-amber-300'}`}>
                        {l.source}
                      </span>
                      <span className="text-[10px] text-white/30 font-mono">
                        {new Date(l.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                      </span>
                      {l.parsed && <span className="ml-auto text-[9px] font-mono text-emerald-300/70">parsed</span>}
                      {!l.parsed && <span className="ml-auto text-[9px] font-mono text-rose-300/70">unmapped</span>}
                    </div>
                    <code className="text-[11px] font-mono text-white/80 break-all block">{l.action}</code>
                    {l.parsed && (
                      <div className="mt-1 text-[10px] font-mono text-white/40 flex gap-3">
                        <span>lin={l.parsed.linear.toFixed(2)}</span>
                        <span>ang={l.parsed.angular.toFixed(2)}</span>
                        <span>dur={l.parsed.duration?.toFixed?.(1)}s</span>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {/* Manual */}
          <form onSubmit={handleManualSubmit} className="border-t border-white/10 p-3 flex gap-2">
            <input
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="/cmd_vel linear.x=0.3 · forward · turn left 45"
              className="flex-1 bg-white/5 border border-white/10 rounded-md px-3 py-2 text-xs font-mono text-white placeholder:text-white/25 focus:outline-none focus:border-cyan-400/30"
            />
            <button type="submit" className="px-3 py-2 rounded-md bg-white text-black text-xs font-semibold flex items-center gap-1 hover:bg-white/90">
              <Send size={12} /> Send
            </button>
          </form>
        </aside>
      </div>
    </div>
  )
}

function TelemetryRow({ label, value, unit, highlight }) {
  return (
    <div className={`flex items-center justify-between py-1 text-[11px] font-mono border-b border-white/5 last:border-b-0 ${highlight ? 'text-cyan-200' : 'text-white/60'}`}>
      <span className="uppercase tracking-widest text-white/40">{label}</span>
      <span>{value}<span className="text-white/30 ml-1">{unit}</span></span>
    </div>
  )
}

function circleIntersectsRect(cx, cy, r, rect) {
  const nx = clamp(cx, rect.x - rect.w / 2, rect.x + rect.w / 2)
  const ny = clamp(cy, rect.y - rect.h / 2, rect.y + rect.h / 2)
  const dx = cx - nx
  const dy = cy - ny
  return dx * dx + dy * dy < r * r
}

// ───── Renderer ─────
function render(canvas, s, now) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width
  const H = canvas.height
  const pm = ppm(W)

  // Clear with soft vignette
  ctx.fillStyle = '#030608'
  ctx.fillRect(0, 0, W, H)

  // Grid
  ctx.strokeStyle = 'rgba(120, 200, 240, 0.05)'
  ctx.lineWidth = 1
  for (let x = 0; x <= WORLD.w; x += 0.5) {
    ctx.beginPath()
    ctx.moveTo(x * pm, 0)
    ctx.lineTo(x * pm, H)
    ctx.stroke()
  }
  for (let y = 0; y <= WORLD.h; y += 0.5) {
    ctx.beginPath()
    ctx.moveTo(0, y * pm)
    ctx.lineTo(W, y * pm)
    ctx.stroke()
  }
  // Bolder 1m grid
  ctx.strokeStyle = 'rgba(120, 200, 240, 0.12)'
  for (let x = 0; x <= WORLD.w; x += 1) {
    ctx.beginPath()
    ctx.moveTo(x * pm, 0)
    ctx.lineTo(x * pm, H)
    ctx.stroke()
  }
  for (let y = 0; y <= WORLD.h; y += 1) {
    ctx.beginPath()
    ctx.moveTo(0, y * pm)
    ctx.lineTo(W, y * pm)
    ctx.stroke()
  }

  // Origin axes
  ctx.strokeStyle = 'rgba(255, 120, 120, 0.35)'
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0.4 * pm, 0); ctx.stroke()
  ctx.strokeStyle = 'rgba(120, 255, 180, 0.35)'
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, 0.4 * pm); ctx.stroke()

  // Obstacles
  for (const o of OBSTACLES) {
    ctx.fillStyle = 'rgba(120, 140, 180, 0.15)'
    ctx.strokeStyle = 'rgba(180, 210, 240, 0.45)'
    ctx.lineWidth = 1.5
    const x = (o.x - o.w / 2) * pm
    const y = (o.y - o.h / 2) * pm
    const w = o.w * pm
    const h = o.h * pm
    ctx.fillRect(x, y, w, h)
    ctx.strokeRect(x, y, w, h)
    ctx.fillStyle = 'rgba(180, 210, 240, 0.6)'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.fillText(o.label, x + 4, y + 14)
  }

  // Trail (fading)
  if (s.trail.length > 1) {
    ctx.lineWidth = 2
    for (let i = 1; i < s.trail.length; i++) {
      const age = (now - s.trail[i].ts) / 5000
      const alpha = Math.max(0, 0.9 - age)
      ctx.strokeStyle = `rgba(0, 220, 255, ${alpha * 0.6})`
      ctx.beginPath()
      ctx.moveTo(s.trail[i - 1].x * pm, s.trail[i - 1].y * pm)
      ctx.lineTo(s.trail[i].x * pm, s.trail[i].y * pm)
      ctx.stroke()
    }
  }

  // Robot
  const rx = s.x * pm
  const ry = s.y * pm
  const rr = ROBOT_RADIUS * pm

  // Glow
  const grad = ctx.createRadialGradient(rx, ry, rr * 0.6, rx, ry, rr * 3)
  grad.addColorStop(0, 'rgba(0, 220, 255, 0.25)')
  grad.addColorStop(1, 'rgba(0, 220, 255, 0)')
  ctx.fillStyle = grad
  ctx.beginPath(); ctx.arc(rx, ry, rr * 3, 0, Math.PI * 2); ctx.fill()

  // Body
  ctx.fillStyle = '#0a0f14'
  ctx.strokeStyle = '#22e0ff'
  ctx.lineWidth = 2
  ctx.beginPath(); ctx.arc(rx, ry, rr, 0, Math.PI * 2); ctx.fill(); ctx.stroke()

  // Direction arrow
  const hx = rx + Math.cos(s.heading) * rr * 1.4
  const hy = ry + Math.sin(s.heading) * rr * 1.4
  ctx.strokeStyle = '#22e0ff'
  ctx.lineWidth = 3
  ctx.lineCap = 'round'
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(hx, hy); ctx.stroke()

  // Wheels (stylistic)
  ctx.fillStyle = 'rgba(180, 220, 255, 0.4)'
  const perpX = -Math.sin(s.heading)
  const perpY = Math.cos(s.heading)
  for (const side of [-1, 1]) {
    const wx = rx + perpX * side * rr * 0.9
    const wy = ry + perpY * side * rr * 0.9
    ctx.beginPath(); ctx.arc(wx, wy, rr * 0.2, 0, Math.PI * 2); ctx.fill()
  }

  // Label
  ctx.fillStyle = 'rgba(255,255,255,0.7)'
  ctx.font = '10px JetBrains Mono, monospace'
  ctx.fillText('EDEN-01', rx - 22, ry - rr - 6)
}
