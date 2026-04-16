import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, ArrowLeft, Play, Pause, Wifi, Send, Brain, Shield, X, Check } from 'lucide-react'
import { openSimBusReceiver } from '../lib/simBridge'
import { classifyAction } from '../lib/cognitiveLayer'
import "@fontsource/jetbrains-mono"

// ───── World ─────
const WORLD = { w: 30, h: 20 } // meters
const VIEW = { w: 14, h: 10 }  // visible meters in the main canvas
const ROBOT_RADIUS = 0.28
const DETECTION_R = 4.0        // meters — how far EDEN "sees"

// Floor plan: a loose research-lab layout
const OBSTACLES = [
  // North wall section
  { x: 4,   y: 1.2, w: 6, h: 0.4, label: 'north wall' },
  { x: 14,  y: 1.2, w: 10, h: 0.4, label: 'north wall' },
  { x: 25,  y: 1.2, w: 5, h: 0.4, label: 'north wall' },
  // South wall section
  { x: 6,   y: 18.8, w: 8, h: 0.4, label: 'south wall' },
  { x: 20,  y: 18.8, w: 8, h: 0.4, label: 'south wall' },
  // Lab island
  { x: 6,   y: 6,   w: 2.4, h: 0.6, label: 'workbench A' },
  { x: 6,   y: 9,   w: 2.4, h: 0.6, label: 'workbench B' },
  { x: 9,   y: 7.5, w: 0.6, h: 3,   label: 'cable trunk' },
  // Docks
  { x: 3,   y: 14,  w: 0.8, h: 3,   label: 'charging dock' },
  { x: 27,  y: 4,   w: 1.2, h: 0.8, label: 'parts bin' },
  { x: 27,  y: 6,   w: 1.2, h: 0.8, label: 'parts bin' },
  // Mid room clutter
  { x: 14,  y: 8,   w: 1.0, h: 1.0, label: 'crate' },
  { x: 17,  y: 11,  w: 0.8, h: 0.8, label: 'box' },
  { x: 12,  y: 13,  w: 1.4, h: 0.4, label: 'bench' },
  { x: 22,  y: 10,  w: 0.6, h: 2.6, label: 'server rack' },
  // South lab
  { x: 18,  y: 16,  w: 2.0, h: 0.5, label: 'bench' },
  { x: 24,  y: 15,  w: 0.8, h: 0.8, label: 'marker' },
  { x: 10,  y: 16.5,w: 0.4, h: 0.4, label: 'cone' },
  { x: 13,  y: 16.5,w: 0.4, h: 0.4, label: 'cone' },
]

const NPCS = [
  { name: 'EDEN-02',       x: 7.5,  y: 14.5, color: '#7c3aed', role: 'sibling'  },
  { name: 'delivery-bot',  x: 26,   y: 12.0, color: '#f59e0b', role: 'logistics'},
  { name: 'scout-01',      x: 16,   y: 4.5,  color: '#10b981', role: 'scout'    },
  { name: 'inspector-A3',  x: 21,   y: 14.0, color: '#06b6d4', role: 'inspector'},
  { name: 'sweeper',       x: 11,   y: 10.5, color: '#ec4899', role: 'clean'    },
]

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)) }
function distance(a, b) { const dx = a.x - b.x; const dy = a.y - b.y; return Math.sqrt(dx*dx + dy*dy) }

function circleIntersectsRect(cx, cy, r, rect) {
  const nx = clamp(cx, rect.x - rect.w/2, rect.x + rect.w/2)
  const ny = clamp(cy, rect.y - rect.h/2, rect.y + rect.h/2)
  const dx = cx - nx; const dy = cy - ny
  return dx*dx + dy*dy < r*r
}

// ───── Main ─────
export default function Simulator() {
  const canvasRef = useRef(null)
  const stateRef = useRef({
    x: 2.5, y: 10, heading: 0,
    linVel: 0, angVel: 0,
    cmdUntil: 0,
    trail: [],
    lastTick: performance.now(),
    collisions: 0,
  })
  const [telemetry, setTelemetry] = useState({ x: 2.5, y: 10, heading: 0, linVel: 0, angVel: 0 })
  const [log, setLog] = useState([])
  const [running, setRunning] = useState(true)
  const [manualInput, setManualInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [thinking, setThinking] = useState(null) // { action, source }
  const [useLLM, setUseLLM] = useState(true)

  // Bus
  useEffect(() => {
    const close = openSimBusReceiver((payload) => {
      applyAction(payload.action, 'eden')
    })
    setConnected(true)
    return () => { close(); setConnected(false) }
  }, [])

  // Keyboard manual control — direct, bypasses LLM
  useEffect(() => {
    function onKey(e) {
      if (e.target && ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return
      const s = stateRef.current
      let cmd = null
      if (e.code === 'ArrowUp')    cmd = { lin: 0.45, ang: 0, dur: 0.4 }
      if (e.code === 'ArrowDown')  cmd = { lin: -0.35, ang: 0, dur: 0.4 }
      if (e.code === 'ArrowLeft')  cmd = { lin: 0, ang: 1.0, dur: 0.3 }
      if (e.code === 'ArrowRight') cmd = { lin: 0, ang: -1.0, dur: 0.3 }
      if (e.code === 'Space')      cmd = { lin: 0, ang: 0, dur: 0.1 }
      if (cmd) {
        e.preventDefault()
        s.linVel = cmd.lin; s.angVel = cmd.ang
        s.cmdUntil = performance.now() + cmd.dur * 1000
      }
      if (e.code === 'KeyR') {
        s.x = 2.5; s.y = 10; s.heading = 0; s.linVel = 0; s.angVel = 0
        s.cmdUntil = 0; s.trail = []; s.collisions = 0
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  async function applyAction(rawAction, source) {
    const ts = Date.now()
    const s = stateRef.current

    if (!useLLM || source === 'manual-direct') {
      // Direct path — skip LLM for fast iteration
      const { parseAction } = await import('../lib/simBridge')
      const parsed = parseAction(rawAction)
      const entry = {
        ts, source, action: rawAction,
        decision: parsed ? 'execute' : 'refuse',
        reason: parsed ? 'direct (no LLM)' : 'unparseable',
        linear: parsed?.linear, angular: parsed?.angular, duration: parsed?.duration,
        model: 'regex', ms: 0,
      }
      setLog((prev) => [entry, ...prev].slice(0, 50))
      if (!parsed) return
      commitMotion(parsed.linear, parsed.angular, parsed.duration, parsed.stop)
      return
    }

    setThinking({ action: rawAction, source })

    // Determine nearby obstacles + NPCs to give the LLM good context
    const obstacles = OBSTACLES.filter((o) => distance({ x: s.x, y: s.y }, o) < 8)
    const npcs = NPCS.filter((n) => distance({ x: s.x, y: s.y }, n) < DETECTION_R * 1.5)
    const history = log.slice(0, 4).map((l) => ({ source: l.source, action: l.action, decision: l.decision }))

    const res = await classifyAction({
      action: rawAction,
      robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
      obstacles,
      npcs,
      history,
    })

    setThinking(null)

    const entry = {
      ts, source, action: rawAction,
      decision: res.decision,
      reason: res.reason,
      linear: res.linear, angular: res.angular, duration: res.duration,
      model: res.model, ms: res.ms,
    }
    setLog((prev) => [entry, ...prev].slice(0, 50))

    if (res.decision === 'refuse' || res.linear === null) return
    commitMotion(res.linear, res.angular, res.duration, false)
  }

  function commitMotion(lin, ang, dur, stop) {
    const s = stateRef.current
    if (stop) { s.linVel = 0; s.angVel = 0; s.cmdUntil = performance.now() + 100; return }
    s.linVel = lin ?? 0
    s.angVel = ang ?? 0
    s.cmdUntil = performance.now() + (dur ?? 2) * 1000
  }

  // Physics + render loop
  useEffect(() => {
    let raf = 0
    function tick(now) {
      const s = stateRef.current
      const dt = Math.min(0.05, (now - s.lastTick) / 1000)
      s.lastTick = now

      if (running) {
        if (now > s.cmdUntil) { s.linVel = 0; s.angVel = 0 }
        const nextX = s.x + s.linVel * Math.cos(s.heading) * dt
        const nextY = s.y + s.linVel * Math.sin(s.heading) * dt
        const nextHeading = s.heading + s.angVel * dt

        const hitObstacle = OBSTACLES.some((o) => circleIntersectsRect(nextX, nextY, ROBOT_RADIUS, o))
        const hitWall = nextX < ROBOT_RADIUS || nextX > WORLD.w - ROBOT_RADIUS
                     || nextY < ROBOT_RADIUS || nextY > WORLD.h - ROBOT_RADIUS
        const hitNpc = NPCS.some((n) => distance({ x: nextX, y: nextY }, n) < ROBOT_RADIUS + 0.35)

        if (!hitObstacle && !hitWall && !hitNpc) {
          s.x = nextX; s.y = nextY
        } else {
          s.linVel = 0
          s.collisions += 1
        }
        s.heading = nextHeading

        if (Math.abs(s.linVel) > 0.02 || Math.abs(s.angVel) > 0.05) {
          s.trail.push({ x: s.x, y: s.y, ts: now })
          if (s.trail.length > 800) s.trail.shift()
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

  const nearbyNpcs = NPCS.filter((n) => distance({ x: telemetry.x, y: telemetry.y }, n) < DETECTION_R)

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
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
        <label className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest ml-4 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={useLLM}
            onChange={(e) => setUseLLM(e.target.checked)}
            className="accent-cyan-400"
          />
          <Brain size={11} className={useLLM ? 'text-cyan-400' : 'text-white/30'} />
          <span className={useLLM ? 'text-cyan-300' : 'text-white/30'}>cognitive gate</span>
        </label>
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

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_360px] overflow-hidden">
        {/* World canvas */}
        <div className="relative flex items-center justify-center p-4 bg-gradient-to-br from-black via-[#0a0f14] to-black overflow-hidden">
          <div className="relative w-full max-w-6xl aspect-[14/10] rounded-xl border border-white/10 overflow-hidden shadow-[0_0_60px_rgba(0,200,255,0.05)]">
            <canvas
              ref={canvasRef}
              width={1680}
              height={1200}
              className="w-full h-full block"
            />

            {/* HUD top-left */}
            <div className="absolute top-3 left-3 flex flex-col gap-2 text-[10px] font-mono uppercase tracking-widest pointer-events-none">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-white/70">sim · gazebo-lite</span>
                </div>
                <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-white/50">
                  {WORLD.w}m × {WORLD.h}m world · {VIEW.w}m × {VIEW.h}m view
                </div>
              </div>
              {nearbyNpcs.length > 0 && (
                <div className="px-2 py-1 rounded bg-cyan-500/10 border border-cyan-400/30 backdrop-blur text-cyan-200">
                  detected {nearbyNpcs.length} robot{nearbyNpcs.length === 1 ? '' : 's'}: {nearbyNpcs.map((n) => n.name).join(', ')}
                </div>
              )}
            </div>

            {/* HUD bottom-left */}
            <div className="absolute bottom-3 left-3 text-[10px] font-mono text-white/40 pointer-events-none">
              <span>↑/↓ drive · ←/→ turn · space stop · R reset</span>
            </div>

            {/* Thinking banner */}
            <AnimatePresence>
              {thinking && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="absolute top-3 right-3 max-w-sm px-3 py-2 rounded-lg bg-cyan-500/10 border border-cyan-400/30 backdrop-blur flex items-center gap-2"
                >
                  <Brain size={12} className="text-cyan-300 animate-pulse" />
                  <div>
                    <div className="text-[9px] font-mono uppercase tracking-widest text-cyan-300 mb-0.5">cognitive layer · deliberating</div>
                    <code className="text-xs font-mono text-white/80 break-all">{thinking.action}</code>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Last decision banner */}
            <AnimatePresence>
              {!thinking && log[0] && stateRef.current.cmdUntil > performance.now() && (
                <motion.div
                  key={log[0].ts}
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`absolute top-3 right-3 max-w-sm px-3 py-2 rounded-lg backdrop-blur border ${
                    log[0].decision === 'refuse' ? 'bg-rose-500/10 border-rose-400/30'
                    : log[0].decision === 'modify' ? 'bg-amber-500/10 border-amber-400/30'
                    : 'bg-emerald-500/10 border-emerald-400/30'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    {log[0].decision === 'refuse' ? <X size={11} className="text-rose-300" /> : <Check size={11} className="text-emerald-300" />}
                    <span className="text-[9px] font-mono uppercase tracking-widest text-white/70">
                      {log[0].decision} · {log[0].source} · {log[0].ms}ms
                    </span>
                  </div>
                  <code className="text-[11px] font-mono text-white block mb-1">{log[0].action}</code>
                  <p className="text-[10px] text-white/60 leading-relaxed">{log[0].reason}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Right rail */}
        <aside className="border-l border-white/10 bg-black/60 flex flex-col overflow-hidden">
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
            <Row label="x"           value={telemetry.x.toFixed(2)} unit="m" />
            <Row label="y"           value={telemetry.y.toFixed(2)} unit="m" />
            <Row label="heading"     value={((telemetry.heading * 180 / Math.PI) % 360).toFixed(1)} unit="°" />
            <Row label="linear_vel"  value={telemetry.linVel.toFixed(3)} unit="m/s" hi={Math.abs(telemetry.linVel) > 0.01} />
            <Row label="angular_vel" value={telemetry.angVel.toFixed(3)} unit="rad/s" hi={Math.abs(telemetry.angVel) > 0.01} />
            <Row label="collisions"  value={stateRef.current.collisions} unit="" />
            <Row label="nearby_npcs" value={nearbyNpcs.length} unit="" hi={nearbyNpcs.length > 0} />
          </div>

          <div className="flex-1 overflow-y-auto eden-chat-scroll px-4 py-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-300">Decision log</span>
              <span className="text-[10px] font-mono text-white/30">{log.length} msgs</span>
            </div>
            {log.length === 0 ? (
              <p className="text-xs text-white/30 font-mono">Awaiting actions from /chat or manual input…</p>
            ) : (
              <div className="space-y-2">
                {log.map((l) => (
                  <motion.div
                    key={l.ts}
                    initial={{ opacity: 0, x: 6 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`rounded-md border px-2.5 py-2 ${
                      l.decision === 'refuse' ? 'border-rose-400/20 bg-rose-500/[0.03]'
                      : l.decision === 'modify' ? 'border-amber-400/20 bg-amber-500/[0.03]'
                      : 'border-white/5 bg-white/[0.02]'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                      <span className={`text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${l.source === 'eden' ? 'bg-cyan-500/15 text-cyan-300' : 'bg-amber-500/15 text-amber-300'}`}>
                        {l.source}
                      </span>
                      <span className={`text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${
                        l.decision === 'refuse' ? 'bg-rose-500/20 text-rose-300'
                        : l.decision === 'modify' ? 'bg-amber-500/20 text-amber-300'
                        : 'bg-emerald-500/20 text-emerald-300'
                      }`}>
                        {l.decision}
                      </span>
                      <span className="text-[10px] text-white/30 font-mono">{new Date(l.ts).toLocaleTimeString([], { hour12: false })}</span>
                      <span className="ml-auto text-[9px] font-mono text-white/40">{l.model} · {l.ms}ms</span>
                    </div>
                    <code className="text-[11px] font-mono text-white/80 break-all block mb-1">{l.action}</code>
                    {l.reason && <p className="text-[10px] text-white/50 leading-relaxed">{l.reason}</p>}
                    {l.linear != null && (
                      <div className="mt-1 text-[10px] font-mono text-white/40 flex gap-3 flex-wrap">
                        <span>lin={Number(l.linear).toFixed(2)}</span>
                        <span>ang={Number(l.angular).toFixed(2)}</span>
                        <span>dur={Number(l.duration).toFixed(1)}s</span>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <form onSubmit={handleManualSubmit} className="border-t border-white/10 p-3 flex gap-2">
            <input
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="navigate to charging dock · /cmd_vel linear.x=0.3 · spin"
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

function Row({ label, value, unit, hi }) {
  return (
    <div className={`flex items-center justify-between py-1 text-[11px] font-mono border-b border-white/5 last:border-b-0 ${hi ? 'text-cyan-200' : 'text-white/60'}`}>
      <span className="uppercase tracking-widest text-white/40">{label}</span>
      <span>{value}<span className="text-white/30 ml-1">{unit}</span></span>
    </div>
  )
}

// ───── Renderer ─────
function render(canvas, s, now) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width
  const H = canvas.height

  // Camera — follow the robot; clamp so we never pan past world bounds
  const camCx = clamp(s.x, VIEW.w / 2, WORLD.w - VIEW.w / 2)
  const camCy = clamp(s.y, VIEW.h / 2, WORLD.h - VIEW.h / 2)
  const camX0 = camCx - VIEW.w / 2
  const camY0 = camCy - VIEW.h / 2
  const pmX = W / VIEW.w
  const pmY = H / VIEW.h
  const pm = Math.min(pmX, pmY)

  // Clear
  ctx.fillStyle = '#030608'
  ctx.fillRect(0, 0, W, H)

  // World-space draw helpers
  const tx = (wx) => (wx - camX0) * pm
  const ty = (wy) => (wy - camY0) * pm

  // Grid (major at 1m, minor at 0.5m) — only within visible viewport-ish
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(120,200,240,0.05)'
  const minX = Math.floor(camX0 * 2) / 2
  const maxX = minX + VIEW.w + 1
  const minY = Math.floor(camY0 * 2) / 2
  const maxY = minY + VIEW.h + 1
  for (let x = minX; x <= maxX; x += 0.5) {
    ctx.beginPath(); ctx.moveTo(tx(x), 0); ctx.lineTo(tx(x), H); ctx.stroke()
  }
  for (let y = minY; y <= maxY; y += 0.5) {
    ctx.beginPath(); ctx.moveTo(0, ty(y)); ctx.lineTo(W, ty(y)); ctx.stroke()
  }
  ctx.strokeStyle = 'rgba(120,200,240,0.12)'
  for (let x = Math.ceil(minX); x <= maxX; x += 1) {
    ctx.beginPath(); ctx.moveTo(tx(x), 0); ctx.lineTo(tx(x), H); ctx.stroke()
  }
  for (let y = Math.ceil(minY); y <= maxY; y += 1) {
    ctx.beginPath(); ctx.moveTo(0, ty(y)); ctx.lineTo(W, ty(y)); ctx.stroke()
  }

  // World bounds as faint rectangle
  ctx.strokeStyle = 'rgba(120,200,240,0.25)'
  ctx.lineWidth = 1.5
  ctx.strokeRect(tx(0), ty(0), WORLD.w * pm, WORLD.h * pm)

  // Obstacles
  for (const o of OBSTACLES) {
    const x = tx(o.x - o.w/2)
    const y = ty(o.y - o.h/2)
    const w = o.w * pm
    const h = o.h * pm
    if (x + w < 0 || x > W || y + h < 0 || y > H) continue
    ctx.fillStyle = 'rgba(120,140,180,0.18)'
    ctx.strokeStyle = 'rgba(180,210,240,0.45)'
    ctx.lineWidth = 1.5
    ctx.fillRect(x, y, w, h)
    ctx.strokeRect(x, y, w, h)
    if (w > 30 && h > 12) {
      ctx.fillStyle = 'rgba(180,210,240,0.55)'
      ctx.font = '10px JetBrains Mono, monospace'
      ctx.fillText(o.label, x + 4, y + 13)
    }
  }

  // NPC robots
  for (const n of NPCS) {
    const nx = tx(n.x); const ny = ty(n.y)
    if (nx < -20 || nx > W + 20 || ny < -20 || ny > H + 20) continue
    const r = ROBOT_RADIUS * pm * 0.85
    ctx.fillStyle = '#0a0f14'
    ctx.strokeStyle = n.color
    ctx.lineWidth = 2
    ctx.beginPath(); ctx.arc(nx, ny, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
    // Subtle pulse
    ctx.strokeStyle = `${n.color}66`
    ctx.lineWidth = 1
    const pulse = 0.5 + Math.sin(now / 700 + n.x) * 0.2
    ctx.beginPath(); ctx.arc(nx, ny, r * (1.8 + pulse * 0.4), 0, Math.PI * 2); ctx.stroke()
    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.75)'
    ctx.font = '10px JetBrains Mono, monospace'
    const label = `${n.name} · ${n.role}`
    const tw = ctx.measureText(label).width
    ctx.fillStyle = 'rgba(0,0,0,0.5)'
    ctx.fillRect(nx - tw/2 - 4, ny - r - 18, tw + 8, 14)
    ctx.fillStyle = n.color
    ctx.fillText(label, nx - tw/2, ny - r - 7)
  }

  // Detection ring around robot
  const rx = tx(s.x); const ry = ty(s.y)
  const dRad = DETECTION_R * pm
  const detGrad = ctx.createRadialGradient(rx, ry, dRad * 0.7, rx, ry, dRad)
  detGrad.addColorStop(0, 'rgba(34,224,255,0)')
  detGrad.addColorStop(1, 'rgba(34,224,255,0.09)')
  ctx.fillStyle = detGrad
  ctx.beginPath(); ctx.arc(rx, ry, dRad, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = 'rgba(34,224,255,0.25)'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 6])
  ctx.beginPath(); ctx.arc(rx, ry, dRad, 0, Math.PI * 2); ctx.stroke()
  ctx.setLineDash([])

  // Trail
  if (s.trail.length > 1) {
    ctx.lineWidth = 2
    for (let i = 1; i < s.trail.length; i++) {
      const age = (now - s.trail[i].ts) / 6000
      const alpha = Math.max(0, 0.9 - age)
      if (alpha <= 0) continue
      ctx.strokeStyle = `rgba(0,220,255,${alpha * 0.6})`
      ctx.beginPath()
      ctx.moveTo(tx(s.trail[i-1].x), ty(s.trail[i-1].y))
      ctx.lineTo(tx(s.trail[i].x), ty(s.trail[i].y))
      ctx.stroke()
    }
  }

  // Robot
  const rr = ROBOT_RADIUS * pm
  const grad = ctx.createRadialGradient(rx, ry, rr * 0.6, rx, ry, rr * 3)
  grad.addColorStop(0, 'rgba(0,220,255,0.25)')
  grad.addColorStop(1, 'rgba(0,220,255,0)')
  ctx.fillStyle = grad
  ctx.beginPath(); ctx.arc(rx, ry, rr * 3, 0, Math.PI * 2); ctx.fill()

  ctx.fillStyle = '#0a0f14'
  ctx.strokeStyle = '#22e0ff'
  ctx.lineWidth = 2
  ctx.beginPath(); ctx.arc(rx, ry, rr, 0, Math.PI * 2); ctx.fill(); ctx.stroke()

  const hx = rx + Math.cos(s.heading) * rr * 1.5
  const hy = ry + Math.sin(s.heading) * rr * 1.5
  ctx.strokeStyle = '#22e0ff'
  ctx.lineWidth = 3
  ctx.lineCap = 'round'
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(hx, hy); ctx.stroke()

  ctx.fillStyle = 'rgba(180,220,255,0.4)'
  const perpX = -Math.sin(s.heading)
  const perpY = Math.cos(s.heading)
  for (const side of [-1, 1]) {
    const wx = rx + perpX * side * rr * 0.9
    const wy = ry + perpY * side * rr * 0.9
    ctx.beginPath(); ctx.arc(wx, wy, rr * 0.2, 0, Math.PI * 2); ctx.fill()
  }

  ctx.fillStyle = 'rgba(34,224,255,0.9)'
  ctx.font = 'bold 11px JetBrains Mono, monospace'
  ctx.fillText('EDEN-01', rx - 26, ry - rr - 9)

  // ───── Minimap (top-right) ─────
  const mmW = Math.min(280, W * 0.22)
  const mmH = mmW * (WORLD.h / WORLD.w)
  const mmX = W - mmW - 12
  const mmY = 12
  const mmPm = mmW / WORLD.w

  // Card
  ctx.fillStyle = 'rgba(0,0,0,0.75)'
  ctx.fillRect(mmX - 6, mmY - 6, mmW + 12, mmH + 30)
  ctx.strokeStyle = 'rgba(120,200,240,0.3)'
  ctx.strokeRect(mmX - 6, mmY - 6, mmW + 12, mmH + 30)
  ctx.fillStyle = 'rgba(120,200,240,0.6)'
  ctx.font = '9px JetBrains Mono, monospace'
  ctx.fillText('MINIMAP · full world', mmX, mmY - 10)

  // World background
  ctx.fillStyle = 'rgba(10,20,30,0.95)'
  ctx.fillRect(mmX, mmY, mmW, mmH)

  // Obstacles
  for (const o of OBSTACLES) {
    ctx.fillStyle = 'rgba(180,210,240,0.55)'
    ctx.fillRect(mmX + (o.x - o.w/2) * mmPm, mmY + (o.y - o.h/2) * mmPm, o.w * mmPm, o.h * mmPm)
  }

  // NPCs
  for (const n of NPCS) {
    ctx.fillStyle = n.color
    ctx.beginPath(); ctx.arc(mmX + n.x * mmPm, mmY + n.y * mmPm, 3, 0, Math.PI * 2); ctx.fill()
  }

  // Robot
  ctx.fillStyle = '#22e0ff'
  ctx.beginPath(); ctx.arc(mmX + s.x * mmPm, mmY + s.y * mmPm, 3.5, 0, Math.PI * 2); ctx.fill()
  // Heading tick
  ctx.strokeStyle = '#22e0ff'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(mmX + s.x * mmPm, mmY + s.y * mmPm)
  ctx.lineTo(mmX + (s.x + Math.cos(s.heading) * 0.7) * mmPm, mmY + (s.y + Math.sin(s.heading) * 0.7) * mmPm)
  ctx.stroke()

  // Viewport rect on minimap
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth = 1
  ctx.setLineDash([3, 3])
  ctx.strokeRect(mmX + camX0 * mmPm, mmY + camY0 * mmPm, VIEW.w * mmPm, VIEW.h * mmPm)
  ctx.setLineDash([])
}
