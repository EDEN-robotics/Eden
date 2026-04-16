import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, ArrowLeft, Play, Pause, Wifi, Send, Brain, Shield, X, Check, Radar, Activity, Radio } from 'lucide-react'
import { openSimBusReceiver } from '../lib/simBridge'
import { classifyAction } from '../lib/cognitiveLayer'
import { publish, listTopics, Twist, Odometry, LaserScan, TFMessage } from '../lib/rosTopics'
import "@fontsource/jetbrains-mono"

// ───── World ─────
const WORLD = { w: 30, h: 20 }
const VIEW  = { w: 14, h: 10 }

// Robot parameters — realistic diff-drive (roughly TurtleBot3 Waffle scale)
const R = {
  radius: 0.28,         // body radius (m)
  wheel_r: 0.08,        // wheel radius (m)
  wheel_base: 0.35,     // track width (m)
  mass: 12.0,           // kg
  max_lin: 0.6,         // m/s
  max_ang: 2.0,         // rad/s
  max_lin_accel: 1.2,   // m/s²
  max_ang_accel: 4.0,   // rad/s²
  motor_tau: 0.14,      // first-order motor response time constant (s)
  lidar_range: 6.0,     // m
  lidar_fov: Math.PI * 1.5, // 270°
  lidar_rays: 180,
  detection_r: 4.0,
  odom_noise: 0.025,    // fraction of step — drift source
}

const OBSTACLES = [
  { x: 4,  y: 1.2, w: 6, h: 0.4, label: 'north wall' },
  { x: 14, y: 1.2, w: 10, h: 0.4, label: 'north wall' },
  { x: 25, y: 1.2, w: 5, h: 0.4, label: 'north wall' },
  { x: 6,  y: 18.8, w: 8, h: 0.4, label: 'south wall' },
  { x: 20, y: 18.8, w: 8, h: 0.4, label: 'south wall' },
  { x: 6,  y: 6,   w: 2.4, h: 0.6, label: 'workbench A' },
  { x: 6,  y: 9,   w: 2.4, h: 0.6, label: 'workbench B' },
  { x: 9,  y: 7.5, w: 0.6, h: 3,   label: 'cable trunk' },
  { x: 3,  y: 14,  w: 0.8, h: 3,   label: 'charging dock' },
  { x: 27, y: 4,   w: 1.2, h: 0.8, label: 'parts bin' },
  { x: 27, y: 6,   w: 1.2, h: 0.8, label: 'parts bin' },
  { x: 14, y: 8,   w: 1.0, h: 1.0, label: 'crate' },
  { x: 17, y: 11,  w: 0.8, h: 0.8, label: 'box' },
  { x: 12, y: 13,  w: 1.4, h: 0.4, label: 'bench' },
  { x: 22, y: 10,  w: 0.6, h: 2.6, label: 'server rack' },
  { x: 18, y: 16,  w: 2.0, h: 0.5, label: 'bench' },
  { x: 24, y: 15,  w: 0.8, h: 0.8, label: 'marker' },
  { x: 10, y: 16.5,w: 0.4, h: 0.4, label: 'cone' },
  { x: 13, y: 16.5,w: 0.4, h: 0.4, label: 'cone' },
]

const NPCS = [
  { name: 'EDEN-02',      x: 7.5,  y: 14.5, color: '#7c3aed', role: 'sibling'   },
  { name: 'delivery-bot', x: 26,   y: 12.0, color: '#f59e0b', role: 'logistics' },
  { name: 'scout-01',     x: 16,   y: 4.5,  color: '#10b981', role: 'scout'     },
  { name: 'inspector-A3', x: 21,   y: 14.0, color: '#06b6d4', role: 'inspector' },
  { name: 'sweeper',      x: 11,   y: 10.5, color: '#ec4899', role: 'clean'     },
]

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
function dist(a, b) { const dx=a.x-b.x, dy=a.y-b.y; return Math.sqrt(dx*dx+dy*dy) }

function circleIntersectsRect(cx, cy, r, rect) {
  const nx = clamp(cx, rect.x - rect.w/2, rect.x + rect.w/2)
  const ny = clamp(cy, rect.y - rect.h/2, rect.y + rect.h/2)
  const dx = cx-nx, dy = cy-ny
  return dx*dx+dy*dy < r*r
}

// Ray vs axis-aligned rectangle (slab method). Returns t (meters) or null.
function raycastRect(ox, oy, dx, dy, rect) {
  const minX = rect.x - rect.w/2, maxX = rect.x + rect.w/2
  const minY = rect.y - rect.h/2, maxY = rect.y + rect.h/2
  const inv_dx = dx !== 0 ? 1/dx : 1e9
  const inv_dy = dy !== 0 ? 1/dy : 1e9
  const tx1 = (minX-ox) * inv_dx, tx2 = (maxX-ox) * inv_dx
  const tmin = Math.min(tx1, tx2), tmax = Math.max(tx1, tx2)
  const ty1 = (minY-oy) * inv_dy, ty2 = (maxY-oy) * inv_dy
  const tymin = Math.min(ty1, ty2), tymax = Math.max(ty1, ty2)
  const entry = Math.max(tmin, tymin)
  const exit  = Math.min(tmax, tymax)
  if (entry > exit || exit < 0) return null
  return entry < 0 ? null : entry
}

function raycastWorldBounds(ox, oy, dx, dy) {
  let best = Infinity
  if (dx > 0) { const t = (WORLD.w - ox) / dx; if (t > 0 && t < best) best = t }
  if (dx < 0) { const t = (0 - ox) / dx; if (t > 0 && t < best) best = t }
  if (dy > 0) { const t = (WORLD.h - oy) / dy; if (t > 0 && t < best) best = t }
  if (dy < 0) { const t = (0 - oy) / dy; if (t > 0 && t < best) best = t }
  return best === Infinity ? null : best
}

function simulateLidar(s) {
  const N = R.lidar_rays
  const fov = R.lidar_fov
  const ranges = new Array(N)
  for (let i = 0; i < N; i++) {
    const a = s.heading - fov/2 + (i / (N - 1)) * fov
    const dx = Math.cos(a), dy = Math.sin(a)
    let best = R.lidar_range
    for (const o of OBSTACLES) {
      const t = raycastRect(s.x, s.y, dx, dy, o)
      if (t != null && t < best) best = t
    }
    const tw = raycastWorldBounds(s.x, s.y, dx, dy)
    if (tw != null && tw < best) best = tw
    // Add a bit of sensor noise
    const noisy = best + (Math.random() - 0.5) * 0.03
    ranges[i] = clamp(noisy, 0.05, R.lidar_range)
  }
  return ranges
}

// ───── Main component ─────
export default function Simulator() {
  const canvasRef = useRef(null)
  const stateRef = useRef({
    // ground truth pose
    x: 2.5, y: 10, heading: 0,
    // actual velocities (after motor dynamics)
    linVel: 0, angVel: 0,
    // target velocities from current cmd
    cmdLin: 0, cmdAng: 0, cmdUntil: 0,
    // wheel angular velocities (rad/s)
    wheelL: 0, wheelR: 0,
    // odometry (noisy estimate of pose)
    odomX: 2.5, odomY: 10, odomHeading: 0,
    // lidar scan
    lidar: new Array(R.lidar_rays).fill(R.lidar_range),
    // housekeeping
    trail: [],
    lastTick: performance.now(),
    lastOdomPub: 0,
    lastScanPub: 0,
    lastTFPub: 0,
    collisions: 0,
  })
  const [telemetry, setTelemetry] = useState({
    x: 2.5, y: 10, heading: 0, linVel: 0, angVel: 0,
    wheelL: 0, wheelR: 0, odomDrift: 0,
  })
  const [topicsSnap, setTopicsSnap] = useState([])
  const [log, setLog] = useState([])
  const [running, setRunning] = useState(true)
  const [manualInput, setManualInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [thinking, setThinking] = useState(null)
  const [useLLM, setUseLLM] = useState(true)
  const [showLidar, setShowLidar] = useState(true)
  const [showOdom, setShowOdom] = useState(true)

  // Action bus
  useEffect(() => {
    const close = openSimBusReceiver((payload) => { applyAction(payload.action, 'eden') })
    setConnected(true)
    return () => { close(); setConnected(false) }
  }, [])

  // Keyboard manual
  useEffect(() => {
    function onKey(e) {
      if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return
      const s = stateRef.current
      if (e.code === 'ArrowUp')    { s.cmdLin =  0.45; s.cmdAng = 0;   s.cmdUntil = performance.now()+400; e.preventDefault() }
      if (e.code === 'ArrowDown')  { s.cmdLin = -0.35; s.cmdAng = 0;   s.cmdUntil = performance.now()+400; e.preventDefault() }
      if (e.code === 'ArrowLeft')  { s.cmdLin =  0;    s.cmdAng = 1.0; s.cmdUntil = performance.now()+300; e.preventDefault() }
      if (e.code === 'ArrowRight') { s.cmdLin =  0;    s.cmdAng =-1.0; s.cmdUntil = performance.now()+300; e.preventDefault() }
      if (e.code === 'Space')      { s.cmdLin = 0;     s.cmdAng = 0;   s.cmdUntil = performance.now()+100; e.preventDefault() }
      if (e.code === 'KeyR') {
        s.x = 2.5; s.y = 10; s.heading = 0
        s.linVel = 0; s.angVel = 0; s.cmdLin = 0; s.cmdAng = 0; s.cmdUntil = 0
        s.odomX = 2.5; s.odomY = 10; s.odomHeading = 0
        s.trail = []; s.collisions = 0
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  async function applyAction(rawAction, source) {
    const ts = Date.now()
    const s = stateRef.current

    if (!useLLM || source === 'manual-direct') {
      const { parseAction } = await import('../lib/simBridge')
      const parsed = parseAction(rawAction)
      const entry = {
        ts, source, action: rawAction,
        decision: parsed ? 'execute' : 'refuse',
        reason: parsed ? 'direct (no LLM)' : 'unparseable',
        linear: parsed?.linear, angular: parsed?.angular, duration: parsed?.duration,
        model: 'regex', ms: 0,
      }
      setLog((p) => [entry, ...p].slice(0, 50))
      if (!parsed) return
      commitCmd(parsed.linear, parsed.angular, parsed.duration, parsed.stop)
      return
    }

    setThinking({ action: rawAction, source })
    const obstacles = OBSTACLES.filter((o) => dist({x:s.x,y:s.y}, o) < 8)
    const npcs = NPCS.filter((n) => dist({x:s.x,y:s.y}, n) < R.detection_r * 1.5)
    const history = log.slice(0, 4).map((l) => ({ source: l.source, action: l.action, decision: l.decision }))
    const res = await classifyAction({
      action: rawAction,
      robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
      obstacles, npcs, history,
    })
    setThinking(null)
    const entry = { ts, source, action: rawAction, decision: res.decision, reason: res.reason,
      linear: res.linear, angular: res.angular, duration: res.duration, model: res.model, ms: res.ms }
    setLog((p) => [entry, ...p].slice(0, 50))
    if (res.decision === 'refuse' || res.linear === null) return
    commitCmd(res.linear, res.angular, res.duration, false)
  }

  function commitCmd(lin, ang, dur, stop) {
    const s = stateRef.current
    const now = performance.now()
    if (stop) { s.cmdLin = 0; s.cmdAng = 0; s.cmdUntil = now + 100 }
    else {
      s.cmdLin = lin ?? 0
      s.cmdAng = ang ?? 0
      s.cmdUntil = now + (dur ?? 2) * 1000
    }
    // Publish the Twist on /cmd_vel
    publish('/cmd_vel', Twist(s.cmdLin, s.cmdAng))
  }

  // Physics + rendering + topic publish loop
  useEffect(() => {
    let raf = 0
    function tick(now) {
      const s = stateRef.current
      const dt = Math.min(0.05, (now - s.lastTick) / 1000)
      s.lastTick = now

      if (running) {
        // Target velocities — zero out if cmd expired
        const targetLin = now > s.cmdUntil ? 0 : s.cmdLin
        const targetAng = now > s.cmdUntil ? 0 : s.cmdAng

        // First-order motor dynamics (low-pass filter)
        const alpha = 1 - Math.exp(-dt / R.motor_tau)
        let newLin = s.linVel + (targetLin - s.linVel) * alpha
        let newAng = s.angVel + (targetAng - s.angVel) * alpha

        // Enforce acceleration limits (jerk-limited)
        const maxDlin = R.max_lin_accel * dt
        const maxDang = R.max_ang_accel * dt
        newLin = clamp(newLin, s.linVel - maxDlin, s.linVel + maxDlin)
        newAng = clamp(newAng, s.angVel - maxDang, s.angVel + maxDang)

        // Clamp to absolute maxima
        newLin = clamp(newLin, -R.max_lin, R.max_lin)
        newAng = clamp(newAng, -R.max_ang, R.max_ang)

        s.linVel = newLin
        s.angVel = newAng

        // Diff-drive inverse kinematics for wheel speeds
        s.wheelL = (s.linVel - R.wheel_base/2 * s.angVel) / R.wheel_r
        s.wheelR = (s.linVel + R.wheel_base/2 * s.angVel) / R.wheel_r

        // Integrate ground-truth pose (semi-implicit: heading first)
        const nextHeading = s.heading + s.angVel * dt
        const nextX = s.x + s.linVel * Math.cos(nextHeading) * dt
        const nextY = s.y + s.linVel * Math.sin(nextHeading) * dt

        const hit = OBSTACLES.some((o) => circleIntersectsRect(nextX, nextY, R.radius, o))
                 || nextX < R.radius || nextX > WORLD.w - R.radius
                 || nextY < R.radius || nextY > WORLD.h - R.radius
                 || NPCS.some((n) => dist({x:nextX,y:nextY}, n) < R.radius + 0.35)
        if (!hit) { s.x = nextX; s.y = nextY }
        else { s.linVel = 0; s.collisions += 1 }
        s.heading = nextHeading

        // Odometry — noisy integration (accumulates drift)
        const n = R.odom_noise
        const odomLin = s.linVel * (1 + (Math.random()-0.5)*n)
        const odomAng = s.angVel * (1 + (Math.random()-0.5)*n*2)
        s.odomHeading += odomAng * dt
        s.odomX += odomLin * Math.cos(s.odomHeading) * dt
        s.odomY += odomLin * Math.sin(s.odomHeading) * dt

        // Trail
        if (Math.abs(s.linVel) > 0.02 || Math.abs(s.angVel) > 0.05) {
          s.trail.push({ x: s.x, y: s.y, ts: now })
          if (s.trail.length > 800) s.trail.shift()
        }

        // LIDAR — cast rays every frame (cheap, 180 rays)
        s.lidar = simulateLidar(s)

        // Publish topics at realistic rates
        if (now - s.lastOdomPub > 20) { // ~50 Hz
          publish('/odom', Odometry({ x: s.odomX, y: s.odomY, heading: s.odomHeading, linVel: s.linVel, angVel: s.angVel, ts: now }))
          s.lastOdomPub = now
        }
        if (now - s.lastTFPub > 20) {
          publish('/tf', TFMessage({ x: s.x, y: s.y, heading: s.heading, ts: now }))
          s.lastTFPub = now
        }
        if (now - s.lastScanPub > 100) { // 10 Hz
          publish('/scan', LaserScan({
            angle_min: -R.lidar_fov/2, angle_max: R.lidar_fov/2,
            ranges: s.lidar, range_max: R.lidar_range, ts: now,
          }))
          s.lastScanPub = now
        }
      }

      render(canvasRef.current, s, now, { showLidar, showOdom })
      setTelemetry({
        x: s.x, y: s.y, heading: s.heading,
        linVel: s.linVel, angVel: s.angVel,
        wheelL: s.wheelL, wheelR: s.wheelR,
        odomDrift: Math.hypot(s.odomX - s.x, s.odomY - s.y),
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running, showLidar, showOdom])

  // Refresh topic table 5 Hz
  useEffect(() => {
    const h = setInterval(() => setTopicsSnap(listTopics()), 200)
    return () => clearInterval(h)
  }, [])

  function handleManualSubmit(e) {
    e?.preventDefault()
    if (!manualInput.trim()) return
    applyAction(manualInput.trim(), 'manual')
    setManualInput('')
  }

  const nearbyNpcs = NPCS.filter((n) => dist({x:telemetry.x,y:telemetry.y}, n) < R.detection_r)
  const lidarMin = stateRef.current.lidar ? Math.min(...stateRef.current.lidar) : R.lidar_range

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="flex items-center gap-4 px-6 py-4 border-b border-white/10 bg-black/80 backdrop-blur">
        <Link to="/" className="flex items-center gap-2 text-xs text-white/50 hover:text-white"><ArrowLeft size={12}/> Home</Link>
        <div className="h-4 w-px bg-white/10" />
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-rose-400" />
          <span className="text-sm font-semibold">EDEN Simulator</span>
          <span className="text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-400/30">
            dynamics · lidar · odom · ROS-bus
          </span>
        </div>
        <label className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest ml-4 cursor-pointer select-none">
          <input type="checkbox" checked={useLLM} onChange={(e) => setUseLLM(e.target.checked)} className="accent-cyan-400"/>
          <Brain size={11} className={useLLM ? 'text-cyan-400' : 'text-white/30'} />
          <span className={useLLM ? 'text-cyan-300' : 'text-white/30'}>cognitive gate</span>
        </label>
        <label className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest ml-2 cursor-pointer select-none">
          <input type="checkbox" checked={showLidar} onChange={(e) => setShowLidar(e.target.checked)} className="accent-rose-400"/>
          <Radar size={11} className={showLidar ? 'text-rose-400' : 'text-white/30'}/>
          <span className={showLidar ? 'text-rose-300' : 'text-white/30'}>lidar</span>
        </label>
        <label className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest ml-2 cursor-pointer select-none">
          <input type="checkbox" checked={showOdom} onChange={(e) => setShowOdom(e.target.checked)} className="accent-amber-400"/>
          <Activity size={11} className={showOdom ? 'text-amber-400' : 'text-white/30'}/>
          <span className={showOdom ? 'text-amber-300' : 'text-white/30'}>odom</span>
        </label>
        <div className="flex-1" />
        <div className={`flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-widest ${connected ? 'text-emerald-300' : 'text-white/30'}`}>
          <Wifi size={11}/>
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-white/30'}`}/>
          {connected ? 'action bus' : 'disconnected'}
        </div>
        <Link to="/chat" className="text-xs text-cyan-300 hover:text-cyan-200 font-mono uppercase tracking-widest">Open Chat →</Link>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_380px] overflow-hidden">
        {/* World */}
        <div className="relative flex items-center justify-center p-4 bg-gradient-to-br from-black via-[#0a0f14] to-black overflow-hidden">
          <div className="relative w-full max-w-6xl aspect-[14/10] rounded-xl border border-white/10 overflow-hidden shadow-[0_0_60px_rgba(0,200,255,0.05)]">
            <canvas ref={canvasRef} width={1680} height={1200} className="w-full h-full block"/>

            <div className="absolute top-3 left-3 flex flex-col gap-2 text-[10px] font-mono uppercase tracking-widest pointer-events-none">
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/>
                  <span className="text-white/70">sim · diff-drive · TurtleBot3-scale</span>
                </div>
                <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-white/50">
                  wheel_base {R.wheel_base}m · wheel_r {R.wheel_r}m · motor_tau {R.motor_tau}s
                </div>
                <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-rose-300">
                  LIDAR 270° · {R.lidar_rays} rays · {R.lidar_range}m · min {lidarMin.toFixed(2)}m
                </div>
              </div>
              {nearbyNpcs.length > 0 && (
                <div className="px-2 py-1 rounded bg-cyan-500/10 border border-cyan-400/30 backdrop-blur text-cyan-200">
                  detected {nearbyNpcs.length}: {nearbyNpcs.map((n) => n.name).join(', ')}
                </div>
              )}
            </div>

            <div className="absolute bottom-3 left-3 text-[10px] font-mono text-white/40 pointer-events-none">
              ↑↓ drive · ←→ turn · space stop · R reset
            </div>

            <AnimatePresence>
              {thinking && (
                <motion.div key="t" initial={{opacity:0,y:-6}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                  className="absolute top-3 right-3 max-w-sm px-3 py-2 rounded-lg bg-cyan-500/10 border border-cyan-400/30 backdrop-blur flex items-center gap-2">
                  <Brain size={12} className="text-cyan-300 animate-pulse"/>
                  <div>
                    <div className="text-[9px] font-mono uppercase tracking-widest text-cyan-300 mb-0.5">cognitive layer · deliberating</div>
                    <code className="text-xs font-mono text-white/80 break-all">{thinking.action}</code>
                  </div>
                </motion.div>
              )}
              {!thinking && log[0] && stateRef.current.cmdUntil > performance.now() && (
                <motion.div key={log[0].ts} initial={{opacity:0,y:-6}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                  className={`absolute top-3 right-3 max-w-sm px-3 py-2 rounded-lg backdrop-blur border ${
                    log[0].decision === 'refuse' ? 'bg-rose-500/10 border-rose-400/30'
                    : log[0].decision === 'modify' ? 'bg-amber-500/10 border-amber-400/30'
                    : 'bg-emerald-500/10 border-emerald-400/30'}`}>
                  <div className="flex items-center gap-1.5 mb-1">
                    {log[0].decision === 'refuse' ? <X size={11} className="text-rose-300"/> : <Check size={11} className="text-emerald-300"/>}
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
          {/* Telemetry */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-300">Telemetry</span>
              <button onClick={() => setRunning((r) => !r)}
                className="flex items-center gap-1 text-[10px] font-mono uppercase tracking-widest px-2 py-1 rounded border border-white/10 hover:border-white/30 text-white/60 hover:text-white">
                {running ? <><Pause size={10}/> pause</> : <><Play size={10}/> resume</>}
              </button>
            </div>
            <Row l="x"           v={telemetry.x.toFixed(2)}          u="m"/>
            <Row l="y"           v={telemetry.y.toFixed(2)}          u="m"/>
            <Row l="heading"     v={((telemetry.heading * 180/Math.PI) % 360).toFixed(1)} u="°"/>
            <Row l="linear_vel"  v={telemetry.linVel.toFixed(3)}     u="m/s"  hi={Math.abs(telemetry.linVel)>0.01}/>
            <Row l="angular_vel" v={telemetry.angVel.toFixed(3)}     u="rad/s" hi={Math.abs(telemetry.angVel)>0.01}/>
            <Row l="wheel_L"     v={telemetry.wheelL.toFixed(2)}     u="rad/s"/>
            <Row l="wheel_R"     v={telemetry.wheelR.toFixed(2)}     u="rad/s"/>
            <Row l="odom_drift"  v={telemetry.odomDrift.toFixed(3)}  u="m"     hi={telemetry.odomDrift>0.1}/>
            <Row l="lidar_min"   v={lidarMin.toFixed(2)}             u="m"     hi={lidarMin<0.8}/>
            <Row l="collisions"  v={stateRef.current.collisions}     u=""      hi={stateRef.current.collisions>0}/>
          </div>

          {/* Topic bus */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center gap-1.5 mb-2">
              <Radio size={11} className="text-emerald-300"/>
              <span className="text-[11px] font-mono uppercase tracking-widest text-emerald-300">ROS-style topic bus</span>
            </div>
            {topicsSnap.length === 0 ? (
              <p className="text-[10px] text-white/30 font-mono">no topics yet…</p>
            ) : (
              <table className="w-full text-[10px] font-mono">
                <thead>
                  <tr className="text-white/30 uppercase tracking-widest">
                    <th className="text-left">topic</th>
                    <th className="text-right">Hz</th>
                    <th className="text-right">msgs</th>
                  </tr>
                </thead>
                <tbody>
                  {topicsSnap.map((t) => (
                    <tr key={t.name} className="border-t border-white/5">
                      <td className="text-cyan-300 py-1">{t.name}</td>
                      <td className="text-right text-white/70">{t.rate.toFixed(1)}</td>
                      <td className="text-right text-white/40">{t.msgCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Decisions */}
          <div className="flex-1 overflow-y-auto eden-chat-scroll px-4 py-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-300">Decision log</span>
              <span className="text-[10px] font-mono text-white/30">{log.length}</span>
            </div>
            {log.length === 0 ? (
              <p className="text-xs text-white/30 font-mono">awaiting actions…</p>
            ) : (
              <div className="space-y-2">
                {log.map((l) => (
                  <div key={l.ts}
                    className={`rounded-md border px-2.5 py-2 ${
                      l.decision === 'refuse' ? 'border-rose-400/20 bg-rose-500/[0.03]'
                      : l.decision === 'modify' ? 'border-amber-400/20 bg-amber-500/[0.03]'
                      : 'border-white/5 bg-white/[0.02]'}`}>
                    <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                      <span className={`text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${l.source === 'eden' ? 'bg-cyan-500/15 text-cyan-300' : 'bg-amber-500/15 text-amber-300'}`}>{l.source}</span>
                      <span className={`text-[9px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${
                        l.decision === 'refuse' ? 'bg-rose-500/20 text-rose-300'
                        : l.decision === 'modify' ? 'bg-amber-500/20 text-amber-300'
                        : 'bg-emerald-500/20 text-emerald-300'}`}>{l.decision}</span>
                      <span className="ml-auto text-[9px] font-mono text-white/40">{l.model} · {l.ms}ms</span>
                    </div>
                    <code className="text-[11px] font-mono text-white/80 break-all block mb-1">{l.action}</code>
                    {l.reason && <p className="text-[10px] text-white/50 leading-relaxed">{l.reason}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>

          <form onSubmit={handleManualSubmit} className="border-t border-white/10 p-3 flex gap-2">
            <input value={manualInput} onChange={(e) => setManualInput(e.target.value)}
              placeholder="navigate to dock · /cmd_vel linear.x=0.3 · spin"
              className="flex-1 bg-white/5 border border-white/10 rounded-md px-3 py-2 text-xs font-mono text-white placeholder:text-white/25 focus:outline-none focus:border-cyan-400/30"/>
            <button type="submit" className="px-3 py-2 rounded-md bg-white text-black text-xs font-semibold flex items-center gap-1 hover:bg-white/90">
              <Send size={12}/> Send
            </button>
          </form>
        </aside>
      </div>
    </div>
  )
}

function Row({ l, v, u, hi }) {
  return (
    <div className={`flex items-center justify-between py-0.5 text-[11px] font-mono border-b border-white/5 last:border-b-0 ${hi ? 'text-cyan-200' : 'text-white/60'}`}>
      <span className="uppercase tracking-widest text-white/40">{l}</span>
      <span>{v}<span className="text-white/30 ml-1">{u}</span></span>
    </div>
  )
}

// ───── Renderer ─────
function render(canvas, s, now, opts) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width, H = canvas.height

  const camCx = clamp(s.x, VIEW.w/2, WORLD.w - VIEW.w/2)
  const camCy = clamp(s.y, VIEW.h/2, WORLD.h - VIEW.h/2)
  const camX0 = camCx - VIEW.w/2, camY0 = camCy - VIEW.h/2
  const pm = Math.min(W/VIEW.w, H/VIEW.h)
  const tx = (wx) => (wx - camX0) * pm
  const ty = (wy) => (wy - camY0) * pm

  ctx.fillStyle = '#030608'
  ctx.fillRect(0, 0, W, H)

  // Grid
  const drawGrid = (step, stroke) => {
    ctx.strokeStyle = stroke; ctx.lineWidth = 1
    const minX = Math.floor(camX0/step)*step
    const maxX = minX + VIEW.w + step
    const minY = Math.floor(camY0/step)*step
    const maxY = minY + VIEW.h + step
    for (let x = minX; x <= maxX; x += step) { ctx.beginPath(); ctx.moveTo(tx(x),0); ctx.lineTo(tx(x),H); ctx.stroke() }
    for (let y = minY; y <= maxY; y += step) { ctx.beginPath(); ctx.moveTo(0,ty(y)); ctx.lineTo(W,ty(y)); ctx.stroke() }
  }
  drawGrid(0.5, 'rgba(120,200,240,0.05)')
  drawGrid(1.0, 'rgba(120,200,240,0.12)')

  // World bounds
  ctx.strokeStyle = 'rgba(120,200,240,0.25)'; ctx.lineWidth = 1.5
  ctx.strokeRect(tx(0), ty(0), WORLD.w*pm, WORLD.h*pm)

  // Obstacles
  for (const o of OBSTACLES) {
    const x = tx(o.x - o.w/2), y = ty(o.y - o.h/2)
    const w = o.w*pm, h = o.h*pm
    if (x+w < 0 || x > W || y+h < 0 || y > H) continue
    ctx.fillStyle = 'rgba(120,140,180,0.18)'
    ctx.strokeStyle = 'rgba(180,210,240,0.45)'; ctx.lineWidth = 1.5
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h)
    if (w > 30 && h > 12) {
      ctx.fillStyle = 'rgba(180,210,240,0.55)'
      ctx.font = '10px JetBrains Mono, monospace'
      ctx.fillText(o.label, x+4, y+13)
    }
  }

  // NPCs
  for (const n of NPCS) {
    const nx = tx(n.x), ny = ty(n.y)
    if (nx < -20 || nx > W+20 || ny < -20 || ny > H+20) continue
    const r = R.radius * pm * 0.85
    ctx.fillStyle = '#0a0f14'; ctx.strokeStyle = n.color; ctx.lineWidth = 2
    ctx.beginPath(); ctx.arc(nx, ny, r, 0, Math.PI*2); ctx.fill(); ctx.stroke()
    const pulse = 0.5 + Math.sin(now/700 + n.x)*0.2
    ctx.strokeStyle = `${n.color}55`; ctx.lineWidth = 1
    ctx.beginPath(); ctx.arc(nx, ny, r*(1.8+pulse*0.4), 0, Math.PI*2); ctx.stroke()
    ctx.font = '10px JetBrains Mono, monospace'
    const lbl = `${n.name} · ${n.role}`
    const tw = ctx.measureText(lbl).width
    ctx.fillStyle = 'rgba(0,0,0,0.5)'
    ctx.fillRect(nx-tw/2-4, ny-r-18, tw+8, 14)
    ctx.fillStyle = n.color
    ctx.fillText(lbl, nx-tw/2, ny-r-7)
  }

  // Detection ring
  const rx = tx(s.x), ry = ty(s.y)
  const dR = R.detection_r * pm
  const dg = ctx.createRadialGradient(rx, ry, dR*0.7, rx, ry, dR)
  dg.addColorStop(0, 'rgba(34,224,255,0)'); dg.addColorStop(1, 'rgba(34,224,255,0.07)')
  ctx.fillStyle = dg; ctx.beginPath(); ctx.arc(rx, ry, dR, 0, Math.PI*2); ctx.fill()

  // LIDAR rays
  if (opts.showLidar && s.lidar) {
    const fov = R.lidar_fov
    const N = s.lidar.length
    // rays
    for (let i = 0; i < N; i++) {
      const a = s.heading - fov/2 + (i/(N-1)) * fov
      const range = s.lidar[i]
      const hit = range < R.lidar_range - 0.1
      const ex = s.x + Math.cos(a) * range
      const ey = s.y + Math.sin(a) * range
      ctx.strokeStyle = hit ? 'rgba(244,63,94,0.35)' : 'rgba(244,63,94,0.08)'
      ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(tx(ex), ty(ey)); ctx.stroke()
    }
    // hit points
    for (let i = 0; i < N; i++) {
      const a = s.heading - fov/2 + (i/(N-1)) * fov
      const range = s.lidar[i]
      if (range >= R.lidar_range - 0.1) continue
      const ex = s.x + Math.cos(a) * range
      const ey = s.y + Math.sin(a) * range
      ctx.fillStyle = 'rgba(255,80,100,0.9)'
      ctx.beginPath(); ctx.arc(tx(ex), ty(ey), 1.8, 0, Math.PI*2); ctx.fill()
    }
  }

  // Trail (ground truth)
  if (s.trail.length > 1) {
    ctx.lineWidth = 2
    for (let i = 1; i < s.trail.length; i++) {
      const age = (now - s.trail[i].ts) / 6000
      const alpha = Math.max(0, 0.9 - age)
      if (alpha <= 0) continue
      ctx.strokeStyle = `rgba(0,220,255,${alpha*0.55})`
      ctx.beginPath()
      ctx.moveTo(tx(s.trail[i-1].x), ty(s.trail[i-1].y))
      ctx.lineTo(tx(s.trail[i].x), ty(s.trail[i].y))
      ctx.stroke()
    }
  }

  // Odometry ghost
  if (opts.showOdom) {
    const ox = tx(s.odomX), oy = ty(s.odomY), rr = R.radius*pm
    ctx.strokeStyle = 'rgba(245,158,11,0.7)'; ctx.setLineDash([5,4]); ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.arc(ox, oy, rr, 0, Math.PI*2); ctx.stroke()
    const hxo = ox + Math.cos(s.odomHeading)*rr*1.4
    const hyo = oy + Math.sin(s.odomHeading)*rr*1.4
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(hxo, hyo); ctx.stroke()
    ctx.setLineDash([])
    // Label
    ctx.fillStyle = 'rgba(245,158,11,0.9)'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.fillText('/odom (est)', ox-26, oy-rr-6)
  }

  // Robot (ground truth)
  const rr = R.radius*pm
  const grad = ctx.createRadialGradient(rx, ry, rr*0.6, rx, ry, rr*3)
  grad.addColorStop(0, 'rgba(0,220,255,0.25)'); grad.addColorStop(1, 'rgba(0,220,255,0)')
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(rx, ry, rr*3, 0, Math.PI*2); ctx.fill()
  ctx.fillStyle = '#0a0f14'; ctx.strokeStyle = '#22e0ff'; ctx.lineWidth = 2
  ctx.beginPath(); ctx.arc(rx, ry, rr, 0, Math.PI*2); ctx.fill(); ctx.stroke()
  const hx = rx + Math.cos(s.heading)*rr*1.5
  const hy = ry + Math.sin(s.heading)*rr*1.5
  ctx.strokeStyle = '#22e0ff'; ctx.lineWidth = 3; ctx.lineCap = 'round'
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(hx, hy); ctx.stroke()
  // wheels
  ctx.fillStyle = 'rgba(180,220,255,0.4)'
  const px = -Math.sin(s.heading), py = Math.cos(s.heading)
  for (const side of [-1,1]) {
    const wwx = rx + px*side*rr*0.9, wwy = ry + py*side*rr*0.9
    ctx.beginPath(); ctx.arc(wwx, wwy, rr*0.2, 0, Math.PI*2); ctx.fill()
  }
  ctx.fillStyle = 'rgba(34,224,255,0.9)'
  ctx.font = 'bold 11px JetBrains Mono, monospace'
  ctx.fillText('EDEN-01 /tf base_link', rx-60, ry-rr-9)

  // Minimap
  const mmW = Math.min(280, W*0.22)
  const mmH = mmW * (WORLD.h / WORLD.w)
  const mmX = W - mmW - 12, mmY = 12
  const mmPm = mmW / WORLD.w
  ctx.fillStyle = 'rgba(0,0,0,0.75)'
  ctx.fillRect(mmX-6, mmY-6, mmW+12, mmH+30)
  ctx.strokeStyle = 'rgba(120,200,240,0.3)'
  ctx.strokeRect(mmX-6, mmY-6, mmW+12, mmH+30)
  ctx.fillStyle = 'rgba(120,200,240,0.6)'
  ctx.font = '9px JetBrains Mono, monospace'
  ctx.fillText('MINIMAP · /map', mmX, mmY-10)
  ctx.fillStyle = 'rgba(10,20,30,0.95)'
  ctx.fillRect(mmX, mmY, mmW, mmH)
  for (const o of OBSTACLES) {
    ctx.fillStyle = 'rgba(180,210,240,0.55)'
    ctx.fillRect(mmX + (o.x - o.w/2)*mmPm, mmY + (o.y - o.h/2)*mmPm, o.w*mmPm, o.h*mmPm)
  }
  for (const n of NPCS) {
    ctx.fillStyle = n.color
    ctx.beginPath(); ctx.arc(mmX + n.x*mmPm, mmY + n.y*mmPm, 3, 0, Math.PI*2); ctx.fill()
  }
  ctx.fillStyle = '#22e0ff'
  ctx.beginPath(); ctx.arc(mmX + s.x*mmPm, mmY + s.y*mmPm, 3.5, 0, Math.PI*2); ctx.fill()
  ctx.strokeStyle = '#22e0ff'; ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(mmX + s.x*mmPm, mmY + s.y*mmPm)
  ctx.lineTo(mmX + (s.x + Math.cos(s.heading)*0.7)*mmPm, mmY + (s.y + Math.sin(s.heading)*0.7)*mmPm)
  ctx.stroke()
  // odom ghost on minimap
  if (opts.showOdom) {
    ctx.fillStyle = 'rgba(245,158,11,0.9)'
    ctx.beginPath(); ctx.arc(mmX + s.odomX*mmPm, mmY + s.odomY*mmPm, 2.5, 0, Math.PI*2); ctx.fill()
  }
  // viewport rect
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1; ctx.setLineDash([3,3])
  ctx.strokeRect(mmX + camX0*mmPm, mmY + camY0*mmPm, VIEW.w*mmPm, VIEW.h*mmPm)
  ctx.setLineDash([])
}
