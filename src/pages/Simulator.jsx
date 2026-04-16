import React, { useEffect, useRef, useState, useMemo, Suspense } from 'react'
import { Link } from 'react-router-dom'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Html, Grid, Environment, PerspectiveCamera } from '@react-three/drei'
import { motion, AnimatePresence } from 'framer-motion'
import * as THREE from 'three'
import { Cpu, ArrowLeft, Play, Pause, Wifi, Send, Brain, X, Check, Radar, Activity, Radio, Box as BoxIcon, Orbit } from 'lucide-react'
import { openSimBusReceiver } from '../lib/simBridge'
import { classifyAction } from '../lib/cognitiveLayer'
import { publish, listTopics, Twist, Odometry, LaserScan, TFMessage } from '../lib/rosTopics'
import "@fontsource/jetbrains-mono"

// ───── World ─────
const WORLD = { w: 30, h: 20 }
const R = {
  radius: 0.28,
  wheel_r: 0.08,
  wheel_base: 0.35,
  max_lin: 0.6,
  max_ang: 2.0,
  max_lin_accel: 1.2,
  max_ang_accel: 4.0,
  motor_tau: 0.14,
  lidar_range: 6.0,
  lidar_fov: Math.PI * 1.5,
  lidar_rays: 180,
  detection_r: 4.0,
  odom_noise: 0.025,
}
const OBSTACLES = [
  { x: 4,  y: 1.2, w: 6, h: 0.4, label: 'north wall',       height: 1.4 },
  { x: 14, y: 1.2, w: 10, h: 0.4, label: 'north wall',      height: 1.4 },
  { x: 25, y: 1.2, w: 5, h: 0.4, label: 'north wall',       height: 1.4 },
  { x: 6,  y: 18.8, w: 8, h: 0.4, label: 'south wall',      height: 1.4 },
  { x: 20, y: 18.8, w: 8, h: 0.4, label: 'south wall',      height: 1.4 },
  { x: 6,  y: 6,   w: 2.4, h: 0.6, label: 'workbench A',    height: 0.9 },
  { x: 6,  y: 9,   w: 2.4, h: 0.6, label: 'workbench B',    height: 0.9 },
  { x: 9,  y: 7.5, w: 0.6, h: 3,   label: 'cable trunk',    height: 0.5 },
  { x: 3,  y: 14,  w: 0.8, h: 3,   label: 'charging dock',  height: 1.8, color: '#22e0ff' },
  { x: 27, y: 4,   w: 1.2, h: 0.8, label: 'parts bin',      height: 0.7 },
  { x: 27, y: 6,   w: 1.2, h: 0.8, label: 'parts bin',      height: 0.7 },
  { x: 14, y: 8,   w: 1.0, h: 1.0, label: 'crate',          height: 0.6 },
  { x: 17, y: 11,  w: 0.8, h: 0.8, label: 'box',            height: 0.5 },
  { x: 12, y: 13,  w: 1.4, h: 0.4, label: 'bench',          height: 0.5 },
  { x: 22, y: 10,  w: 0.6, h: 2.6, label: 'server rack',    height: 1.9, color: '#64748b' },
  { x: 18, y: 16,  w: 2.0, h: 0.5, label: 'bench',          height: 0.5 },
  { x: 24, y: 15,  w: 0.8, h: 0.8, label: 'marker',         height: 0.3 },
  { x: 10, y: 16.5,w: 0.4, h: 0.4, label: 'cone',           height: 0.4 },
  { x: 13, y: 16.5,w: 0.4, h: 0.4, label: 'cone',           height: 0.4 },
]
const NPCS = [
  { name: 'EDEN-02',      x: 7.5,  y: 14.5, color: '#a78bfa', role: 'sibling'   },
  { name: 'delivery-bot', x: 26,   y: 12.0, color: '#f59e0b', role: 'logistics' },
  { name: 'scout-01',     x: 16,   y: 4.5,  color: '#34d399', role: 'scout'     },
  { name: 'inspector-A3', x: 21,   y: 14.0, color: '#06b6d4', role: 'inspector' },
  { name: 'sweeper',      x: 11,   y: 10.5, color: '#f472b6', role: 'clean'     },
]

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)

function circleIntersectsRect(cx, cy, r, rect) {
  const nx = clamp(cx, rect.x - rect.w/2, rect.x + rect.w/2)
  const ny = clamp(cy, rect.y - rect.h/2, rect.y + rect.h/2)
  const dx = cx-nx, dy = cy-ny
  return dx*dx + dy*dy < r*r
}

function raycastRect(ox, oy, dx, dy, rect) {
  const minX = rect.x - rect.w/2, maxX = rect.x + rect.w/2
  const minY = rect.y - rect.h/2, maxY = rect.y + rect.h/2
  const inv_dx = dx !== 0 ? 1/dx : 1e9
  const inv_dy = dy !== 0 ? 1/dy : 1e9
  const tx1 = (minX-ox)*inv_dx, tx2 = (maxX-ox)*inv_dx
  const tmin = Math.min(tx1, tx2), tmax = Math.max(tx1, tx2)
  const ty1 = (minY-oy)*inv_dy, ty2 = (maxY-oy)*inv_dy
  const tymin = Math.min(ty1, ty2), tymax = Math.max(ty1, ty2)
  const entry = Math.max(tmin, tymin), exit = Math.min(tmax, tymax)
  if (entry > exit || exit < 0) return null
  return entry < 0 ? null : entry
}

function raycastBounds(ox, oy, dx, dy) {
  let best = Infinity
  if (dx > 0) { const t = (WORLD.w-ox)/dx; if (t > 0 && t < best) best = t }
  if (dx < 0) { const t = (0-ox)/dx; if (t > 0 && t < best) best = t }
  if (dy > 0) { const t = (WORLD.h-oy)/dy; if (t > 0 && t < best) best = t }
  if (dy < 0) { const t = (0-oy)/dy; if (t > 0 && t < best) best = t }
  return best === Infinity ? null : best
}

function simulateLidar(s) {
  const N = R.lidar_rays, fov = R.lidar_fov
  const ranges = new Array(N)
  for (let i = 0; i < N; i++) {
    const a = s.heading - fov/2 + (i/(N-1))*fov
    const dx = Math.cos(a), dy = Math.sin(a)
    let best = R.lidar_range
    for (const o of OBSTACLES) {
      const t = raycastRect(s.x, s.y, dx, dy, o)
      if (t != null && t < best) best = t
    }
    const tw = raycastBounds(s.x, s.y, dx, dy)
    if (tw != null && tw < best) best = tw
    ranges[i] = clamp(best + (Math.random()-0.5)*0.03, 0.05, R.lidar_range)
  }
  return ranges
}

// ───── 3D scene subcomponents ─────
// Coordinate convention: world is XY plane, Z is up. Three.js uses Y-up by
// default; we place everything on the XZ plane and flip signs so camera
// "bird's eye" makes sense. We treat three.js coords as: x=world.x, y=height, z=world.y.

function Floor() {
  return (
    <>
      {/* Main ground plane */}
      <mesh rotation={[-Math.PI/2, 0, 0]} position={[WORLD.w/2, 0, WORLD.h/2]}>
        <planeGeometry args={[WORLD.w, WORLD.h]} />
        <meshStandardMaterial color="#0b1220" roughness={0.9} metalness={0.1} />
      </mesh>
      {/* Grid — via drei */}
      <Grid
        position={[WORLD.w/2, 0.001, WORLD.h/2]}
        args={[WORLD.w, WORLD.h]}
        cellSize={0.5}
        cellThickness={0.6}
        cellColor="#1e3a8a"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#22d3ee"
        fadeDistance={35}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={false}
      />
      {/* Origin axes */}
      <group position={[0, 0.01, 0]}>
        <mesh position={[0.3, 0, 0]}>
          <boxGeometry args={[0.6, 0.02, 0.05]} />
          <meshBasicMaterial color="#ef4444" />
        </mesh>
        <mesh position={[0, 0, 0.3]}>
          <boxGeometry args={[0.05, 0.02, 0.6]} />
          <meshBasicMaterial color="#22c55e" />
        </mesh>
      </group>
    </>
  )
}

function Walls() {
  // world perimeter
  const t = 0.05 // thickness
  const h = 0.3  // height
  return (
    <group>
      <mesh position={[WORLD.w/2, h/2, 0]}><boxGeometry args={[WORLD.w, h, t]}/><meshStandardMaterial color="#1e3a5f" transparent opacity={0.6}/></mesh>
      <mesh position={[WORLD.w/2, h/2, WORLD.h]}><boxGeometry args={[WORLD.w, h, t]}/><meshStandardMaterial color="#1e3a5f" transparent opacity={0.6}/></mesh>
      <mesh position={[0, h/2, WORLD.h/2]}><boxGeometry args={[t, h, WORLD.h]}/><meshStandardMaterial color="#1e3a5f" transparent opacity={0.6}/></mesh>
      <mesh position={[WORLD.w, h/2, WORLD.h/2]}><boxGeometry args={[t, h, WORLD.h]}/><meshStandardMaterial color="#1e3a5f" transparent opacity={0.6}/></mesh>
    </group>
  )
}

function Obstacles() {
  return OBSTACLES.map((o, i) => (
    <group key={i} position={[o.x, 0, o.y]}>
      <mesh position={[0, o.height/2, 0]}>
        <boxGeometry args={[o.w, o.height, o.h]} />
        <meshStandardMaterial
          color={o.color || '#5b7392'}
          roughness={0.7}
          metalness={0.2}
          transparent
          opacity={0.85}
        />
      </mesh>
      {/* Wireframe accent */}
      <lineSegments position={[0, o.height/2, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(o.w, o.height, o.h)]} />
        <lineBasicMaterial color={o.color || '#b3cce6'} transparent opacity={0.5} />
      </lineSegments>
      <Html position={[0, o.height + 0.15, 0]} center distanceFactor={14} zIndexRange={[0, 0]}>
        <div className="px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-widest bg-black/60 text-white/70 rounded whitespace-nowrap pointer-events-none">
          {o.label}
        </div>
      </Html>
    </group>
  ))
}

function NpcRobots({ npcsState }) {
  return npcsState.map((n) => (
    <group key={n.name} position={[n.x, 0, n.y]} rotation={[0, -n.heading, 0]}>
      <mesh position={[0, 0.18, 0]} castShadow>
        <cylinderGeometry args={[R.radius * 0.8, R.radius * 0.9, 0.35, 16]} />
        <meshStandardMaterial color={n.color} emissive={n.color} emissiveIntensity={0.25} roughness={0.4} metalness={0.3} />
      </mesh>
      {/* sensor dot */}
      <mesh position={[0, 0.4, 0]}>
        <sphereGeometry args={[0.04, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      {/* heading arrow */}
      <mesh position={[R.radius, 0.18, 0]} rotation={[0, 0, 0]}>
        <coneGeometry args={[0.06, 0.18, 8]} />
        <meshBasicMaterial color={n.color} />
      </mesh>
      <Html position={[0, 0.8, 0]} center distanceFactor={12} zIndexRange={[0, 0]}>
        <div className="px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-widest rounded whitespace-nowrap pointer-events-none"
             style={{ background: 'rgba(0,0,0,0.55)', color: n.color }}>
          {n.name} · {n.role}
        </div>
      </Html>
    </group>
  ))
}

function RobotMesh({ stateRef, showOdom }) {
  const bodyRef = useRef()
  const odomRef = useRef()
  const trailRef = useRef()
  const detectionRef = useRef()

  useFrame(() => {
    const s = stateRef.current
    if (bodyRef.current) {
      bodyRef.current.position.x = s.x
      bodyRef.current.position.z = s.y
      bodyRef.current.rotation.y = -s.heading
    }
    if (odomRef.current) {
      odomRef.current.position.x = s.odomX
      odomRef.current.position.z = s.odomY
      odomRef.current.rotation.y = -s.odomHeading
      odomRef.current.visible = showOdom
    }
    if (detectionRef.current) {
      detectionRef.current.position.x = s.x
      detectionRef.current.position.z = s.y
    }
    // Update trail geometry
    if (trailRef.current && s.trail.length >= 2) {
      const positions = new Float32Array(s.trail.length * 3)
      for (let i = 0; i < s.trail.length; i++) {
        positions[i*3]   = s.trail[i].x
        positions[i*3+1] = 0.02
        positions[i*3+2] = s.trail[i].y
      }
      trailRef.current.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      trailRef.current.geometry.setDrawRange(0, s.trail.length)
      trailRef.current.geometry.attributes.position.needsUpdate = true
    }
  })

  return (
    <>
      {/* Robot body */}
      <group ref={bodyRef}>
        {/* chassis cylinder */}
        <mesh position={[0, 0.16, 0]} castShadow>
          <cylinderGeometry args={[R.radius, R.radius * 1.05, 0.3, 24]} />
          <meshStandardMaterial color="#0a0f14" roughness={0.35} metalness={0.6} />
        </mesh>
        {/* cyan ring */}
        <mesh position={[0, 0.32, 0]} rotation={[Math.PI/2, 0, 0]}>
          <torusGeometry args={[R.radius * 0.85, 0.02, 12, 48]} />
          <meshBasicMaterial color="#22e0ff" />
        </mesh>
        {/* sensor post (lidar) */}
        <mesh position={[0, 0.45, 0]} castShadow>
          <cylinderGeometry args={[0.05, 0.05, 0.08, 16]} />
          <meshStandardMaterial color="#0a0f14" roughness={0.2} metalness={0.9} />
        </mesh>
        <mesh position={[0, 0.5, 0]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshBasicMaterial color="#22e0ff" />
        </mesh>
        {/* heading arrow */}
        <mesh position={[R.radius + 0.05, 0.16, 0]} rotation={[0, 0, -Math.PI/2]}>
          <coneGeometry args={[0.1, 0.2, 12]} />
          <meshBasicMaterial color="#22e0ff" />
        </mesh>
        {/* wheels */}
        {[-1, 1].map((s) => (
          <mesh key={s} position={[0, 0.08, s * R.wheel_base/2]} rotation={[0, 0, Math.PI/2]}>
            <cylinderGeometry args={[R.wheel_r, R.wheel_r, 0.05, 12]} />
            <meshStandardMaterial color="#0f172a" roughness={0.95} />
          </mesh>
        ))}
        {/* label */}
        <Html position={[0, 0.85, 0]} center distanceFactor={10} zIndexRange={[0, 0]}>
          <div className="px-2 py-0.5 text-[9px] font-mono uppercase tracking-widest bg-cyan-500/20 text-cyan-200 border border-cyan-400/40 rounded whitespace-nowrap pointer-events-none">
            EDEN-01 · /tf base_link
          </div>
        </Html>
      </group>

      {/* Odometry ghost */}
      <group ref={odomRef}>
        <mesh position={[0, 0.16, 0]}>
          <cylinderGeometry args={[R.radius * 1.1, R.radius * 1.1, 0.3, 24]} />
          <meshBasicMaterial color="#f59e0b" wireframe transparent opacity={0.45} />
        </mesh>
      </group>

      {/* Trail */}
      <line ref={trailRef}>
        <bufferGeometry />
        <lineBasicMaterial color="#22d3ee" transparent opacity={0.7} linewidth={2} />
      </line>

      {/* Detection ring */}
      <mesh ref={detectionRef} position={[0, 0.005, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <ringGeometry args={[R.detection_r - 0.04, R.detection_r, 64]} />
        <meshBasicMaterial color="#22e0ff" transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>
    </>
  )
}

function LidarRays({ stateRef, show }) {
  const linesRef = useRef()
  const pointsRef = useRef()

  const geom = useMemo(() => {
    const lineGeom = new THREE.BufferGeometry()
    const N = R.lidar_rays
    lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 2 * 3), 3))
    const ptGeom = new THREE.BufferGeometry()
    ptGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3))
    return { lineGeom, ptGeom }
  }, [])

  useFrame(() => {
    if (!show) return
    const s = stateRef.current
    const lidar = s.lidar
    if (!lidar) return
    const N = R.lidar_rays, fov = R.lidar_fov
    const linePos = linesRef.current?.geometry.attributes.position.array
    const ptPos = pointsRef.current?.geometry.attributes.position.array
    let hitCount = 0
    for (let i = 0; i < N; i++) {
      const a = s.heading - fov/2 + (i/(N-1))*fov
      const range = lidar[i]
      const ex = s.x + Math.cos(a) * range
      const ey = s.y + Math.sin(a) * range
      const base = i * 6
      if (linePos) {
        linePos[base]   = s.x; linePos[base+1] = 0.45; linePos[base+2] = s.y
        linePos[base+3] = ex;  linePos[base+4] = 0.45; linePos[base+5] = ey
      }
      if (range < R.lidar_range - 0.1 && ptPos) {
        ptPos[hitCount*3]   = ex
        ptPos[hitCount*3+1] = 0.45
        ptPos[hitCount*3+2] = ey
        hitCount++
      }
    }
    if (linesRef.current) {
      linesRef.current.geometry.attributes.position.needsUpdate = true
    }
    if (pointsRef.current) {
      pointsRef.current.geometry.attributes.position.needsUpdate = true
      pointsRef.current.geometry.setDrawRange(0, hitCount)
    }
  })

  if (!show) return null
  return (
    <>
      <lineSegments ref={linesRef}>
        <primitive object={geom.lineGeom} attach="geometry" />
        <lineBasicMaterial color="#ef4444" transparent opacity={0.18} />
      </lineSegments>
      <points ref={pointsRef}>
        <primitive object={geom.ptGeom} attach="geometry" />
        <pointsMaterial color="#ff5566" size={0.08} sizeAttenuation />
      </points>
    </>
  )
}

function CameraRig({ stateRef, follow }) {
  const { camera } = useThree()
  useFrame(() => {
    if (!follow) return
    const s = stateRef.current
    // isometric behind-and-above follow
    const desiredX = s.x - 6 * Math.cos(s.heading)
    const desiredZ = s.y - 6 * Math.sin(s.heading)
    camera.position.x += (desiredX - camera.position.x) * 0.05
    camera.position.y += (5 - camera.position.y) * 0.05
    camera.position.z += (desiredZ - camera.position.z) * 0.05
    camera.lookAt(s.x, 0.4, s.y)
  })
  return null
}

// ───── Minimap 2D overlay ─────
function Minimap({ stateRef }) {
  const canvasRef = useRef()
  useEffect(() => {
    let raf
    function draw() {
      const c = canvasRef.current
      if (!c) { raf = requestAnimationFrame(draw); return }
      const ctx = c.getContext('2d')
      const W = c.width, H = c.height
      const pm = W / WORLD.w
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = 'rgba(8, 16, 28, 0.9)'
      ctx.fillRect(0, 0, W, H)
      // obstacles
      for (const o of OBSTACLES) {
        ctx.fillStyle = 'rgba(180,210,240,0.55)'
        ctx.fillRect((o.x - o.w/2) * pm, (o.y - o.h/2) * pm, o.w * pm, o.h * pm)
      }
      // npcs
      for (const n of NPCS) {
        ctx.fillStyle = n.color
        ctx.beginPath(); ctx.arc(n.x * pm, n.y * pm, 3, 0, Math.PI*2); ctx.fill()
      }
      const s = stateRef.current
      ctx.fillStyle = '#22e0ff'
      ctx.beginPath(); ctx.arc(s.x * pm, s.y * pm, 4, 0, Math.PI*2); ctx.fill()
      ctx.strokeStyle = '#22e0ff'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(s.x * pm, s.y * pm)
      ctx.lineTo((s.x + Math.cos(s.heading)*0.9) * pm, (s.y + Math.sin(s.heading)*0.9) * pm)
      ctx.stroke()
      ctx.fillStyle = 'rgba(245,158,11,0.8)'
      ctx.beginPath(); ctx.arc(s.odomX * pm, s.odomY * pm, 2.5, 0, Math.PI*2); ctx.fill()
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])
  const W = 240, H = W * (WORLD.h / WORLD.w)
  return (
    <div className="absolute top-3 right-3 pointer-events-none">
      <div className="mb-1 text-[9px] font-mono uppercase tracking-widest text-white/50">MINIMAP · /map</div>
      <canvas ref={canvasRef} width={W} height={H} className="rounded border border-white/15 shadow-2xl" />
    </div>
  )
}

// ───── Main ─────
export default function Simulator() {
  const stateRef = useRef({
    x: 2.5, y: 10, heading: 0,
    linVel: 0, angVel: 0,
    cmdLin: 0, cmdAng: 0, cmdUntil: 0,
    wheelL: 0, wheelR: 0,
    odomX: 2.5, odomY: 10, odomHeading: 0,
    lidar: new Array(R.lidar_rays).fill(R.lidar_range),
    trail: [],
    lastTick: performance.now(),
    lastOdomPub: 0, lastScanPub: 0, lastTFPub: 0,
    collisions: 0,
  })
  const [telemetry, setTelemetry] = useState({ x: 2.5, y: 10, heading: 0, linVel: 0, angVel: 0, wheelL: 0, wheelR: 0, odomDrift: 0 })
  const [topicsSnap, setTopicsSnap] = useState([])
  const [log, setLog] = useState([])
  const [running, setRunning] = useState(true)
  const [manualInput, setManualInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [thinking, setThinking] = useState(null)
  const [useLLM, setUseLLM] = useState(true)
  const [showLidar, setShowLidar] = useState(true)
  const [showOdom, setShowOdom] = useState(true)
  const [followCam, setFollowCam] = useState(true)
  const [npcsState, setNpcsState] = useState(NPCS.map((n) => ({ ...n, heading: Math.random() * Math.PI * 2 })))

  useEffect(() => {
    const close = openSimBusReceiver((payload) => { applyAction(payload.action, 'eden') })
    setConnected(true)
    return () => { close(); setConnected(false) }
  }, [])

  // Keyboard
  useEffect(() => {
    function onKey(e) {
      if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return
      const s = stateRef.current
      let cmd = null
      if (e.code === 'ArrowUp')    cmd = { l:  0.45, a: 0,    d: 400 }
      if (e.code === 'ArrowDown')  cmd = { l: -0.35, a: 0,    d: 400 }
      if (e.code === 'ArrowLeft')  cmd = { l: 0,     a: 1.0,  d: 300 }
      if (e.code === 'ArrowRight') cmd = { l: 0,     a: -1.0, d: 300 }
      if (e.code === 'Space')      cmd = { l: 0,     a: 0,    d: 100 }
      if (cmd) {
        e.preventDefault()
        s.cmdLin = cmd.l; s.cmdAng = cmd.a; s.cmdUntil = performance.now() + cmd.d
      }
      if (e.code === 'KeyR') {
        s.x = 2.5; s.y = 10; s.heading = 0
        s.linVel = 0; s.angVel = 0; s.cmdLin = 0; s.cmdAng = 0; s.cmdUntil = 0
        s.odomX = 2.5; s.odomY = 10; s.odomHeading = 0
        s.trail = []; s.collisions = 0
      }
      if (e.code === 'KeyC') setFollowCam((f) => !f)
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
      const entry = { ts, source, action: rawAction, decision: parsed ? 'execute' : 'refuse', reason: parsed ? 'direct' : 'unparseable', linear: parsed?.linear, angular: parsed?.angular, duration: parsed?.duration, model: 'regex', ms: 0 }
      setLog((p) => [entry, ...p].slice(0, 50))
      if (!parsed) return
      commitCmd(parsed.linear, parsed.angular, parsed.duration, parsed.stop)
      return
    }
    setThinking({ action: rawAction, source })
    const obstacles = OBSTACLES.filter((o) => dist({x:s.x,y:s.y}, o) < 8)
    const npcs = npcsState.filter((n) => dist({x:s.x,y:s.y}, n) < R.detection_r * 1.5)
    const history = log.slice(0, 4).map((l) => ({ source: l.source, action: l.action, decision: l.decision }))
    const res = await classifyAction({
      action: rawAction,
      robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
      obstacles, npcs, history,
    })
    setThinking(null)
    const entry = { ts, source, action: rawAction, decision: res.decision, reason: res.reason, linear: res.linear, angular: res.angular, duration: res.duration, model: res.model, ms: res.ms }
    setLog((p) => [entry, ...p].slice(0, 50))
    if (res.decision === 'refuse' || res.linear === null) return
    commitCmd(res.linear, res.angular, res.duration, false)
  }

  function commitCmd(lin, ang, dur, stop) {
    const s = stateRef.current
    const now = performance.now()
    if (stop) { s.cmdLin = 0; s.cmdAng = 0; s.cmdUntil = now + 100 }
    else { s.cmdLin = lin ?? 0; s.cmdAng = ang ?? 0; s.cmdUntil = now + (dur ?? 2) * 1000 }
    publish('/cmd_vel', Twist(s.cmdLin, s.cmdAng))
  }

  // Physics (RAF)
  useEffect(() => {
    let raf = 0
    function tick(now) {
      const s = stateRef.current
      const dt = Math.min(0.05, (now - s.lastTick) / 1000)
      s.lastTick = now
      if (running) {
        const targetLin = now > s.cmdUntil ? 0 : s.cmdLin
        const targetAng = now > s.cmdUntil ? 0 : s.cmdAng
        const alpha = 1 - Math.exp(-dt / R.motor_tau)
        let newLin = s.linVel + (targetLin - s.linVel) * alpha
        let newAng = s.angVel + (targetAng - s.angVel) * alpha
        const maxDlin = R.max_lin_accel * dt, maxDang = R.max_ang_accel * dt
        newLin = clamp(newLin, s.linVel - maxDlin, s.linVel + maxDlin)
        newAng = clamp(newAng, s.angVel - maxDang, s.angVel + maxDang)
        newLin = clamp(newLin, -R.max_lin, R.max_lin)
        newAng = clamp(newAng, -R.max_ang, R.max_ang)
        s.linVel = newLin; s.angVel = newAng
        s.wheelL = (s.linVel - R.wheel_base/2 * s.angVel) / R.wheel_r
        s.wheelR = (s.linVel + R.wheel_base/2 * s.angVel) / R.wheel_r

        const nextHeading = s.heading + s.angVel * dt
        const nextX = s.x + s.linVel * Math.cos(nextHeading) * dt
        const nextY = s.y + s.linVel * Math.sin(nextHeading) * dt
        const hit = OBSTACLES.some((o) => circleIntersectsRect(nextX, nextY, R.radius, o))
                 || nextX < R.radius || nextX > WORLD.w - R.radius
                 || nextY < R.radius || nextY > WORLD.h - R.radius
                 || npcsState.some((n) => dist({x:nextX,y:nextY}, n) < R.radius + 0.35)
        if (!hit) { s.x = nextX; s.y = nextY }
        else { s.linVel = 0; s.collisions += 1 }
        s.heading = nextHeading

        const n = R.odom_noise
        const oL = s.linVel * (1 + (Math.random()-0.5)*n)
        const oA = s.angVel * (1 + (Math.random()-0.5)*n*2)
        s.odomHeading += oA * dt
        s.odomX += oL * Math.cos(s.odomHeading) * dt
        s.odomY += oL * Math.sin(s.odomHeading) * dt

        if (Math.abs(s.linVel) > 0.02 || Math.abs(s.angVel) > 0.05) {
          s.trail.push({ x: s.x, y: s.y, ts: now })
          if (s.trail.length > 800) s.trail.shift()
        }

        s.lidar = simulateLidar(s)

        if (now - s.lastOdomPub > 20) {
          publish('/odom', Odometry({ x: s.odomX, y: s.odomY, heading: s.odomHeading, linVel: s.linVel, angVel: s.angVel, ts: now }))
          s.lastOdomPub = now
        }
        if (now - s.lastTFPub > 20) {
          publish('/tf', TFMessage({ x: s.x, y: s.y, heading: s.heading, ts: now }))
          s.lastTFPub = now
        }
        if (now - s.lastScanPub > 100) {
          publish('/scan', LaserScan({ angle_min: -R.lidar_fov/2, angle_max: R.lidar_fov/2, ranges: s.lidar, range_max: R.lidar_range, ts: now }))
          s.lastScanPub = now
        }
      }
      setTelemetry({ x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel, wheelL: s.wheelL, wheelR: s.wheelR, odomDrift: Math.hypot(s.odomX - s.x, s.odomY - s.y) })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [running, npcsState])

  // Slow NPC wander
  useEffect(() => {
    const h = setInterval(() => {
      setNpcsState((prev) => prev.map((n) => {
        const jitterH = n.heading + (Math.random() - 0.5) * 0.15
        const dx = Math.cos(jitterH) * 0.1, dy = Math.sin(jitterH) * 0.1
        let nx = n.x + dx, ny = n.y + dy
        if (nx < 1 || nx > WORLD.w - 1 || ny < 1 || ny > WORLD.h - 1) {
          return { ...n, heading: jitterH + Math.PI }
        }
        if (OBSTACLES.some((o) => circleIntersectsRect(nx, ny, R.radius, o))) {
          return { ...n, heading: jitterH + Math.PI/2 }
        }
        return { ...n, x: nx, y: ny, heading: jitterH }
      }))
    }, 300)
    return () => clearInterval(h)
  }, [])

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

  const nearbyNpcs = npcsState.filter((n) => dist({x:telemetry.x,y:telemetry.y}, n) < R.detection_r)
  const lidarMin = stateRef.current.lidar ? Math.min(...stateRef.current.lidar) : R.lidar_range

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="flex items-center gap-4 px-6 py-4 border-b border-white/10 bg-black/80 backdrop-blur">
        <Link to="/" className="flex items-center gap-2 text-xs text-white/50 hover:text-white"><ArrowLeft size={12}/> Home</Link>
        <div className="h-4 w-px bg-white/10" />
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-rose-400" />
          <span className="text-sm font-semibold">EDEN Simulator · 3D</span>
          <span className="text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-400/30">
            three.js · r3f · dynamics · lidar · ROS-bus
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
        <label className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest ml-2 cursor-pointer select-none">
          <input type="checkbox" checked={followCam} onChange={(e) => setFollowCam(e.target.checked)} className="accent-cyan-400"/>
          <Orbit size={11} className={followCam ? 'text-cyan-400' : 'text-white/30'}/>
          <span className={followCam ? 'text-cyan-300' : 'text-white/30'}>follow</span>
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
        {/* 3D World */}
        <div className="relative overflow-hidden">
          <Canvas camera={{ position: [WORLD.w/2, 18, WORLD.h + 6], fov: 45 }} style={{ background: 'radial-gradient(ellipse at center, #0a1220 0%, #000 85%)' }}>
            <Suspense fallback={null}>
              <ambientLight intensity={0.35} />
              <directionalLight position={[10, 16, 8]} intensity={0.8} />
              <pointLight position={[WORLD.w/2, 8, WORLD.h/2]} intensity={0.3} color="#22e0ff" />
              <fog attach="fog" args={['#050810', 20, 50]} />

              <Floor />
              <Walls />
              <Obstacles />
              <NpcRobots npcsState={npcsState} />
              <RobotMesh stateRef={stateRef} showOdom={showOdom} />
              <LidarRays stateRef={stateRef} show={showLidar} />

              <CameraRig stateRef={stateRef} follow={followCam} />
              {!followCam && <OrbitControls enableDamping dampingFactor={0.12} target={[WORLD.w/2, 0, WORLD.h/2]} />}
            </Suspense>
          </Canvas>

          {/* HUD */}
          <div className="absolute top-3 left-3 flex flex-col gap-2 text-[10px] font-mono uppercase tracking-widest pointer-events-none">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/>
                <span className="text-white/70">3D sim · diff-drive · TurtleBot3-scale</span>
              </div>
              <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-white/50">
                {WORLD.w}m × {WORLD.h}m · motor_tau {R.motor_tau}s
              </div>
              <div className="px-2 py-1 rounded bg-black/60 border border-white/10 backdrop-blur text-rose-300">
                LIDAR 270° · {R.lidar_rays} rays · min {lidarMin.toFixed(2)}m
              </div>
            </div>
            {nearbyNpcs.length > 0 && (
              <div className="px-2 py-1 rounded bg-cyan-500/10 border border-cyan-400/30 backdrop-blur text-cyan-200">
                detected {nearbyNpcs.length}: {nearbyNpcs.map((n) => n.name).join(', ')}
              </div>
            )}
          </div>
          <div className="absolute bottom-3 left-3 text-[10px] font-mono text-white/40 pointer-events-none">
            ↑↓ drive · ←→ turn · space stop · R reset · C toggle follow cam · drag to orbit (when follow off)
          </div>

          <Minimap stateRef={stateRef} />

          <AnimatePresence>
            {thinking && (
              <motion.div key="t" initial={{opacity:0,y:-6}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                className="absolute top-40 right-3 max-w-sm px-3 py-2 rounded-lg bg-cyan-500/10 border border-cyan-400/30 backdrop-blur flex items-center gap-2 pointer-events-none">
                <Brain size={12} className="text-cyan-300 animate-pulse"/>
                <div>
                  <div className="text-[9px] font-mono uppercase tracking-widest text-cyan-300 mb-0.5">cognitive layer · deliberating</div>
                  <code className="text-xs font-mono text-white/80 break-all">{thinking.action}</code>
                </div>
              </motion.div>
            )}
            {!thinking && log[0] && stateRef.current.cmdUntil > performance.now() && (
              <motion.div key={log[0].ts} initial={{opacity:0,y:-6}} animate={{opacity:1,y:0}} exit={{opacity:0}}
                className={`absolute top-40 right-3 max-w-sm px-3 py-2 rounded-lg backdrop-blur border pointer-events-none ${
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

        {/* Right rail */}
        <aside className="border-l border-white/10 bg-black/60 flex flex-col overflow-hidden">
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
            <Row l="heading"     v={((telemetry.heading*180/Math.PI)%360).toFixed(1)} u="°"/>
            <Row l="linear_vel"  v={telemetry.linVel.toFixed(3)}     u="m/s" hi={Math.abs(telemetry.linVel)>0.01}/>
            <Row l="angular_vel" v={telemetry.angVel.toFixed(3)}     u="rad/s" hi={Math.abs(telemetry.angVel)>0.01}/>
            <Row l="wheel_L"     v={telemetry.wheelL.toFixed(2)}     u="rad/s"/>
            <Row l="wheel_R"     v={telemetry.wheelR.toFixed(2)}     u="rad/s"/>
            <Row l="odom_drift"  v={telemetry.odomDrift.toFixed(3)}  u="m" hi={telemetry.odomDrift>0.1}/>
            <Row l="lidar_min"   v={lidarMin.toFixed(2)}             u="m" hi={lidarMin<0.8}/>
            <Row l="collisions"  v={stateRef.current.collisions}     u="" hi={stateRef.current.collisions>0}/>
          </div>

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
                    <th className="text-left">topic</th><th className="text-right">Hz</th><th className="text-right">msgs</th>
                  </tr>
                </thead>
                <tbody>
                  {topicsSnap.map((t) => (
                    <tr key={t.name} className="border-t border-white/5">
                      <td className="text-cyan-300 py-1 truncate max-w-0">{t.name}</td>
                      <td className="text-right text-white/70">{t.rate.toFixed(1)}</td>
                      <td className="text-right text-white/40">{t.msgCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

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
