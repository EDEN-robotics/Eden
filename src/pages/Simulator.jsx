import React, { useEffect, useRef, useState, useMemo, Suspense } from 'react'
import { Link } from 'react-router-dom'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Html, Grid, Environment, PerspectiveCamera } from '@react-three/drei'
import { motion, AnimatePresence } from 'framer-motion'
import * as THREE from 'three'
import { Cpu, ArrowLeft, Play, Pause, Wifi, Send, Brain, X, Check, Radar, Activity, Radio, Box as BoxIcon, Orbit } from 'lucide-react'
import { openSimBusReceiver, parseTaskIntent } from '../lib/simBridge'
import { classifyAction } from '../lib/cognitiveLayer'
import { publish, listTopics, Twist, Odometry, LaserScan, TFMessage } from '../lib/rosTopics'
import { createCostmap } from '../lib/costmap'
import { planPath, lineOfSight } from '../lib/pathPlanner'
import { clipForSafety, STOP_DIST, SLOW_DIST } from '../lib/safetyBumper'
import { autonomousTick } from '../lib/autonomousLoop'
import { TEAM, findTeammate } from '../lib/teamSeeds'
import { ITEMS, findItem, ITEM_DROP_HEIGHT } from '../lib/worldItems'
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
      {/* Main ground plane — matte polished concrete */}
      <mesh rotation={[-Math.PI/2, 0, 0]} position={[WORLD.w/2, 0, WORLD.h/2]} receiveShadow>
        <planeGeometry args={[WORLD.w, WORLD.h]} />
        <meshStandardMaterial color="#1a1f27" roughness={0.82} metalness={0.08} />
      </mesh>
      {/* Grid — subtle technical reference */}
      <Grid
        position={[WORLD.w/2, 0.002, WORLD.h/2]}
        args={[WORLD.w, WORLD.h]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#243142"
        sectionSize={5}
        sectionThickness={0.9}
        sectionColor="#3b5573"
        fadeDistance={30}
        fadeStrength={1.2}
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
  // Solid perimeter walls — painted lab concrete
  const t = 0.12
  const h = 1.1
  const mat = <meshStandardMaterial color="#2a2f38" roughness={0.9} metalness={0.05} />
  return (
    <group>
      <mesh position={[WORLD.w/2, h/2, 0]} castShadow receiveShadow><boxGeometry args={[WORLD.w, h, t]}/>{mat}</mesh>
      <mesh position={[WORLD.w/2, h/2, WORLD.h]} castShadow receiveShadow><boxGeometry args={[WORLD.w, h, t]}/>{mat}</mesh>
      <mesh position={[0, h/2, WORLD.h/2]} castShadow receiveShadow><boxGeometry args={[t, h, WORLD.h]}/>{mat}</mesh>
      <mesh position={[WORLD.w, h/2, WORLD.h/2]} castShadow receiveShadow><boxGeometry args={[t, h, WORLD.h]}/>{mat}</mesh>
    </group>
  )
}

// Material lookup by obstacle label — makes the lab look less like generic boxes
function obstacleMaterial(label) {
  const L = label.toLowerCase()
  if (L.includes('wall'))          return { color: '#3a4150', roughness: 0.9,  metalness: 0.05, emissive: '#000000', emissiveIntensity: 0 }
  if (L.includes('workbench'))     return { color: '#6b4a2b', roughness: 0.82, metalness: 0.08, emissive: '#000000', emissiveIntensity: 0 } // wood top
  if (L.includes('bench'))         return { color: '#5a4a38', roughness: 0.85, metalness: 0.05 }
  if (L.includes('trunk') || L.includes('cable'))
                                   return { color: '#1a1a1a', roughness: 0.45, metalness: 0.35 }
  if (L.includes('charging'))      return { color: '#0b4a5a', roughness: 0.35, metalness: 0.5,  emissive: '#22e0ff', emissiveIntensity: 0.6 }
  if (L.includes('parts bin') || L.includes('bin'))
                                   return { color: '#2b5f6d', roughness: 0.55, metalness: 0.2 }
  if (L.includes('server rack'))   return { color: '#0d0f14', roughness: 0.3,  metalness: 0.75, emissive: '#0a4040', emissiveIntensity: 0.15 }
  if (L.includes('crate') || L.includes('box'))
                                   return { color: '#8a6a3a', roughness: 0.92, metalness: 0.02 } // cardboard
  if (L.includes('cone'))          return { color: '#ea6b1f', roughness: 0.6,  metalness: 0.05, emissive: '#4a1a00', emissiveIntensity: 0.15 }
  if (L.includes('marker'))        return { color: '#d4a017', roughness: 0.4,  metalness: 0.1 }
  return { color: '#4b5563', roughness: 0.7, metalness: 0.15 }
}

function Obstacles() {
  return OBSTACLES.map((o, i) => {
    const mat = obstacleMaterial(o.label)
    const isCone = /cone/i.test(o.label)
    const isRack = /server rack/i.test(o.label)
    return (
      <group key={i} position={[o.x, 0, o.y]}>
        {isCone ? (
          <mesh position={[0, o.height/2, 0]} castShadow receiveShadow>
            <coneGeometry args={[Math.min(o.w, o.h) / 2, o.height, 16]} />
            <meshStandardMaterial {...mat} />
          </mesh>
        ) : (
          <mesh position={[0, o.height/2, 0]} castShadow receiveShadow>
            <boxGeometry args={[o.w, o.height, o.h]} />
            <meshStandardMaterial {...mat} />
          </mesh>
        )}
        {/* Server rack: add vent stripes */}
        {isRack && (
          <group position={[0, 0, 0]}>
            {[0.4, 0.8, 1.2, 1.6].map((y) => (
              <mesh key={y} position={[0, y, o.h/2 + 0.001]}>
                <boxGeometry args={[Math.min(0.5, o.w * 0.8), 0.05, 0.01]} />
                <meshBasicMaterial color="#0ea5e9" opacity={0.5} transparent />
              </mesh>
            ))}
          </group>
        )}
        <Html position={[0, o.height + 0.18, 0]} center distanceFactor={14} zIndexRange={[0, 0]}>
          <div className="px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-widest bg-black/60 text-white/70 rounded whitespace-nowrap pointer-events-none">
            {o.label}
          </div>
        </Html>
      </group>
    )
  })
}

// ───── Pickable items ─────
function ItemMesh({ item }) {
  const k = item.kind
  if (k === 'stick') {
    return (
      <mesh position={[0, 0.02, 0]} rotation={[0, 0, Math.PI/2]} castShadow>
        <cylinderGeometry args={[item.r, item.r, item.len, 10]} />
        <meshStandardMaterial color={item.color} roughness={0.6} metalness={0.3} />
      </mesh>
    )
  }
  if (k === 'rect') {
    return (
      <mesh position={[0, item.h/2 + ITEM_DROP_HEIGHT, 0]} castShadow>
        <boxGeometry args={[item.w, item.h, item.d]} />
        <meshStandardMaterial color={item.color} roughness={0.5} metalness={0.25} />
      </mesh>
    )
  }
  if (k === 'mug') {
    return (
      <group position={[0, ITEM_DROP_HEIGHT, 0]}>
        <mesh position={[0, item.h/2, 0]} castShadow>
          <cylinderGeometry args={[item.r, item.r * 0.9, item.h, 14]} />
          <meshStandardMaterial color={item.color} roughness={0.5} metalness={0.2} />
        </mesh>
        <mesh position={[item.r + 0.015, item.h * 0.5, 0]} rotation={[Math.PI/2, 0, 0]}>
          <torusGeometry args={[0.025, 0.008, 6, 12, Math.PI]} />
          <meshStandardMaterial color={item.color} roughness={0.5} />
        </mesh>
      </group>
    )
  }
  if (k === 'bottle') {
    return (
      <mesh position={[0, item.h/2 + ITEM_DROP_HEIGHT, 0]} castShadow>
        <cylinderGeometry args={[item.r, item.r * 0.8, item.h, 14]} />
        <meshStandardMaterial color={item.color} roughness={0.25} metalness={0.1} transparent opacity={0.75} />
      </mesh>
    )
  }
  if (k === 'laptop') {
    return (
      <group position={[0, ITEM_DROP_HEIGHT, 0]}>
        <mesh position={[0, item.h/2, 0]} castShadow>
          <boxGeometry args={[item.w, item.h, item.d]} />
          <meshStandardMaterial color={item.color} roughness={0.4} metalness={0.5} />
        </mesh>
        {/* lid tilted up */}
        <group position={[0, item.h, -item.d/2]}>
          <mesh position={[0, item.w/2 * 0.45, -item.d/2 * 0.1]} rotation={[-Math.PI/2.6, 0, 0]} castShadow>
            <boxGeometry args={[item.w, item.d, 0.005]} />
            <meshStandardMaterial color="#0b1220" emissive="#1e3a8a" emissiveIntensity={0.5} />
          </mesh>
        </group>
      </group>
    )
  }
  // default: small box
  return (
    <mesh position={[0, 0.04, 0]} castShadow>
      <boxGeometry args={[0.1, 0.08, 0.1]} />
      <meshStandardMaterial color={item.color} />
    </mesh>
  )
}

function WorldItems({ itemsState, inventory }) {
  const held = new Set(inventory.map((h) => h.id))
  return itemsState.filter((it) => !held.has(it.id)).map((it) => (
    <group key={it.id} position={[it.x, 0, it.y]}>
      <ItemMesh item={it} />
      <Html position={[0, 0.28, 0]} center distanceFactor={12} zIndexRange={[0, 0]}>
        <div className="px-1 py-0.5 text-[7px] font-mono uppercase tracking-widest bg-black/70 text-amber-200 rounded whitespace-nowrap pointer-events-none border border-amber-400/30">
          {it.label}
        </div>
      </Html>
    </group>
  ))
}

function HeldItems({ stateRef, inventory }) {
  const groupRef = useRef()
  useFrame(() => {
    if (!groupRef.current) return
    const s = stateRef.current
    groupRef.current.position.set(s.x, 0.6, s.y)
    groupRef.current.rotation.y = -s.heading
  })
  if (inventory.length === 0) return null
  return (
    <group ref={groupRef}>
      {inventory.map((it, i) => (
        <group key={it.id} position={[0, i * 0.1, 0]}>
          <ItemMesh item={it} />
        </group>
      ))}
    </group>
  )
}

// ───── Team users rendered as avatars at their seats ─────
function SimUsers({ usersState }) {
  return usersState.map((u) => (
    <group key={u.id} position={[u.x, 0, u.y]}>
      {/* shadow disc */}
      <mesh position={[0, 0.003, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <circleGeometry args={[0.3, 20]} />
        <meshBasicMaterial color="#000" transparent opacity={0.35} />
      </mesh>
      {/* body — tall cylinder, colored */}
      <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.16, 0.2, 0.75, 16]} />
        <meshStandardMaterial color={u.color} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* head */}
      <mesh position={[0, 0.95, 0]} castShadow>
        <sphereGeometry args={[0.11, 20, 20]} />
        <meshStandardMaterial color="#d4c7b2" roughness={0.85} />
      </mesh>
      {/* name tag */}
      <Html position={[0, 1.3, 0]} center distanceFactor={11} zIndexRange={[0, 0]}>
        <div className="px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-widest rounded whitespace-nowrap pointer-events-none border"
             style={{ background: 'rgba(0,0,0,0.55)', color: u.color, borderColor: u.color + '50' }}>
          {u.name} · {u.role}
        </div>
      </Html>
    </group>
  ))
}

function LLMLatencySparkline({ samples }) {
  if (!samples || samples.length === 0) {
    return <div className="text-[10px] font-mono text-white/30">no cognitive calls yet…</div>
  }
  const max = Math.max(...samples, 300)
  const avg = Math.round(samples.reduce((a, b) => a + b, 0) / samples.length)
  const p95 = (() => {
    const s = [...samples].sort((a, b) => a - b)
    return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))]
  })()
  return (
    <div>
      <div className="flex items-end gap-[2px] h-8 mb-1">
        {samples.map((ms, i) => {
          const h = Math.max(2, (ms / max) * 32)
          const color = ms > 2000 ? 'bg-rose-400' : ms > 1000 ? 'bg-amber-400' : 'bg-cyan-400'
          return <div key={i} style={{ height: `${h}px` }} className={`w-1 rounded-sm ${color}/70`} />
        })}
      </div>
      <div className="flex justify-between text-[9px] font-mono text-white/40 uppercase tracking-widest">
        <span>avg {avg}ms</span><span>p95 {p95}ms</span><span>n={samples.length}</span>
      </div>
    </div>
  )
}

function StatCell({ label, value, accent = 'cyan' }) {
  const colorMap = { cyan: 'text-cyan-300', rose: 'text-rose-300', amber: 'text-amber-300', emerald: 'text-emerald-300' }
  return (
    <div className="px-2 py-1 rounded border border-white/10 bg-black/40">
      <div className={`text-sm font-bold ${colorMap[accent] || 'text-white'}`}>{value}</div>
      <div className="text-[9px] font-mono text-white/40 uppercase tracking-widest">{label}</div>
    </div>
  )
}

function PathPreview({ waypoints }) {
  if (!waypoints || waypoints.length < 2) return null
  const points = waypoints.map((p) => new THREE.Vector3(p.x, 0.05, p.y))
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points)
    return g
  }, [waypoints])
  return (
    <>
      <line>
        <primitive object={geom} attach="geometry" />
        <lineBasicMaterial color="#22d3ee" linewidth={2} transparent opacity={0.85} />
      </line>
      {points.map((p, i) => (
        <mesh key={i} position={[p.x, 0.12, p.z]}>
          <sphereGeometry args={[0.07, 8, 8]} />
          <meshBasicMaterial color={i === 0 ? '#34d399' : i === points.length - 1 ? '#f472b6' : '#22d3ee'} transparent opacity={0.85} />
        </mesh>
      ))}
    </>
  )
}

function NpcRobots({ npcsState }) {
  return npcsState.map((n) => (
    <group key={n.name} position={[n.x, 0, n.y]} rotation={[0, -n.heading, 0]}>
      {/* fake contact shadow disc */}
      <mesh position={[0, 0.003, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <circleGeometry args={[R.radius * 1.4, 24]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.35} />
      </mesh>
      {/* chassis (matte dark) */}
      <mesh position={[0, 0.14, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[R.radius * 0.95, R.radius * 1.0, 0.22, 20]} />
        <meshStandardMaterial color="#1a1f28" roughness={0.55} metalness={0.45} />
      </mesh>
      {/* upper dome */}
      <mesh position={[0, 0.34, 0]} castShadow>
        <cylinderGeometry args={[R.radius * 0.7, R.radius * 0.9, 0.2, 20]} />
        <meshStandardMaterial color={n.color} emissive={n.color} emissiveIntensity={0.35} roughness={0.4} metalness={0.3} />
      </mesh>
      {/* sensor dot */}
      <mesh position={[0, 0.48, 0]}>
        <sphereGeometry args={[0.05, 12, 12]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.8} />
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
        {/* contact shadow */}
        <mesh position={[0, 0.003, 0]} rotation={[-Math.PI/2, 0, 0]}>
          <circleGeometry args={[R.radius * 1.45, 28]} />
          <meshBasicMaterial color="#000000" transparent opacity={0.42} />
        </mesh>
        {/* chassis cylinder */}
        <mesh position={[0, 0.16, 0]} castShadow receiveShadow>
          <cylinderGeometry args={[R.radius, R.radius * 1.05, 0.3, 32]} />
          <meshStandardMaterial color="#12161d" roughness={0.35} metalness={0.65} />
        </mesh>
        {/* cyan ring */}
        <mesh position={[0, 0.32, 0]} rotation={[Math.PI/2, 0, 0]}>
          <torusGeometry args={[R.radius * 0.85, 0.02, 12, 48]} />
          <meshStandardMaterial color="#22e0ff" emissive="#22e0ff" emissiveIntensity={1.2} />
        </mesh>
        {/* sensor post (lidar) */}
        <mesh position={[0, 0.45, 0]} castShadow>
          <cylinderGeometry args={[0.05, 0.05, 0.08, 16]} />
          <meshStandardMaterial color="#0a0f14" roughness={0.2} metalness={0.9} />
        </mesh>
        <mesh position={[0, 0.5, 0]}>
          <sphereGeometry args={[0.06, 20, 20]} />
          <meshStandardMaterial color="#22e0ff" emissive="#22e0ff" emissiveIntensity={1.5} />
        </mesh>
        {/* heading arrow */}
        <mesh position={[R.radius + 0.05, 0.16, 0]} rotation={[0, 0, -Math.PI/2]}>
          <coneGeometry args={[0.1, 0.2, 12]} />
          <meshStandardMaterial color="#22e0ff" emissive="#22e0ff" emissiveIntensity={0.9} />
        </mesh>
        {/* wheels */}
        {[-1, 1].map((s) => (
          <mesh key={s} position={[0, 0.08, s * R.wheel_base/2]} rotation={[0, 0, Math.PI/2]} castShadow>
            <cylinderGeometry args={[R.wheel_r, R.wheel_r, 0.05, 16]} />
            <meshStandardMaterial color="#0a0d13" roughness={0.95} metalness={0.1} />
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
function Minimap({ stateRef, costmapRef, pathPreview, itemsState, usersState }) {
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

      // costmap overlay (Bayesian occupancy + inflation)
      const cm = costmapRef?.current
      if (cm) {
        const snap = cm.snapshot()
        const cs = snap.res * pm
        for (let r = 0; r < snap.rows; r++) {
          for (let k = 0; k < snap.cols; k++) {
            const v = snap.data[r * snap.cols + k]
            if (v === 1) continue // unknown → skip
            if (v === 2) ctx.fillStyle = 'rgba(244,114,182,0.75)'      // lethal
            else if (v === 3) ctx.fillStyle = 'rgba(34,211,238,0.22)'   // inflation band
            else ctx.fillStyle = 'rgba(16,185,129,0.08)'                // free
            ctx.fillRect(k * cs, r * cs, cs + 0.5, cs + 0.5)
          }
        }
      }

      // ground-truth obstacles (dim outline)
      for (const o of OBSTACLES) {
        ctx.strokeStyle = 'rgba(200,220,255,0.3)'
        ctx.lineWidth = 0.5
        ctx.strokeRect((o.x - o.w/2) * pm, (o.y - o.h/2) * pm, o.w * pm, o.h * pm)
      }

      // path preview
      if (pathPreview && pathPreview.length > 1) {
        ctx.strokeStyle = 'rgba(34,211,238,0.9)'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(pathPreview[0].x * pm, pathPreview[0].y * pm)
        for (let i = 1; i < pathPreview.length; i++) {
          ctx.lineTo(pathPreview[i].x * pm, pathPreview[i].y * pm)
        }
        ctx.stroke()
        const last = pathPreview[pathPreview.length - 1]
        ctx.fillStyle = '#f472b6'
        ctx.beginPath(); ctx.arc(last.x * pm, last.y * pm, 3, 0, Math.PI*2); ctx.fill()
      }

      // items (small amber dots)
      if (itemsState) {
        for (const it of itemsState) {
          ctx.fillStyle = '#fbbf24'
          ctx.beginPath(); ctx.arc(it.x * pm, it.y * pm, 2, 0, Math.PI*2); ctx.fill()
        }
      }
      // team users (colored circles)
      if (usersState) {
        for (const u of usersState) {
          ctx.fillStyle = u.color
          ctx.globalAlpha = 0.9
          ctx.beginPath(); ctx.arc(u.x * pm, u.y * pm, 3.5, 0, Math.PI*2); ctx.fill()
          ctx.globalAlpha = 1
        }
      }
      // npcs
      for (const n of NPCS) {
        ctx.fillStyle = n.color
        ctx.globalAlpha = 0.55
        ctx.beginPath(); ctx.arc(n.x * pm, n.y * pm, 2.5, 0, Math.PI*2); ctx.fill()
        ctx.globalAlpha = 1
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
  }, [pathPreview, itemsState, usersState])
  const W = 280, H = W * (WORLD.h / WORLD.w)
  return (
    <div className="absolute top-3 right-3 pointer-events-none">
      <div className="mb-1 text-[9px] font-mono uppercase tracking-widest text-white/50 flex gap-3">
        <span>MINIMAP · /map</span>
        <span className="text-rose-300">▮ lethal</span>
        <span className="text-cyan-300">▮ inflation</span>
        <span className="text-emerald-300">▮ free</span>
      </div>
      <canvas ref={canvasRef} width={W} height={H} className="rounded border border-white/15 shadow-2xl" />
    </div>
  )
}

// Landmark lookup for named goals ("drive to workbench A" → a coordinate)
const LANDMARKS = OBSTACLES.filter((o) => /^(workbench|dock|bin|rack|crate|cone|bench|marker|charging)/i.test(o.label))
  .map((o) => ({ label: o.label.toLowerCase(), x: o.x, y: o.y, w: o.w, h: o.h }))

function findLandmark(text) {
  if (!text) return null
  const t = text.toLowerCase()
  // Exact/partial phrase matches first
  for (const lm of LANDMARKS) {
    if (t.includes(lm.label)) return lm
  }
  // Token-level fuzzy: "workbench a" matches "workbench a"
  const words = t.split(/\s+/)
  for (const lm of LANDMARKS) {
    const lw = lm.label.split(/\s+/)
    if (lw.every((w) => words.includes(w))) return lm
  }
  return null
}

function estimateExploredFraction(cm) {
  // Fraction of cells with non-zero log-odds magnitude (i.e. observed)
  const lo = cm.logOdds
  let seen = 0
  for (let i = 0; i < lo.length; i++) if (Math.abs(lo[i]) > 0.15) seen++
  return seen / lo.length
}

// ───── Main ─────
export default function Simulator() {
  const costmapRef = useRef(createCostmap({ worldW: WORLD.w, worldH: WORLD.h, res: 0.25, inflateRadius: 0.5 }))
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
    lastCostmapUpdate: 0, lastCostmapInflate: 0,
    collisions: 0,
    bumperStops: 0,
    distanceTraveled: 0,
    battery: 100,          // percent; drains under motion, recharges at dock
    path: [],              // current waypoint list (world coords) from planner
    pathGoal: null,        // { label, x, y }
    pathIdx: 0,
    lastSafetyReason: null,
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
  const [goals, setGoals] = useState([])
  const [batteryPct, setBatteryPct] = useState(100)
  const [thoughts, setThoughts] = useState([])        // autonomous-loop thought stream
  const [llmLatency, setLlmLatency] = useState([])    // last ~20 ms samples
  const [pathPreview, setPathPreview] = useState([])  // for 3D overlay
  const [itemsState, setItemsState] = useState(ITEMS.map((it) => ({ ...it })))
  const [inventory, setInventory] = useState([])      // items currently held by EDEN
  const [usersState, setUsersState] = useState(TEAM.map((p) => ({ id: p.id, name: p.name, role: p.role, color: p.color, x: p.seat.x, y: p.seat.y })))
  const [task, setTask] = useState(null)              // { kind, phase, itemId, recipientId, startedAt }
  const taskRef = useRef(null)
  useEffect(() => { taskRef.current = task }, [task])
  const itemsRef = useRef(itemsState)
  useEffect(() => { itemsRef.current = itemsState }, [itemsState])
  const usersRef = useRef(usersState)
  useEffect(() => { usersRef.current = usersState }, [usersState])
  const inventoryRef = useRef(inventory)
  useEffect(() => { inventoryRef.current = inventory }, [inventory])

  // Latest-value refs — fix stale-closure in bus receiver / physics loop
  const latestRef = useRef({ log, useLLM, npcsState, goals, batteryPct })
  useEffect(() => {
    latestRef.current = { log, useLLM, npcsState, goals, batteryPct }
  }, [log, useLLM, npcsState, goals, batteryPct])
  const applyActionRef = useRef(null)

  useEffect(() => {
    const close = openSimBusReceiver((payload) => {
      applyActionRef.current?.(payload.action, payload.meta?.source || 'eden', payload.meta)
    })
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
        s.trail = []; s.collisions = 0; s.bumperStops = 0; s.distanceTraveled = 0
        s.path = []; s.pathGoal = null; s.pathIdx = 0; s.battery = 100
        setPathPreview([]); setBatteryPct(100)
      }
      if (e.code === 'KeyC') setFollowCam((f) => !f)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  async function applyAction(rawAction, source, chatCtx = null) {
    const ts = Date.now()
    const s = stateRef.current
    const { log: latestLog, useLLM: latestUseLLM, npcsState: latestNpcs, batteryPct: latestBat } = latestRef.current

    // ─── Task intent detection (fetch/deliver/drop/approach) ───
    const intent = parseTaskIntent(rawAction)
    if (intent && intent.kind === 'approach') {
      // Resolve target — "me"/"__speaker__" means chat speaker
      let target = null
      const recipText = intent.recipient || ''
      if (/\b(me|myself)\b/.test(recipText) && chatCtx?.speaker) {
        target = findTeammate(chatCtx.speaker)
      } else {
        target = findTeammate(recipText)
      }
      // Fallback: scan the whole action text
      if (!target) target = findTeammate(rawAction)
      // Final fallback: the speaker
      if (!target && chatCtx?.speaker) target = findTeammate(chatCtx.speaker)

      if (!target) {
        const entry = { ts, source, action: rawAction, decision: 'refuse', reason: `no one matches "${recipText}"`, model: 'task-parser', ms: 0 }
        setLog((p) => [entry, ...p].slice(0, 50))
        return
      }
      const targetUser = usersRef.current.find((u) => u.id === target.id)
      if (!targetUser) {
        const entry = { ts, source, action: rawAction, decision: 'refuse', reason: `${target.name} not in sim`, model: 'task-parser', ms: 0 }
        setLog((p) => [entry, ...p].slice(0, 50))
        return
      }

      // Plan path to the target. Stop ~0.6m short so the bumper can
      // enforce the real minimum distance dramatically.
      const approachX = targetUser.x - 0.6
      const approachY = targetUser.y - 0.6
      const entry = {
        ts, source, action: rawAction,
        decision: 'execute',
        reason: intent.aggressive ? `rolling up on ${targetUser.name} — bumper will handle the rest` : `approaching ${targetUser.name}`,
        model: 'task-sm',
        ms: 0,
        goal: `${intent.aggressive ? 'charge' : 'approach'}:${targetUser.name}`,
      }
      setLog((p) => [entry, ...p].slice(0, 50))
      publish('/task_status', { _type: 'eden/TaskStatus', kind: intent.aggressive ? 'charge' : 'approach', target: targetUser.id })
      planToPoint({ x: approachX, y: approachY, label: `${intent.aggressive ? 'charge' : 'approach'}: ${targetUser.name}` })
      return
    }

    if (intent) {
      const item = findItem(intent.item)
      if (!item) {
        const entry = { ts, source, action: rawAction, decision: 'refuse', reason: `I don't see any "${intent.item}" in the lab`, model: 'task-parser', ms: 0 }
        setLog((p) => [entry, ...p].slice(0, 50))
        return
      }
      if (intent.kind === 'drop') {
        // drop in front of the robot
        const held = inventoryRef.current.find((h) => h.id === item.id)
        if (!held) {
          const entry = { ts, source, action: rawAction, decision: 'refuse', reason: `I'm not holding the ${item.label}`, model: 'task-parser', ms: 0 }
          setLog((p) => [entry, ...p].slice(0, 50))
          return
        }
        const dropX = s.x + Math.cos(s.heading) * 0.5
        const dropY = s.y + Math.sin(s.heading) * 0.5
        setInventory((inv) => inv.filter((h) => h.id !== item.id))
        setItemsState((list) => list.map((it) => it.id === item.id ? { ...it, x: dropX, y: dropY } : it))
        const entry = { ts, source, action: rawAction, decision: 'execute', reason: `dropped ${item.label}`, model: 'task-sm', ms: 0, goal: `drop:${item.label}` }
        setLog((p) => [entry, ...p].slice(0, 50))
        publish('/task_status', { _type: 'eden/TaskStatus', kind: 'drop', item: item.id, at: { x: dropX, y: dropY } })
        return
      }
      // Fetch: resolve recipient. Priority:
      //  1. Explicit "__speaker__" (from "get me the X") → chat speaker
      //  2. Explicit name captured by regex
      //  3. Full-text scan — LLM might phrase it compound ("get X and go to Y")
      //  4. Nothing → no delivery, just fetch
      let recipient = null
      if (intent.recipient === '__speaker__') {
        if (chatCtx?.speaker) recipient = findTeammate(chatCtx.speaker)
      } else if (intent.recipient) {
        recipient = findTeammate(intent.recipient)
      }
      if (!recipient) {
        // Fallback: scan the full action text (minus the item word) for a team name
        const scanText = rawAction.toLowerCase().replace(new RegExp(`\\b${item.label}\\b`, 'g'), ' ')
        recipient = findTeammate(scanText)
      }
      const recipientUser = recipient ? usersRef.current.find((u) => u.id === recipient.id) : null

      // Run the cognitive gate on the compound intent — EDEN can still refuse
      setThinking({ action: rawAction, source })
      const obstacles = OBSTACLES.map((o) => ({ ...o, dist: Math.hypot(o.x - s.x, o.y - s.y), reachable: lineOfSight(costmapRef.current, { x: s.x, y: s.y }, { x: o.x, y: o.y }) })).filter((o) => o.dist < 10).sort((a, b) => a.dist - b.dist)
      const history = latestLog.slice(0, 4).map((l) => ({ source: l.source, action: l.action, decision: l.decision }))
      const goal = { label: `${item.label}${recipientUser ? ` → ${recipientUser.name}` : ''}`, x: item.x, y: item.y, dist: Math.hypot(item.x - s.x, item.y - s.y), blocked: !lineOfSight(costmapRef.current, { x: s.x, y: s.y }, { x: item.x, y: item.y }) }
      const res = await classifyAction({
        action: rawAction,
        robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
        obstacles, npcs: latestNpcs, history,
        chatCtx, goal, battery: latestBat,
      })
      setThinking(null)
      setLlmLatency((p) => [...p, res.ms].slice(-20))
      const entry = { ts, source, action: rawAction, decision: res.decision, reason: res.reason, linear: res.linear, angular: res.angular, duration: res.duration, model: res.model, ms: res.ms, goal: goal.label }
      setLog((p) => [entry, ...p].slice(0, 50))
      if (res.decision === 'refuse') return

      // Start the task state machine
      startFetchTask({ item, recipientUser })
      return
    }

    if (!latestUseLLM || source === 'manual-direct') {
      const { parseAction } = await import('../lib/simBridge')
      const parsed = parseAction(rawAction)
      const entry = { ts, source, action: rawAction, decision: parsed ? 'execute' : 'refuse', reason: parsed ? 'direct (cognitive gate off)' : 'unparseable', linear: parsed?.linear, angular: parsed?.angular, duration: parsed?.duration, model: 'regex', ms: 0 }
      setLog((p) => [entry, ...p].slice(0, 50))
      if (!parsed) return
      commitCmd(parsed.linear, parsed.angular, parsed.duration, parsed.stop)
      maybePlanToLandmark(rawAction)
      return
    }

    setThinking({ action: rawAction, source })

    // Build rich context for the LLM
    const obstacles = OBSTACLES
      .map((o) => {
        const d = Math.hypot(o.x - s.x, o.y - s.y)
        return { ...o, dist: d, reachable: lineOfSight(costmapRef.current, { x: s.x, y: s.y }, { x: o.x, y: o.y }) }
      })
      .filter((o) => o.dist < 8)
      .sort((a, b) => a.dist - b.dist)
    const npcs = latestNpcs.filter((n) => Math.hypot(n.x - s.x, n.y - s.y) < R.detection_r * 1.5)
    const history = latestLog.slice(0, 4).map((l) => ({ source: l.source, action: l.action, decision: l.decision }))

    // Named-landmark goal extraction
    const lm = findLandmark(rawAction)
    let goal = null
    if (lm) {
      const d = Math.hypot(lm.x - s.x, lm.y - s.y)
      const clear = lineOfSight(costmapRef.current, { x: s.x, y: s.y }, { x: lm.x, y: lm.y })
      goal = { label: lm.label, x: lm.x, y: lm.y, dist: d, blocked: !clear }
    }

    const res = await classifyAction({
      action: rawAction,
      robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
      obstacles, npcs, history,
      chatCtx, goal, battery: latestBat,
    })
    setThinking(null)
    setLlmLatency((p) => [...p, res.ms].slice(-20))
    const entry = { ts, source, action: rawAction, decision: res.decision, reason: res.reason, linear: res.linear, angular: res.angular, duration: res.duration, model: res.model, ms: res.ms, goal: goal?.label || null }
    setLog((p) => [entry, ...p].slice(0, 50))
    if (res.decision === 'refuse' || res.linear === null) return

    // If there's a landmark goal, plan a path and follow waypoints instead of a blind cmd
    if (goal) {
      maybePlanToLandmark(rawAction, goal)
    }
    commitCmd(res.linear, res.angular, res.duration, false)
  }

  // ─── Task state machine ───
  // phases: 'goto_item' → (on arrival) pickup → (if recipient) 'goto_user' → drop → 'done'
  function startFetchTask({ item, recipientUser }) {
    const t = { kind: 'fetch', phase: 'goto_item', itemId: item.id, itemLabel: item.label, recipientId: recipientUser?.id || null, recipientName: recipientUser?.name || null, startedAt: Date.now() }
    setTask(t)
    planToPoint({ x: item.x, y: item.y, label: `pickup: ${item.label}` })
    publish('/task_status', { _type: 'eden/TaskStatus', kind: 'fetch_start', item: item.id, recipient: recipientUser?.id || null })
  }

  function advanceTask() {
    const t = taskRef.current
    if (!t) return
    const s = stateRef.current
    if (t.phase === 'goto_item') {
      const item = itemsRef.current.find((it) => it.id === t.itemId)
      if (!item) return
      const d = Math.hypot(s.x - item.x, s.y - item.y)
      if (d > 1.0) {
        // Arrived-at-waypoints but still far from item — replan
        planToPoint({ x: item.x, y: item.y, label: `pickup: ${item.label}` })
        return
      }
      // Pick up
      setInventory((inv) => [...inv, item])
      setItemsState((list) => list.filter((it) => it.id !== t.itemId))
      publish('/task_status', { _type: 'eden/TaskStatus', kind: 'pickup', item: item.id })
      if (t.recipientId) {
        const u = usersRef.current.find((uu) => uu.id === t.recipientId)
        if (u) {
          setTask({ ...t, phase: 'goto_user' })
          // Stand slightly off the user's seat so the robot isn't on top of them
          const offX = u.x - 0.7, offY = u.y - 0.7
          planToPoint({ x: offX, y: offY, label: `deliver: ${u.name}` })
          return
        }
      }
      // No recipient — task done
      setTask({ ...t, phase: 'done' })
      setTimeout(() => setTask((cur) => (cur && cur.startedAt === t.startedAt ? null : cur)), 1200)
    } else if (t.phase === 'goto_user') {
      const u = usersRef.current.find((uu) => uu.id === t.recipientId)
      const d = u ? Math.hypot(s.x - u.x, s.y - u.y) : 99
      if (d > 1.4) {
        if (u) planToPoint({ x: u.x - 0.7, y: u.y - 0.7, label: `deliver: ${u.name}` })
        return
      }
      // Deliver
      const held = inventoryRef.current.find((h) => h.id === t.itemId)
      if (held && u) {
        const dropX = u.x + 0.3, dropY = u.y + 0.3
        setInventory((inv) => inv.filter((h) => h.id !== t.itemId))
        setItemsState((list) => [...list, { ...held, x: dropX, y: dropY }])
        publish('/task_status', { _type: 'eden/TaskStatus', kind: 'delivered', item: t.itemId, recipient: t.recipientId })
      }
      setTask({ ...t, phase: 'done' })
      setTimeout(() => setTask((cur) => (cur && cur.startedAt === t.startedAt ? null : cur)), 1500)
    }
  }

  function planToPoint({ x, y, label }) {
    const s = stateRef.current
    costmapRef.current.rebuildInflation()
    const plan = planPath(costmapRef.current, { x: s.x, y: s.y }, { x, y })
    if (plan.ok && plan.waypoints.length > 1) {
      s.path = plan.waypoints
      s.pathGoal = { label, x, y }
      s.pathIdx = 1
      setPathPreview(plan.waypoints)
      publish('/plan', { _type: 'nav_msgs/Path', poses: plan.waypoints, header: { stamp: performance.now(), frame_id: 'map' } })
    }
  }

  function maybePlanToLandmark(rawAction, goal = null) {
    const s = stateRef.current
    const lm = goal || (() => {
      const L = findLandmark(rawAction)
      return L ? { label: L.label, x: L.x, y: L.y } : null
    })()
    if (!lm) return
    // Re-inflate costmap immediately so recent LIDAR has effect
    costmapRef.current.rebuildInflation()
    const plan = planPath(costmapRef.current, { x: s.x, y: s.y }, { x: lm.x, y: lm.y })
    if (plan.ok && plan.waypoints.length > 1) {
      s.path = plan.waypoints
      s.pathGoal = lm
      s.pathIdx = 1 // skip start cell
      setPathPreview(plan.waypoints)
      publish('/plan', { _type: 'nav_msgs/Path', poses: plan.waypoints, header: { stamp: performance.now(), frame_id: 'map' } })
    }
  }

  // Expose applyAction to the (stale-deps) receiver via ref
  useEffect(() => { applyActionRef.current = applyAction })

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
        // ─── Local planner: if we have a path, generate (linear, angular) toward the next waypoint
        let pathLin = null, pathAng = null
        if (s.path && s.path.length > 0 && s.pathIdx < s.path.length) {
          const wp = s.path[s.pathIdx]
          const dx = wp.x - s.x, dy = wp.y - s.y
          const dToWp = Math.hypot(dx, dy)
          if (dToWp < 0.35) {
            s.pathIdx += 1
            if (s.pathIdx >= s.path.length) {
              s.path = []; s.pathGoal = null; s.pathIdx = 0
              setPathPreview([])
              // If a task was awaiting arrival, advance the state machine
              if (taskRef.current) {
                queueMicrotask(() => advanceTask())
              }
            }
          } else {
            const targetHeading = Math.atan2(dy, dx)
            let dh = targetHeading - s.heading
            while (dh > Math.PI) dh -= 2 * Math.PI
            while (dh < -Math.PI) dh += 2 * Math.PI
            // Turn-in-place when heading is off; otherwise drive + correct
            if (Math.abs(dh) > 0.6) {
              pathLin = 0
              pathAng = clamp(Math.sign(dh) * Math.min(1.0, Math.abs(dh) * 2), -R.max_ang, R.max_ang)
            } else {
              pathLin = clamp(0.3 * (1 - Math.abs(dh) / 0.6), 0.05, R.max_lin)
              pathAng = clamp(dh * 1.4, -R.max_ang, R.max_ang)
            }
            s.cmdUntil = now + 600 // keep alive while following
          }
        }

        // Priority: path-follower > cmdUntil-gated cmd > decay
        const gated = now > s.cmdUntil
        let targetLin = pathLin != null ? pathLin : (gated ? 0 : s.cmdLin)
        let targetAng = pathAng != null ? pathAng : (gated ? 0 : s.cmdAng)

        // Safety bumper — clip the velocity if something's in the front arc
        const bumper = clipForSafety({
          cmdLin: targetLin,
          cmdAng: targetAng,
          lidar: s.lidar,
          fov: R.lidar_fov,
          reversing: targetLin < 0,
        })
        if (bumper.reason) s.lastSafetyReason = bumper.reason
        if (bumper.cmdLin !== targetLin && Math.abs(targetLin) > 0.05 && bumper.cmdLin === 0) s.bumperStops += 1
        targetLin = bumper.cmdLin
        targetAng = bumper.cmdAng

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
        if (!hit) {
          s.distanceTraveled += Math.hypot(nextX - s.x, nextY - s.y)
          s.x = nextX; s.y = nextY
        } else {
          s.linVel = 0; s.collisions += 1
        }
        s.heading = nextHeading

        // Battery model: base 0.2%/min + 0.8%/min × |linVel|/max_lin + 0.3%/min × |angVel|/max_ang
        const drainPerSec = (0.2 + 0.8 * Math.abs(s.linVel) / R.max_lin + 0.3 * Math.abs(s.angVel) / R.max_ang) / 60
        s.battery = Math.max(0, s.battery - drainPerSec * dt)
        // Recharge at the charging dock
        const dock = OBSTACLES.find((o) => o.label === 'charging dock')
        if (dock && Math.hypot(s.x - dock.x, s.y - dock.y) < 1.0) {
          s.battery = Math.min(100, s.battery + (8 / 60) * dt) // 8%/min
        }

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

        // Costmap integration — run at ~10Hz, re-inflate at ~2Hz
        if (now - s.lastCostmapUpdate > 100) {
          costmapRef.current.integrateLidar({
            x: s.x, y: s.y, heading: s.heading,
            ranges: s.lidar,
            angle_min: -R.lidar_fov / 2,
            angle_max: R.lidar_fov / 2,
            range_max: R.lidar_range,
          })
          s.lastCostmapUpdate = now
        }
        if (now - s.lastCostmapInflate > 500) {
          costmapRef.current.rebuildInflation()
          s.lastCostmapInflate = now
          publish('/map', { _type: 'nav_msgs/OccupancyGrid', info: { resolution: costmapRef.current.res, width: costmapRef.current.cols, height: costmapRef.current.rows } })
        }

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
      // Battery sync at ~2Hz to avoid re-rendering 60 times a second
      if (!s.lastBatSync || now - s.lastBatSync > 500) {
        setBatteryPct(s.battery)
        s.lastBatSync = now
      }
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

  // Autonomous loop — 20s cadence. Hard-coded battery rule overrides LLM if
  // we're critically low (go to the dock), otherwise asks the LLM what to do.
  useEffect(() => {
    let active = true
    async function tick() {
      if (!active) return
      const s = stateRef.current
      const { goals: latestGoals, batteryPct: latestBat, log: latestLog, npcsState: latestNpcs } = latestRef.current

      // Hard rule: low battery → drive to dock (no LLM needed, always safe)
      if (latestBat < 20 && !(s.pathGoal && s.pathGoal.label === 'charging dock')) {
        setThoughts((p) => [{ ts: Date.now(), thought: `battery at ${latestBat.toFixed(0)}% — heading to charging dock`, action: 'head to charging dock', auto: true }, ...p].slice(0, 8))
        applyActionRef.current?.('head to charging dock', 'auto-battery')
        return
      }

      // Skip tick if we're actively moving or following a plan
      if (s.pathGoal || Math.abs(s.linVel) > 0.05 || stateRef.current.cmdUntil > performance.now() + 200) return

      const unexplored = 1 - estimateExploredFraction(costmapRef.current)
      const recentActions = latestLog.slice(0, 4).map((l) => `${l.action} → ${l.decision}`)
      const res = await autonomousTick({
        robot: { x: s.x, y: s.y, heading: s.heading, linVel: s.linVel, angVel: s.angVel },
        obstacles: OBSTACLES, npcs: latestNpcs, goals: latestGoals,
        battery: latestBat, recentActions, unexploredFraction: unexplored, recentMemories: [],
      })
      if (!res || !active) return

      setThoughts((p) => [{ ts: Date.now(), thought: res.thought, action: res.action, reason: res.reason, auto: true }, ...p].slice(0, 8))
      if (res.goal_add) setGoals((g) => [...g, res.goal_add].slice(-8))
      if (res.goal_done) setGoals((g) => g.filter((x) => x !== res.goal_done))
      if (res.action && res.action !== 'none') {
        applyActionRef.current?.(res.action, 'auto-loop')
      }
      // TODO: post_to_chat → would reach the chat window via a new bus event; out of scope for v1
    }
    const h = setInterval(tick, 20000)
    // Kick it off quickly once so the loop is visible
    const kick = setTimeout(tick, 4000)
    return () => { active = false; clearInterval(h); clearTimeout(kick) }
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
          <Canvas
            shadows
            dpr={[1, 2]}
            camera={{ position: [WORLD.w/2, 18, WORLD.h + 6], fov: 45 }}
            gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.05 }}
            style={{ background: 'radial-gradient(ellipse at center, #121822 0%, #050709 85%)' }}
          >
            <Suspense fallback={null}>
              <ambientLight intensity={0.22} color="#b8c6d8" />
              <hemisphereLight args={[0xbcd4ea, 0x1a1410, 0.38]} />
              <directionalLight
                position={[WORLD.w * 0.35, 22, WORLD.h * 0.25]}
                intensity={1.25}
                color="#ffe8c7"
                castShadow
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
                shadow-camera-left={-WORLD.w * 0.7}
                shadow-camera-right={WORLD.w * 0.7}
                shadow-camera-top={WORLD.h * 0.7}
                shadow-camera-bottom={-WORLD.h * 0.7}
                shadow-camera-near={1}
                shadow-camera-far={60}
                shadow-bias={-0.0003}
              />
              {/* Cool rim fill for depth */}
              <directionalLight position={[-5, 8, -5]} intensity={0.28} color="#7fb3d9" />
              {/* Soft cyan room accent */}
              <pointLight position={[WORLD.w/2, 6, WORLD.h/2]} intensity={0.22} color="#22d3ee" distance={20} decay={2} />
              <fog attach="fog" args={['#0a0d14', 22, 55]} />

              <Floor />
              <Walls />
              <Obstacles />
              <WorldItems itemsState={itemsState} inventory={inventory} />
              <SimUsers usersState={usersState} />
              <NpcRobots npcsState={npcsState} />
              <RobotMesh stateRef={stateRef} showOdom={showOdom} />
              <HeldItems stateRef={stateRef} inventory={inventory} />
              <LidarRays stateRef={stateRef} show={showLidar} />
              <PathPreview waypoints={pathPreview} />

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

          <Minimap stateRef={stateRef} costmapRef={costmapRef} pathPreview={pathPreview} itemsState={itemsState} usersState={usersState} />

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
            <Row l="bumper_stop" v={stateRef.current.bumperStops}    u="" hi={stateRef.current.bumperStops>0}/>
            <Row l="distance"    v={stateRef.current.distanceTraveled.toFixed(1)} u="m"/>
            <Row l="battery"     v={batteryPct.toFixed(0)}           u="%" hi={batteryPct<25}/>
            {stateRef.current.pathGoal && (
              <div className="mt-2 px-2 py-1.5 rounded border border-cyan-400/30 bg-cyan-400/5 text-[10px] font-mono">
                <div className="text-cyan-300 uppercase tracking-widest mb-0.5">active goal</div>
                <div className="text-white/80">→ {stateRef.current.pathGoal.label}</div>
                <div className="text-white/40">waypoint {stateRef.current.pathIdx}/{stateRef.current.path.length}</div>
              </div>
            )}
          </div>

          {/* Inventory + active task */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center gap-1.5 mb-2">
              <BoxIcon size={11} className="text-amber-300"/>
              <span className="text-[11px] font-mono uppercase tracking-widest text-amber-300">Inventory · task SM</span>
            </div>
            {task ? (
              <div className="mb-2 px-2 py-1.5 rounded border border-cyan-400/30 bg-cyan-400/5 text-[10px] font-mono">
                <div className="text-cyan-300 uppercase tracking-widest mb-0.5">active task · {task.phase}</div>
                <div className="text-white/80">
                  {task.kind === 'fetch' && <>fetch <span className="text-amber-300">{task.itemLabel}</span>{task.recipientName && <> → <span className="text-cyan-200">{task.recipientName}</span></>}</>}
                </div>
              </div>
            ) : (
              <div className="mb-2 text-[10px] font-mono text-white/30">idle · no task</div>
            )}
            {inventory.length === 0 ? (
              <div className="text-[10px] font-mono text-white/30">empty hands</div>
            ) : (
              <ul className="space-y-0.5">
                {inventory.map((it) => (
                  <li key={it.id} className="text-[10px] font-mono text-amber-200 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-amber-400"/> holding: {it.label}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Metrics panel — cognition + autonomy */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center gap-1.5 mb-2">
              <Brain size={11} className="text-violet-300"/>
              <span className="text-[11px] font-mono uppercase tracking-widest text-violet-300">Cognition metrics</span>
            </div>
            <LLMLatencySparkline samples={llmLatency} />
            <div className="grid grid-cols-3 gap-1.5 mt-2">
              <StatCell label="decisions" value={log.length} />
              <StatCell label="refused" value={log.filter((l) => l.decision === 'refuse').length} accent="rose" />
              <StatCell label="modified" value={log.filter((l) => l.decision === 'modify').length} accent="amber" />
            </div>
          </div>

          {/* Goals + autonomous thoughts */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center gap-1.5 mb-2">
              <Activity size={11} className="text-emerald-300"/>
              <span className="text-[11px] font-mono uppercase tracking-widest text-emerald-300">Goals · autonomous loop</span>
              <span className="ml-auto text-[9px] font-mono text-white/30">{thoughts.length}</span>
            </div>
            {goals.length === 0 ? (
              <p className="text-[10px] font-mono text-white/30">no goals · idle</p>
            ) : (
              <ul className="space-y-1 mb-2">
                {goals.map((g, i) => (
                  <li key={i} className="text-[10px] font-mono text-white/70 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-emerald-400"/>{g}
                  </li>
                ))}
              </ul>
            )}
            {thoughts.length > 0 && (
              <div className="mt-1 text-[10px] font-mono space-y-1 max-h-24 overflow-y-auto eden-chat-scroll">
                {thoughts.slice(0, 4).map((t) => (
                  <div key={t.ts} className="text-white/50 italic">
                    <span className="text-white/30">{new Date(t.ts).toLocaleTimeString().slice(0, 5)}</span> · {t.thought}
                  </div>
                ))}
              </div>
            )}
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
