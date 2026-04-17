// Safety bumper: a pre-commit motion filter.
//
// Given a commanded (linear, angular) and the current LIDAR scan in robot
// frame, decide how much to scale the command so the robot won't slam into
// whatever is in front of it. Pure, no state. Called every physics tick.
//
// Front-arc definition: ±45° of the heading direction (the direction of
// travel for positive linear velocity). When reversing, mirror the arc.

// Safety band parameters (meters / seconds)
export const STOP_DIST = 0.30     // inside this: zero linear
export const SLOW_DIST = 0.90     // inside this: scale down
export const ARC_HALF = Math.PI / 4

export function clipForSafety({ cmdLin, cmdAng, lidar, fov, reversing = false }) {
  if (!lidar || lidar.length === 0 || Math.abs(cmdLin) < 0.02) {
    return { cmdLin, cmdAng, nearest: Infinity, reason: null }
  }
  const n = lidar.length
  const step = fov / (n - 1)
  const frontAngle = reversing ? Math.PI : 0
  let nearest = Infinity
  for (let i = 0; i < n; i++) {
    // Ray angle relative to robot heading: from -fov/2 to +fov/2
    const a = -fov / 2 + i * step
    const delta = Math.abs(((a - frontAngle) + Math.PI * 3) % (Math.PI * 2) - Math.PI)
    if (delta > ARC_HALF) continue
    if (lidar[i] < nearest) nearest = lidar[i]
  }
  if (!isFinite(nearest)) return { cmdLin, cmdAng, nearest, reason: null }
  if (nearest < STOP_DIST) {
    return { cmdLin: 0, cmdAng, nearest, reason: `bumper_stop@${nearest.toFixed(2)}m` }
  }
  if (nearest < SLOW_DIST) {
    const scale = (nearest - STOP_DIST) / (SLOW_DIST - STOP_DIST)
    return { cmdLin: cmdLin * scale, cmdAng, nearest, reason: `bumper_slow@${nearest.toFixed(2)}m×${scale.toFixed(2)}` }
  }
  return { cmdLin, cmdAng, nearest, reason: null }
}
