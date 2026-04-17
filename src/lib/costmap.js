// 2D occupancy costmap — Nav2-style.
//
// A fixed-resolution grid (cells of `res` meters) that tracks three states
// per cell: UNKNOWN, FREE, OCCUPIED. Cells get updated online from LIDAR
// rays using a log-odds Bayesian update (bounded), then a Gaussian-ish
// inflation radius adds soft cost around obstacles so the planner keeps
// distance from walls.
//
// Math conventions:
//   world (x, y) in meters, origin (0, 0) at map corner.
//   grid (c, r) in cells, c = floor(x / res), r = floor(y / res).
//   log-odds stored as Float32 for speed; + = occupied, - = free.

const LO_FREE = -0.4      // per free hit
const LO_OCC = 0.85       // per occupied hit
const LO_MIN = -4.0
const LO_MAX = 4.0
const LO_OCC_THRESH = 0.7 // log-odds above = treated as hard obstacle
const LO_FREE_THRESH = -0.2

export function createCostmap({ worldW, worldH, res = 0.25, inflateRadius = 0.45 }) {
  const cols = Math.ceil(worldW / res)
  const rows = Math.ceil(worldH / res)
  const logOdds = new Float32Array(cols * rows)
  const inflated = new Uint8Array(cols * rows) // 0 free, 255 lethal, 1..254 cost
  const inflateCells = Math.ceil(inflateRadius / res)

  function idx(c, r) { return r * cols + c }
  function inBounds(c, r) { return c >= 0 && c < cols && r >= 0 && r < rows }

  function worldToCell(x, y) {
    return { c: Math.floor(x / res), r: Math.floor(y / res) }
  }
  function cellToWorld(c, r) {
    return { x: (c + 0.5) * res, y: (r + 0.5) * res }
  }

  // Bresenham-style ray walk. Updates cells along the ray as FREE, then the
  // terminating cell as OCCUPIED if the ray actually hit something (distance
  // < maxRange — otherwise it's a no-hit and we only mark free).
  function integrateRay(ox, oy, ex, ey, hit) {
    const a = worldToCell(ox, oy)
    const b = worldToCell(ex, ey)
    let c0 = a.c, r0 = a.r
    const c1 = b.c, r1 = b.r
    const dc = Math.abs(c1 - c0), dr = Math.abs(r1 - r0)
    const sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1
    let err = dc - dr
    let steps = 0
    const maxSteps = dc + dr + 2
    while (steps++ < maxSteps) {
      if (inBounds(c0, r0) && !(c0 === c1 && r0 === r1)) {
        const i = idx(c0, r0)
        logOdds[i] = Math.max(LO_MIN, logOdds[i] + LO_FREE)
      }
      if (c0 === c1 && r0 === r1) break
      const e2 = err * 2
      if (e2 > -dr) { err -= dr; c0 += sc }
      if (e2 < dc)  { err += dc; r0 += sr }
    }
    if (hit && inBounds(b.c, b.r)) {
      const i = idx(b.c, b.r)
      logOdds[i] = Math.min(LO_MAX, logOdds[i] + LO_OCC)
    }
  }

  function integrateLidar({ x, y, heading, ranges, angle_min, angle_max, range_max }) {
    const n = ranges.length
    if (n < 2) return
    const step = (angle_max - angle_min) / (n - 1)
    for (let i = 0; i < n; i++) {
      const a = heading + angle_min + i * step
      const r = ranges[i]
      const hit = r < range_max - 0.05
      const ex = x + Math.cos(a) * r
      const ey = y + Math.sin(a) * r
      integrateRay(x, y, ex, ey, hit)
    }
  }

  function rebuildInflation() {
    inflated.fill(0)
    // First pass: mark cells whose log-odds crossed occupied threshold as lethal
    const lethalCells = []
    for (let i = 0; i < logOdds.length; i++) {
      if (logOdds[i] >= LO_OCC_THRESH) {
        inflated[i] = 255
        const c = i % cols, r = (i - c) / cols
        lethalCells.push([c, r])
      }
    }
    // Second pass: Chebyshev-style inflation with linear falloff
    const rad = inflateCells
    for (const [c, r] of lethalCells) {
      for (let dr = -rad; dr <= rad; dr++) {
        for (let dc = -rad; dc <= rad; dc++) {
          const nc = c + dc, nr = r + dr
          if (!inBounds(nc, nr)) continue
          const j = idx(nc, nr)
          if (inflated[j] === 255) continue
          const d = Math.max(Math.abs(dc), Math.abs(dr))
          // 254 right next to lethal, falling to 1 at the edge of inflation
          const cost = Math.round(254 * (1 - d / (rad + 1)))
          if (cost > inflated[j]) inflated[j] = cost
        }
      }
    }
  }

  function costAtWorld(x, y) {
    const { c, r } = worldToCell(x, y)
    if (!inBounds(c, r)) return 255
    return inflated[idx(c, r)]
  }

  function isCellBlocked(c, r) {
    if (!inBounds(c, r)) return true
    return inflated[idx(c, r)] >= 200
  }

  function snapshot() {
    // Return a flat byte view for minimap drawing: 0 free, 1 unknown, 2 occupied (inflation band=3)
    const out = new Uint8Array(cols * rows)
    for (let i = 0; i < logOdds.length; i++) {
      const lo = logOdds[i]
      if (inflated[i] >= 200) out[i] = 2
      else if (inflated[i] > 0) out[i] = 3
      else if (lo < LO_FREE_THRESH) out[i] = 0
      else out[i] = 1
    }
    return { cols, rows, res, data: out }
  }

  return {
    cols, rows, res,
    integrateLidar,
    rebuildInflation,
    costAtWorld,
    isCellBlocked,
    worldToCell,
    cellToWorld,
    snapshot,
    get logOdds() { return logOdds },
    get inflated() { return inflated },
  }
}
