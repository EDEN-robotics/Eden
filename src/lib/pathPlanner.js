// A* global planner over the costmap. Returns a list of world-space
// waypoints from start to goal that respect the inflation layer.
//
// Heuristic: octile distance (8-connected grid).
// Cost: base + inflated cost (0..254) so the planner leans away from walls.

function octile(dc, dr) {
  const ac = Math.abs(dc), ar = Math.abs(dr)
  return (ac + ar) + (Math.SQRT2 - 2) * Math.min(ac, ar)
}

// 8-neighborhood, cardinal cost 1.0, diagonal cost sqrt(2)
const NEIGHBORS = [
  [ 1, 0, 1], [-1, 0, 1], [0,  1, 1], [0, -1, 1],
  [ 1, 1, Math.SQRT2], [-1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, -1, Math.SQRT2],
]

// Minimal binary heap keyed by .f
function pushHeap(heap, node) {
  heap.push(node)
  let i = heap.length - 1
  while (i > 0) {
    const p = (i - 1) >> 1
    if (heap[p].f <= heap[i].f) break
    ;[heap[p], heap[i]] = [heap[i], heap[p]]
    i = p
  }
}
function popHeap(heap) {
  if (heap.length === 0) return null
  const top = heap[0]
  const last = heap.pop()
  if (heap.length > 0) {
    heap[0] = last
    let i = 0
    const n = heap.length
    while (true) {
      const l = i * 2 + 1, r = l + 1
      let best = i
      if (l < n && heap[l].f < heap[best].f) best = l
      if (r < n && heap[r].f < heap[best].f) best = r
      if (best === i) break
      ;[heap[best], heap[i]] = [heap[i], heap[best]]
      i = best
    }
  }
  return top
}

function nearestFreeCell(costmap, c, r, maxRadius = 6) {
  if (!costmap.isCellBlocked(c, r)) return { c, r }
  for (let rad = 1; rad <= maxRadius; rad++) {
    for (let dr = -rad; dr <= rad; dr++) {
      for (let dc = -rad; dc <= rad; dc++) {
        if (Math.max(Math.abs(dc), Math.abs(dr)) !== rad) continue
        if (!costmap.isCellBlocked(c + dc, r + dr)) return { c: c + dc, r: r + dr }
      }
    }
  }
  return null
}

export function planPath(costmap, startXY, goalXY, { maxExpansions = 20000 } = {}) {
  const start = costmap.worldToCell(startXY.x, startXY.y)
  const goal = costmap.worldToCell(goalXY.x, goalXY.y)
  const snappedStart = nearestFreeCell(costmap, start.c, start.r)
  const snappedGoal = nearestFreeCell(costmap, goal.c, goal.r)
  if (!snappedStart || !snappedGoal) {
    return { ok: false, reason: 'start_or_goal_unreachable', waypoints: [], expansions: 0 }
  }

  const { cols } = costmap
  const heap = []
  const cameFrom = new Map()
  const gScore = new Map()
  const startKey = snappedStart.r * cols + snappedStart.c
  const goalKey = snappedGoal.r * cols + snappedGoal.c

  gScore.set(startKey, 0)
  pushHeap(heap, { c: snappedStart.c, r: snappedStart.r, g: 0, f: octile(snappedGoal.c - snappedStart.c, snappedGoal.r - snappedStart.r) })

  const closed = new Uint8Array(costmap.cols * costmap.rows)
  let expansions = 0
  while (heap.length && expansions < maxExpansions) {
    const cur = popHeap(heap)
    const curKey = cur.r * cols + cur.c
    if (closed[curKey]) continue
    closed[curKey] = 1
    expansions++
    if (curKey === goalKey) {
      // reconstruct
      const path = [curKey]
      let k = curKey
      while (cameFrom.has(k)) {
        k = cameFrom.get(k)
        path.push(k)
      }
      path.reverse()
      const waypoints = path.map((key) => {
        const c = key % cols, r = (key - c) / cols
        return costmap.cellToWorld(c, r)
      })
      return { ok: true, waypoints: simplify(waypoints), expansions }
    }
    for (const [dc, dr, stepCost] of NEIGHBORS) {
      const nc = cur.c + dc, nr = cur.r + dr
      if (nc < 0 || nc >= cols || nr < 0 || nr >= costmap.rows) continue
      if (costmap.isCellBlocked(nc, nr)) continue
      const nKey = nr * cols + nc
      if (closed[nKey]) continue
      const inflationCost = costmap.inflated[nKey] / 64 // 0..~4
      const tentative = cur.g + stepCost + inflationCost
      if (tentative < (gScore.get(nKey) ?? Infinity)) {
        gScore.set(nKey, tentative)
        cameFrom.set(nKey, curKey)
        const h = octile(snappedGoal.c - nc, snappedGoal.r - nr)
        pushHeap(heap, { c: nc, r: nr, g: tentative, f: tentative + h })
      }
    }
  }
  return { ok: false, reason: expansions >= maxExpansions ? 'max_expansions' : 'no_path', waypoints: [], expansions }
}

// Collinear-triple simplification — keep the path light for visualization & following
function simplify(pts) {
  if (pts.length < 3) return pts
  const out = [pts[0]]
  for (let i = 1; i < pts.length - 1; i++) {
    const a = out[out.length - 1], b = pts[i], c = pts[i + 1]
    const ax = b.x - a.x, ay = b.y - a.y
    const bx = c.x - b.x, by = c.y - b.y
    const cross = ax * by - ay * bx
    if (Math.abs(cross) > 1e-3) out.push(b)
  }
  out.push(pts[pts.length - 1])
  return out
}

// Is there a direct line of sight (no inflated cell >= 200) from a→b?
// Uses the same Bresenham pattern as costmap updates.
export function lineOfSight(costmap, a, b) {
  const aC = costmap.worldToCell(a.x, a.y)
  const bC = costmap.worldToCell(b.x, b.y)
  let c0 = aC.c, r0 = aC.r
  const c1 = bC.c, r1 = bC.r
  const dc = Math.abs(c1 - c0), dr = Math.abs(r1 - r0)
  const sc = c0 < c1 ? 1 : -1, sr = r0 < r1 ? 1 : -1
  let err = dc - dr
  let steps = 0
  const maxSteps = dc + dr + 2
  while (steps++ < maxSteps) {
    if (costmap.isCellBlocked(c0, r0)) return false
    if (c0 === c1 && r0 === r1) return true
    const e2 = err * 2
    if (e2 > -dr) { err -= dr; c0 += sc }
    if (e2 < dc)  { err += dc; r0 += sr }
  }
  return true
}
