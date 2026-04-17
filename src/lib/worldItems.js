// Pickable items in the lab. Every item has a world position and can be
// picked up / dropped by EDEN via the task state machine in Simulator.jsx.
//
// Coordinates are placed roughly: lab-floor scatter, near workbenches, on
// the server rack, and in the parts-bin zone. Each has a 3D render hint
// (kind) so the renderer knows what geometry to use.

export const ITEMS = [
  { id: 'pencil',      label: 'pencil',      kind: 'stick',   color: '#fbbf24', x: 8.0,  y: 6.4,  len: 0.18, r: 0.012 },
  { id: 'screwdriver', label: 'screwdriver', kind: 'stick',   color: '#ef4444', x: 8.2,  y: 9.2,  len: 0.22, r: 0.015 },
  { id: 'caliper',     label: 'caliper',     kind: 'rect',    color: '#e5e7eb', x: 6.8,  y: 6.4,  w: 0.18, d: 0.05, h: 0.02 },
  { id: 'multimeter',  label: 'multimeter',  kind: 'rect',    color: '#eab308', x: 6.8,  y: 9.2,  w: 0.16, d: 0.10, h: 0.04 },
  { id: 'laptop',      label: 'laptop',      kind: 'laptop',  color: '#94a3b8', x: 7.2,  y: 9.0,  w: 0.34, d: 0.24, h: 0.03 },
  { id: 'clipboard',   label: 'clipboard',   kind: 'rect',    color: '#c7d2fe', x: 5.8,  y: 9.0,  w: 0.22, d: 0.30, h: 0.02 },
  { id: 'coffee_mug',  label: 'coffee mug',  kind: 'mug',     color: '#0f172a', x: 6.4,  y: 6.8,  r: 0.055, h: 0.11 },
  { id: 'water_bottle',label: 'water bottle',kind: 'bottle',  color: '#38bdf8', x: 12.2, y: 8.2,  r: 0.035, h: 0.22 },
  { id: 'phone',       label: 'phone',       kind: 'rect',    color: '#0a0d12', x: 13.6, y: 7.4,  w: 0.07, d: 0.14, h: 0.012 },
  { id: 'book',        label: 'book',        kind: 'rect',    color: '#7c2d12', x: 11.8, y: 13.4, w: 0.16, d: 0.22, h: 0.04 },
  { id: 'toolkit',     label: 'toolkit',     kind: 'rect',    color: '#b91c1c', x: 27.0, y: 5.0,  w: 0.32, d: 0.22, h: 0.12 },
  { id: 'usb_drive',   label: 'usb drive',   kind: 'stick',   color: '#6366f1', x: 22.0, y: 10.6, len: 0.06, r: 0.010 },
]

// Items are placed ON surfaces. These "z" heights make them sit above the
// floor/bench plane naturally.
export const ITEM_DROP_HEIGHT = 0.02

// Fuzzy item lookup — matches label or id. Prefers longest substring so
// "coffee mug" wins over "coffee".
export function findItem(text) {
  if (!text) return null
  const t = text.toLowerCase()
  const labels = [...ITEMS].sort((a, b) => b.label.length - a.label.length)
  for (const it of labels) {
    if (t.includes(it.label)) return it
  }
  for (const it of ITEMS) {
    if (t.includes(it.id.replace('_', ' '))) return it
  }
  return null
}
