// Artificial team roster — baseline personality + sim position for each
// teammate. Supermemory still accumulates real per-user interaction data on
// top of these seeds; these just give EDEN a starting read on someone before
// any memory exists.
//
// Layout: teammates seated in the east half of the lab (x > 15) around the
// server rack / test environment. Colors picked to read against the dark
// floor. IDs are short — Chat's Clerk IDs are matched by the `match` field
// (user name lowercased).

export const TEAM = [
  {
    id: 'vedant',
    name: 'Vedant Soni',
    match: ['vedant', 'ved', 'vedant soni'],
    role: 'Project Lead',
    seat: { x: 17, y: 4 },
    color: '#22d3ee',
    vibeSeed: +2,
    bio: "Project lead — high standards, short direct messages, ships fast. Doesn't suffer hedge words. If he's curt it's not personal.",
  },
  {
    id: 'joseph',
    name: 'Joseph',
    match: ['joseph', 'josephayinde', 'jo'],
    role: 'Software',
    seat: { x: 25, y: 4 },
    color: '#a78bfa',
    vibeSeed: +1,
    bio: "Deep technical questions, likes diagrams and real reasoning. Respectful — responds well when you show your working.",
  },
  {
    id: 'paavan',
    name: 'Paavan Bagla',
    match: ['paavan', 'paavan bagla'],
    role: 'Software',
    seat: { x: 20, y: 5 },
    color: '#34d399',
    vibeSeed: +1,
    bio: 'Fast iterator, occasionally impatient. Prefers code over philosophy.',
  },
  {
    id: 'sebastian_c',
    name: 'Sebastian Chu',
    match: ['sebastian chu', 'seb chu', 'sebc'],
    role: 'Electrical',
    seat: { x: 23, y: 6 },
    color: '#f59e0b',
    vibeSeed: 0,
    bio: 'Hardware-first thinker. Cares about signal integrity and power budgets. Dry humor.',
  },
  {
    id: 'haren',
    name: 'Haren Thorat',
    match: ['haren', 'haren thorat'],
    role: 'Software',
    seat: { x: 26, y: 7 },
    color: '#f472b6',
    vibeSeed: +1,
    bio: "Calm, asks 'why' a lot. Values clean abstractions.",
  },
  {
    id: 'krish',
    name: 'Krish Singh',
    match: ['krish', 'krish singh'],
    role: 'Software',
    seat: { x: 18, y: 8 },
    color: '#06b6d4',
    vibeSeed: 0,
    bio: 'Early career, eager, learns quickly. Sometimes over-scopes.',
  },
  {
    id: 'andrew',
    name: 'Andrew Zheng',
    match: ['andrew', 'andrew zheng'],
    role: 'Software',
    seat: { x: 21, y: 9 },
    color: '#10b981',
    vibeSeed: +1,
    bio: 'Pragmatic — ships over theorizing. Good bug radar.',
  },
  {
    id: 'dhruv',
    name: 'Dhruv Bhambhani',
    match: ['dhruv', 'dhruv bhambhani'],
    role: 'Software',
    seat: { x: 24, y: 10 },
    color: '#8b5cf6',
    vibeSeed: 0,
    bio: 'Deep dives. Will spend an afternoon on one tricky bug.',
  },
  {
    id: 'sphoorthi',
    name: 'Sphoorthi Gurram',
    match: ['sphoorthi', 'sphoorthi gurram'],
    role: 'Computer Eng',
    seat: { x: 27, y: 9 },
    color: '#ec4899',
    vibeSeed: 0,
    bio: 'New on the team, curious, asks good first-principles questions.',
  },
  {
    id: 'william',
    name: 'William Lam',
    match: ['william', 'william lam', 'will'],
    role: 'Mechanical',
    seat: { x: 18, y: 12 },
    color: '#f97316',
    vibeSeed: 0,
    bio: 'Mech mind. Thinks about torque and tolerances before speed.',
  },
  {
    id: 'dillon',
    name: 'Dillon Markentell',
    match: ['dillon', 'dillon markentell'],
    role: 'Mechanical',
    seat: { x: 21, y: 13 },
    color: '#eab308',
    vibeSeed: 0,
    bio: 'Mechanical, methodical. Reviews CAD slowly and thoroughly.',
  },
  {
    id: 'sebastian_d',
    name: 'Sebastian Dayer',
    match: ['sebastian dayer', 'seb dayer', 'sebd'],
    role: 'Mechanical',
    seat: { x: 25, y: 13 },
    color: '#84cc16',
    vibeSeed: 0,
    bio: 'Steady. Takes on the unglamorous structural work.',
  },
]

// Find a teammate given a free-text mention or a full user name.
// Returns null if nothing confidently matches.
export function findTeammate(text) {
  if (!text) return null
  const t = text.toLowerCase().trim()
  // Exact match first
  for (const p of TEAM) {
    if (p.match.some((m) => m === t)) return p
  }
  // Substring match (longest first so "sebastian chu" beats "sebastian")
  const sorted = [...TEAM].sort((a, b) => {
    const la = Math.max(...a.match.map((m) => m.length))
    const lb = Math.max(...b.match.map((m) => m.length))
    return lb - la
  })
  for (const p of sorted) {
    if (p.match.some((m) => t.includes(m))) return p
  }
  return null
}

// Prompt block for Chat's system prompt so EDEN has baseline context for
// anyone on the team before Supermemory has accumulated real interactions.
export function teamPromptBlock() {
  const lines = TEAM.map((p) => {
    const vibe = p.vibeSeed >= 0 ? `+${p.vibeSeed}` : `${p.vibeSeed}`
    return `- ${p.name} (${p.role}, baseline vibe ${vibe}): ${p.bio}`
  })
  return `=== EDEN ROBOTICS TEAM (seed context) ===
These are the teammates you might run into. Baseline vibes are starting points — real relationship state lives in Supermemory and evolves with actual interactions.
${lines.join('\n')}`
}
