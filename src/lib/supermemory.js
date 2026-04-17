// Memory store — localStorage-backed (drop-in replacement for the old
// Supermemory API client, which got blocked by CORS from github.io).
//
// Same public API as before so Chat.jsx / Simulator.jsx don't need changes:
//   addMemory, searchMemories, formatMemoriesForPrompt, supermemoryConfigured,
//   captureIdentity, addProfileFact, getKnownPeople, formatPeopleForPrompt,
//   addVibe, getVibeHistory, formatVibeForPrompt, vibeLabel, relativeTime.
//
// Storage scheme (all per-origin, shared across tabs via the localStorage event):
//   eden:mem:user:<userId>  → array of memory entries (personal scope)
//   eden:mem:channel        → array of channel-wide memory entries
//   eden:mem:people         → array of identity + profile-fact entries
//   eden:mem:vibes:<userId> → array of vibe deltas for a user
//
// Retrieval = token-based scoring (phrase bonus + word-overlap + recency).
// Capped at 200 entries per key so the bundle stays fast.

const CHANNEL_TAG = 'eden-channel-eden-bot'
const PEOPLE_TAG = 'eden-people'
const VIBE_TAG = 'eden-vibes'
const MAX_PER_KEY = 200

export function userTag(userId) {
  return `eden-user-${userId}`
}

const SAFE = typeof window !== 'undefined' && !!window.localStorage

function readList(key) {
  if (!SAFE) return []
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return []
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr : []
  } catch { return [] }
}

function writeList(key, arr) {
  if (!SAFE) return
  try {
    const capped = arr.length > MAX_PER_KEY ? arr.slice(arr.length - MAX_PER_KEY) : arr
    window.localStorage.setItem(key, JSON.stringify(capped))
  } catch (err) {
    // Quota-exceeded — drop oldest half and retry
    try {
      window.localStorage.setItem(key, JSON.stringify(arr.slice(-Math.floor(MAX_PER_KEY / 2))))
    } catch { /* give up silently */ }
  }
}

function pushTo(key, entry) {
  const list = readList(key)
  list.push(entry)
  writeList(key, list)
}

export function supermemoryConfigured() {
  // Always true now — localStorage is free and always available.
  return SAFE
}

// ─────────── Add / search memories ───────────

export async function addMemory({ content, userId, userName, role = 'user', extraMetadata = {} }) {
  if (!content) return null
  const metadata = {
    user_id: userId || 'unknown',
    user_name: userName || 'unknown',
    role,
    source: 'eden-chat',
    ts: Date.now(),
    ...extraMetadata,
  }
  const entry = { content, isStatic: false, metadata }
  if (userId) pushTo(`eden:mem:user:${userId}`, entry)
  pushTo('eden:mem:channel', entry)
  return { userOk: true, channelOk: true }
}

function scoreEntry(entry, queryWords, queryPhrase) {
  const text = (entry.content || '').toLowerCase()
  if (!text) return 0
  let score = 0
  if (queryPhrase && text.includes(queryPhrase)) score += 5
  for (const w of queryWords) {
    if (w.length < 3) continue
    if (text.includes(w)) score += 1
  }
  // Recency boost — 0..2 across the last day
  const age = Date.now() - (entry.metadata?.ts || 0)
  const hrs = age / (1000 * 60 * 60)
  if (hrs < 1) score += 2
  else if (hrs < 24) score += 1
  return score
}

export async function searchMemories({ query, userId, limit = 5 }) {
  if (!query) return []
  const q = query.toLowerCase().trim()
  const words = q.split(/\s+/).filter(Boolean)

  const personal = userId ? readList(`eden:mem:user:${userId}`) : []
  const shared = readList('eden:mem:channel')

  const score = (list, source) => list
    .map((e) => ({ ...e, _score: scoreEntry(e, words, q), source }))
    .filter((e) => e._score > 0)

  const all = [...score(personal, 'personal'), ...score(shared, 'channel')]
  // Dedupe by content prefix, keep the higher-scored / personal variant
  const seen = new Map()
  for (const m of all) {
    const key = (m.content || '').slice(0, 120)
    const existing = seen.get(key)
    if (!existing || m._score > existing._score) seen.set(key, m)
  }

  return Array.from(seen.values())
    .sort((a, b) => b._score - a._score)
    .slice(0, limit)
    .map((m) => ({
      content: m.content,
      metadata: m.metadata,
      score: m._score,
      source: m.source,
    }))
}

export function formatMemoriesForPrompt(memories) {
  if (!memories?.length) return ''
  const lines = memories.map((m, i) => {
    const who = m.metadata?.user_name ? `${m.metadata.user_name}` : 'someone'
    const when = m.metadata?.ts ? relativeTime(m.metadata.ts) : ''
    const tag = m.source === 'personal' ? '[personal]' : '[channel]'
    return `  ${i + 1}. ${tag} ${who} (${when}): "${m.content}"`
  })
  return `\n\n=== RELEVANT MEMORY (retrieved from local store) ===\nYou recall these prior exchanges. Use them naturally if relevant; do not force them in.\n${lines.join('\n')}\n=== END MEMORY ===`
}

function relativeTime(ts) {
  const diff = Date.now() - ts
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}

// ─────────── Identity / people ───────────

export async function captureIdentity({ userId, userName, email, avatarUrl, extra = {} }) {
  if (!userId) return null
  const joinedAt = new Date().toISOString()
  const content = [
    `Person: ${userName}`,
    email ? `Email: ${email}` : null,
    `User ID: ${userId}`,
    avatarUrl ? `Avatar: ${avatarUrl}` : null,
    `First seen: ${joinedAt}`,
    extra.role ? `Role: ${extra.role}` : null,
    extra.bio ? `Bio: ${extra.bio}` : null,
  ].filter(Boolean).join('\n')
  const metadata = {
    kind: 'identity',
    user_id: userId,
    user_name: userName,
    email: email || null,
    avatar_url: avatarUrl || null,
    joined_at: joinedAt,
    source: 'perception-layer',
    ...extra,
  }
  const entry = { content, isStatic: true, metadata }

  // Upsert: replace any existing identity row for this user
  const peopleKey = 'eden:mem:people'
  const people = readList(peopleKey).filter((p) => p.metadata?.user_id !== userId || p.metadata?.kind !== 'identity')
  people.push(entry)
  writeList(peopleKey, people)
  pushTo(`eden:mem:user:${userId}`, entry)
  return { ok: true }
}

export async function addProfileFact({ userId, userName, fact }) {
  if (!userId || !fact) return null
  const metadata = {
    kind: 'profile_fact',
    user_id: userId,
    user_name: userName,
    source: 'perception-layer',
    ts: Date.now(),
  }
  const entry = { content: `${userName}: ${fact}`, isStatic: true, metadata }
  pushTo('eden:mem:people', entry)
  pushTo(`eden:mem:user:${userId}`, entry)
  return { ok: true }
}

export async function getKnownPeople({ limit = 20 } = {}) {
  const list = readList('eden:mem:people')
  return list
    .filter((r) => r.metadata?.kind === 'identity' || r.metadata?.kind === 'profile_fact' || r.content)
    .slice(-limit)
    .map((r) => ({ content: r.content || '', metadata: r.metadata || {} }))
}

export function formatPeopleForPrompt(people, currentUser) {
  if (!people?.length && !currentUser) return ''
  const lines = (people || []).slice(0, 12).map((p) => {
    const who = p.metadata?.user_name || 'unknown'
    const email = p.metadata?.email ? ` (${p.metadata.email})` : ''
    const role = p.metadata?.role ? ` — ${p.metadata.role}` : ''
    const first = p.content?.split('\n')[0]?.replace(/^Person:\s*/, '') || who
    return `  - ${first}${email}${role}`
  })
  const meLine = currentUser
    ? `\n\nYou are currently talking with: **${currentUser.name}** (${currentUser.email || 'no email'}). Address them by name naturally.`
    : ''
  const header = lines.length
    ? `\n\n=== PEOPLE IN THIS WORKSPACE (from Perception Layer) ===\n${lines.join('\n')}${meLine}\n=== END PEOPLE ===`
    : (meLine ? `\n\n=== PEOPLE IN THIS WORKSPACE ===${meLine}\n=== END PEOPLE ===` : '')
  return header
}

// ─────────── Vibes / relationship ───────────

const VIBE_CLAMP = 10

export async function addVibe({ userId, userName, delta, reason }) {
  if (!userId) return null
  const d = Math.max(-5, Math.min(5, Math.round(Number(delta) || 0)))
  if (d === 0) return null
  const content = `VIBE ${d > 0 ? '+' : ''}${d} toward ${userName}: ${reason || '(no reason)'}`
  const metadata = {
    kind: 'vibe',
    user_id: userId,
    user_name: userName,
    delta: d,
    reason: reason || '',
    ts: Date.now(),
    source: 'cognitive-layer',
  }
  const entry = { content, isStatic: false, metadata }
  pushTo(`eden:mem:vibes:${userId}`, entry)
  pushTo(`eden:mem:user:${userId}`, entry)
  return { ok: true, delta: d }
}

export async function getVibeHistory({ userId, limit = 30 }) {
  if (!userId) return { total: 0, count: 0, recent: [] }
  const rows = readList(`eden:mem:vibes:${userId}`)
    .filter((r) => r.metadata?.kind === 'vibe')
    .map((r) => ({
      delta: Number(r.metadata?.delta) || 0,
      reason: r.metadata?.reason || '',
      ts: Number(r.metadata?.ts) || 0,
    }))
    .sort((a, b) => b.ts - a.ts)
  const total = rows.reduce((s, r) => s + r.delta, 0)
  return { total, count: rows.length, recent: rows.slice(0, 8) }
}

export function vibeLabel(total) {
  if (total >= 6) return { label: 'loves', emoji: '🥰', tone: 'text-emerald-300', border: 'border-emerald-400/30', bg: 'bg-emerald-500/10' }
  if (total >= 3) return { label: 'likes', emoji: '😊', tone: 'text-emerald-300', border: 'border-emerald-400/30', bg: 'bg-emerald-500/10' }
  if (total >= 1) return { label: 'warming', emoji: '🙂', tone: 'text-cyan-300', border: 'border-cyan-400/30', bg: 'bg-cyan-500/10' }
  if (total === 0) return { label: 'neutral', emoji: '😐', tone: 'text-white/60', border: 'border-white/20', bg: 'bg-white/5' }
  if (total >= -2) return { label: 'uncertain', emoji: '🤨', tone: 'text-amber-300', border: 'border-amber-400/30', bg: 'bg-amber-500/10' }
  if (total >= -5) return { label: 'annoyed', emoji: '😒', tone: 'text-orange-300', border: 'border-orange-400/30', bg: 'bg-orange-500/10' }
  return { label: 'done with', emoji: '🙄', tone: 'text-rose-300', border: 'border-rose-400/30', bg: 'bg-rose-500/10' }
}

export function formatVibeForPrompt(history, userName) {
  if (!history || history.count === 0) {
    return `\n\n=== RELATIONSHIP WITH ${userName} ===\nNo history yet — this is your first real interaction. Form an honest opinion as you go.\n=== END RELATIONSHIP ===`
  }
  const clamped = Math.max(-VIBE_CLAMP, Math.min(VIBE_CLAMP, history.total))
  const mood = vibeLabel(clamped)
  const recent = history.recent.slice(0, 6).map((r) => `  ${r.delta > 0 ? '+' : ''}${r.delta}: ${r.reason}`).join('\n')
  return `\n\n=== RELATIONSHIP WITH ${userName} ===\nRunning total: ${clamped} (mood: ${mood.label} ${mood.emoji}) — based on ${history.count} past judgments\nRecent notes:\n${recent}\n\nLet this shape your response. If you genuinely don't vibe with them, you can be dry, sarcastic, or refuse even reasonable requests — be honest about why. If you like them, be warmer. Don't fake a mood you don't have.\n=== END RELATIONSHIP ===`
}

export { CHANNEL_TAG, PEOPLE_TAG, VIBE_TAG, relativeTime }
