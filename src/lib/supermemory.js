// Supermemory v4 API client for EDEN chat
// Uses per-user containerTag + shared channel tag so the bot has both
// personal context ("what vedan said last week") and collective context.

const API = 'https://api.supermemory.ai/v4'
const KEY = (import.meta.env.VITE_SUPERMEMORY_API_KEY || '').trim()

const CHANNEL_TAG = 'eden-channel-eden-bot'

export function userTag(userId) {
  return `eden-user-${userId}`
}

function authed(extra = {}) {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${KEY}`,
    ...extra,
  }
}

export function supermemoryConfigured() {
  return typeof KEY === 'string' && KEY.length > 10
}

// Add a single memory. Stored against BOTH the user tag and the channel tag
// so retrieval can fuse personal + collective recall.
export async function addMemory({ content, userId, userName, role = 'user', extraMetadata = {} }) {
  if (!supermemoryConfigured() || !content) return null

  const metadata = {
    user_id: userId || 'unknown',
    user_name: userName || 'unknown',
    role,
    source: 'eden-chat',
    ts: Date.now(),
    ...extraMetadata,
  }

  try {
    // Store in user container
    const userRes = await fetch(`${API}/memories`, {
      method: 'POST',
      headers: authed(),
      body: JSON.stringify({
        memories: [{ content, isStatic: false, metadata }],
        containerTag: userTag(userId),
      }),
    })

    // Store in shared channel container (for cross-user recall)
    const channelRes = await fetch(`${API}/memories`, {
      method: 'POST',
      headers: authed(),
      body: JSON.stringify({
        memories: [{ content, isStatic: false, metadata }],
        containerTag: CHANNEL_TAG,
      }),
    })

    return { userOk: userRes.ok, channelOk: channelRes.ok }
  } catch (err) {
    console.warn('[supermemory] addMemory failed:', err)
    return null
  }
}

// Fused search: retrieves from the asking user's private tag AND the shared
// channel. Returns an array of { content, metadata, score, source }.
export async function searchMemories({ query, userId, limit = 5 }) {
  if (!supermemoryConfigured() || !query) return []

  const search = async (tag, source) => {
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: authed(),
        body: JSON.stringify({
          q: query,
          containerTag: tag,
          searchMode: 'hybrid',
          limit,
        }),
      })
      if (!res.ok) return []
      const json = await res.json()
      const results = json.results || json.memories || json.data || []
      return results.map((r) => ({
        content: r.content || r.text || r.chunk || '',
        metadata: r.metadata || {},
        score: r.score ?? r.relevance ?? 0,
        source,
      }))
    } catch (err) {
      console.warn(`[supermemory] search(${tag}) failed:`, err)
      return []
    }
  }

  const [personal, shared] = await Promise.all([
    userId ? search(userTag(userId), 'personal') : Promise.resolve([]),
    search(CHANNEL_TAG, 'channel'),
  ])

  // Dedup by content, prefer personal (higher signal) when tied
  const seen = new Map()
  for (const m of [...personal, ...shared]) {
    if (!m.content) continue
    const key = m.content.slice(0, 120)
    const existing = seen.get(key)
    if (!existing || m.score > existing.score) seen.set(key, m)
  }

  return Array.from(seen.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
}

// Summarize memories into a compact system-prompt injection block.
export function formatMemoriesForPrompt(memories) {
  if (!memories?.length) return ''
  const lines = memories.map((m, i) => {
    const who = m.metadata?.user_name ? `${m.metadata.user_name}` : 'someone'
    const when = m.metadata?.ts ? relativeTime(m.metadata.ts) : ''
    const tag = m.source === 'personal' ? '[personal]' : '[channel]'
    return `  ${i + 1}. ${tag} ${who} (${when}): "${m.content}"`
  })
  return `\n\n=== RELEVANT MEMORY (retrieved via Supermemory) ===\nYou recall these prior exchanges. Use them naturally if relevant; do not force them in.\n${lines.join('\n')}\n=== END MEMORY ===`
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

// ───── Perception / Input Layer: person data ─────
// Captures real identity into Supermemory as permanent (isStatic: true) records
// so EDEN's Cognitive Layer always has grounded context about who is in the room.

const PEOPLE_TAG = 'eden-people'

export async function captureIdentity({ userId, userName, email, avatarUrl, extra = {} }) {
  if (!supermemoryConfigured() || !userId) return null

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

  try {
    // Write to the user's private container AND to the channel-wide People index
    const body = (tag) => ({
      memories: [{ content, isStatic: true, metadata }],
      containerTag: tag,
    })
    await Promise.all([
      fetch(`${API}/memories`, { method: 'POST', headers: authed(), body: JSON.stringify(body(userTag(userId))) }),
      fetch(`${API}/memories`, { method: 'POST', headers: authed(), body: JSON.stringify(body(PEOPLE_TAG)) }),
    ])
    return { ok: true }
  } catch (err) {
    console.warn('[supermemory] captureIdentity failed:', err)
    return null
  }
}

// Add a free-form profile fact ("I'm the hardware lead", "my favorite robot is Spot")
export async function addProfileFact({ userId, userName, fact }) {
  if (!supermemoryConfigured() || !userId || !fact) return null
  const metadata = {
    kind: 'profile_fact',
    user_id: userId,
    user_name: userName,
    source: 'perception-layer',
    ts: Date.now(),
  }
  try {
    await Promise.all([
      fetch(`${API}/memories`, {
        method: 'POST', headers: authed(),
        body: JSON.stringify({
          memories: [{ content: `${userName}: ${fact}`, isStatic: true, metadata }],
          containerTag: userTag(userId),
        }),
      }),
      fetch(`${API}/memories`, {
        method: 'POST', headers: authed(),
        body: JSON.stringify({
          memories: [{ content: `${userName}: ${fact}`, isStatic: true, metadata }],
          containerTag: PEOPLE_TAG,
        }),
      }),
    ])
    return { ok: true }
  } catch (err) {
    console.warn('[supermemory] addProfileFact failed:', err)
    return null
  }
}

// Fetch identities currently in the room — used to ground the system prompt
// so EDEN knows WHO it's talking to before it reasons.
export async function getKnownPeople({ limit = 20 } = {}) {
  if (!supermemoryConfigured()) return []
  try {
    const res = await fetch(`${API}/search`, {
      method: 'POST', headers: authed(),
      body: JSON.stringify({
        q: 'person identity teammate',
        containerTag: PEOPLE_TAG,
        searchMode: 'hybrid',
        limit,
      }),
    })
    if (!res.ok) return []
    const json = await res.json()
    const results = json.results || json.memories || json.data || []
    return results.map((r) => ({
      content: r.content || r.text || '',
      metadata: r.metadata || {},
    })).filter((r) => r.metadata?.kind === 'identity' || r.metadata?.kind === 'profile_fact' || r.content)
  } catch (err) {
    console.warn('[supermemory] getKnownPeople failed:', err)
    return []
  }
}

export function formatPeopleForPrompt(people, currentUser) {
  if (!people?.length) return ''
  const lines = people.slice(0, 12).map((p) => {
    const who = p.metadata?.user_name || 'unknown'
    const email = p.metadata?.email ? ` (${p.metadata.email})` : ''
    const role = p.metadata?.role ? ` — ${p.metadata.role}` : ''
    const first = p.content?.split('\n')[0]?.replace(/^Person:\s*/, '') || who
    return `  - ${first}${email}${role}`
  })
  const meLine = currentUser
    ? `\n\nYou are currently talking with: **${currentUser.name}** (${currentUser.email || 'no email'}). Address them by name naturally.`
    : ''
  return `\n\n=== PEOPLE IN THIS WORKSPACE (from Perception Layer) ===\n${lines.join('\n')}${meLine}\n=== END PEOPLE ===`
}

export { CHANNEL_TAG, PEOPLE_TAG, relativeTime }
