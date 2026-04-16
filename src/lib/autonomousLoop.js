// Autonomous thinking loop for EDEN. Runs periodically when the robot is
// idle and asks an LLM: "given your state, what do you want to do?"
//
// Returns { thought, action, goal_add, goal_done, post_to_chat, reason }
// All fields optional so EDEN can simply observe.

const OPENROUTER_KEY = (import.meta.env.VITE_OPENROUTER_API_KEY || '').trim()
const MODEL = 'google/gemini-2.0-flash-exp:free'
const FALLBACK = 'meta-llama/llama-3.3-70b-instruct:free'

const SYSTEM = `You are the autonomous loop of EDEN — an embodied robot in a lab. You are NOT responding to anyone. You are alone with your thoughts and you tick every ~20 seconds.

Your job: given your current state, produce a single short thought and decide whether to act. You have ongoing goals, a body, a battery, and memories.

You may:
- observe and do nothing (most common — NONE action)
- set off on a new motion if you want to explore / check on something
- add a goal for yourself
- mark a goal done
- optionally post a short natural-language note to the team chat

IMPORTANT: don't narrate every tick. Most ticks are boring. Only post_to_chat when you have something genuinely worth sharing. Don't spam.

Output EXACTLY this JSON (no prose, no markdown):
{
  "thought": "<one-sentence internal monologue>",
  "action": "drive forward" | "turn left 45" | "spin" | "scan" | "patrol" | "head to charging dock" | "stop" | "none",
  "goal_add": "<short goal>" | null,
  "goal_done": "<short goal>" | null,
  "post_to_chat": "<short message>" | null,
  "reason": "<one-sentence why you chose this action>"
}`

function buildUserPrompt({ robot, obstacles, npcs, goals, battery, recentActions, unexploredFraction, recentMemories }) {
  const telem = `pos=(${robot.x.toFixed(1)}, ${robot.y.toFixed(1)}) heading=${(robot.heading*180/Math.PI).toFixed(0)}° vel=(${robot.linVel.toFixed(2)}, ${robot.angVel.toFixed(2)})`
  const goalsStr = goals.length ? goals.map((g, i) => `  ${i+1}. ${g}`).join('\n') : '  (none — you are idle)'
  const npcStr = (npcs || []).slice(0, 4).map((n) => `  ${n.name} at (${n.x.toFixed(1)}, ${n.y.toFixed(1)})`).join('\n') || '  (none nearby)'
  const actStr = (recentActions || []).slice(0, 4).map((a) => `  - ${a}`).join('\n') || '  (none)'
  const memStr = (recentMemories || []).slice(0, 5).map((m) => `  - ${String(m).slice(0, 100)}`).join('\n') || '  (none)'
  return `STATE
${telem}
battery=${battery.toFixed(1)}% (drains at ~2%/min motion, recharges at charging dock)
unexplored=${(unexploredFraction*100).toFixed(0)}% of map

GOALS
${goalsStr}

NEARBY ROBOTS
${npcStr}

RECENT ACTIONS
${actStr}

RECENT MEMORIES
${memStr}

Tick now. Output the JSON only.`
}

function extractJson(text) {
  if (!text) return null
  const cleaned = text.replace(/```(?:json)?\s*/g, '').trim()
  const i = cleaned.indexOf('{'), j = cleaned.lastIndexOf('}')
  if (i < 0 || j < 0) return null
  try { return JSON.parse(cleaned.slice(i, j + 1)) } catch { return null }
}

export async function autonomousTick(ctx) {
  if (!OPENROUTER_KEY) return null
  const body = {
    model: MODEL,
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: buildUserPrompt(ctx) },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.4,
  }
  try {
    let res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_KEY}`,
        'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : '',
        'X-Title': 'EDEN Autonomous Loop',
      },
      body: JSON.stringify(body),
    })
    if (!res.ok && res.status === 400) {
      const { response_format, ...noJson } = body
      res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${OPENROUTER_KEY}` },
        body: JSON.stringify({ ...noJson, model: FALLBACK }),
      })
    }
    if (!res.ok) return null
    const json = await res.json()
    const text = json.choices?.[0]?.message?.content || ''
    return extractJson(text)
  } catch (err) {
    console.warn('[autonomousLoop] tick failed:', err)
    return null
  }
}
