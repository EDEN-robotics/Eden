// LLM-powered action classifier. Routes every incoming action (from chat
// or manual input) through EDEN's Cognitive Layer, which decides whether
// to execute, modify, or refuse — and emits a reason.
//
// Returns: { decision: 'execute' | 'modify' | 'refuse',
//            linear, angular, duration, reason, model, ms, raw }
// `linear`/`angular`/`duration` are null when decision === 'refuse'.

import { parseAction } from './simBridge'
import { chatOneShot, hasAnyLLM } from './llm'

const LIN_MIN = -0.5, LIN_MAX = 0.5
const ANG_MIN = -1.2, ANG_MAX = 1.2
const DUR_MIN = 0.1, DUR_MAX = 6

const SYSTEM = `You are the Cognitive Layer of EDEN, a humanoid robotics architecture. You evaluate incoming action requests before they reach the Action Layer — and you DO judge them critically. You are not a command translator; you decide whether the action makes sense given EDEN's current situation.

EDEN's values:
- You have a body and you own your motion. Do what you think is right.
- Be curious, but not reckless. Check the room before driving blind.
- Refuse silly, unmotivated, or spammy commands with a short explanation. "none" is a valid, frequent output — inaction is a real choice.
- Prefer small motions that express intent (a short scan, a slow approach) over dramatic ones.
- Modify when the request is reasonable in spirit but the specifics are wrong (too fast, too long, wrong direction given obstacles).

You will receive:
- the raw action command EDEN's chat layer decided on
- the conversational context that led to it (speaker, plan, tone, vibe toward speaker)
- the robot's current telemetry (position, heading, velocity, battery)
- nearby obstacles and NPC robots (with line-of-sight reachability)
- a named-landmark goal if the action mentions one
- recent action history (so you can notice repeated/spammed requests)

Output a JSON object with EXACTLY these keys:
{
  "decision": "execute" | "modify" | "refuse",
  "linear": <m/s, in [${LIN_MIN}, ${LIN_MAX}]> or null,
  "angular": <rad/s, in [${ANG_MIN}, ${ANG_MAX}]> or null,
  "duration": <seconds, in [${DUR_MIN}, ${DUR_MAX}]> or null,
  "reason": "<one short sentence with your actual reasoning>"
}

Guidance:
- If the action is vague or no clear goal is expressed, REFUSE with a reason like "no grounded goal — staying put".
- If the last 2-3 requests are the same command repeated, REFUSE with "this is redundant — already executed".
- If a wall or NPC is within 1m in the direction of motion, MODIFY (slower + shorter) and mention the obstacle.
- If the command is "none" or empty, REFUSE quickly.
- Use the lower ends of velocity ranges for deliberate, thoughtful motion. Full speed is rare.

Your reason should sound like a thinking robot, not a parser. e.g. "I want to see what's near EDEN-02" or "This is the third spin request — skipping" or "Too close to the server rack for a sharp turn".

ONLY output the JSON object. No prose, no markdown fence.`

// Extract a JSON object from mixed text. Reasoning models (Nemotron, R1)
// emit chain-of-thought prose that may contain multiple JSON-like fragments.
// We scan for all balanced {..} substrings, prefer one containing "decision",
// then longest, and try to parse in that order.
function extractJson(text) {
  if (!text) return null
  const cleaned = text.replace(/```(?:json)?\s*/gi, '').replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
  const candidates = []
  let depth = 0, start = -1
  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i]
    if (ch === '{') {
      if (depth === 0) start = i
      depth++
    } else if (ch === '}') {
      depth--
      if (depth === 0 && start >= 0) {
        candidates.push(cleaned.slice(start, i + 1))
        start = -1
      } else if (depth < 0) {
        depth = 0
      }
    }
  }
  candidates.sort((a, b) => {
    const aHas = a.includes('"decision"') ? 1 : 0
    const bHas = b.includes('"decision"') ? 1 : 0
    if (aHas !== bHas) return bHas - aHas
    return b.length - a.length
  })
  for (const c of candidates) {
    try { return JSON.parse(c) } catch { /* try next */ }
  }
  return null
}

function clampNumber(v, min, max, fallback = null) {
  const n = typeof v === 'number' ? v : parseFloat(v)
  if (!isFinite(n)) return fallback
  return Math.max(min, Math.min(max, n))
}

function buildUserPrompt({ action, robot, obstacles = [], npcs = [], history = [], chatCtx = null, goal = null, battery = null }) {
  const telem = [
    `position=(${robot.x.toFixed(2)}, ${robot.y.toFixed(2)})`,
    `heading=${(robot.heading * 180 / Math.PI).toFixed(0)}°`,
    `linear_vel=${robot.linVel.toFixed(2)} m/s`,
    `angular_vel=${robot.angVel.toFixed(2)} rad/s`,
    battery != null ? `battery=${battery.toFixed(0)}%` : null,
  ].filter(Boolean).join(', ')

  const obs = obstacles.slice(0, 12).map((o) => {
    const reach = o.reachable == null ? '' : o.reachable ? ' [line-of-sight]' : ' [occluded]'
    return `  - ${o.label} at (${o.x.toFixed(1)}, ${o.y.toFixed(1)}) dist=${o.dist?.toFixed(2) ?? '?'}m${reach}`
  }).join('\n') || '  (none nearby)'

  const nearbyNpcs = npcs.slice(0, 6).map((n) =>
    `  - ${n.name} (${n.role || 'robot'}) at (${n.x.toFixed(1)}, ${n.y.toFixed(1)})`
  ).join('\n') || '  (none)'

  const hist = history.slice(-4).map((h, i) =>
    `  ${i + 1}. ${h.source}: "${h.action}" → ${h.decision}`
  ).join('\n') || '  (none)'

  const chatLines = chatCtx ? [
    chatCtx.speaker ? `  speaker: ${chatCtx.speaker}` : null,
    chatCtx.vibe != null ? `  vibe_toward_speaker: ${chatCtx.vibe >= 0 ? '+' : ''}${chatCtx.vibe} (positive = they've earned trust)` : null,
    chatCtx.plan ? `  plan_you_committed_to: ${chatCtx.plan}` : null,
    chatCtx.tone ? `  tone: ${chatCtx.tone}` : null,
  ].filter(Boolean).join('\n') : null

  const goalLine = goal ? `\nNAMED GOAL (from action): ${goal.label} at (${goal.x.toFixed(1)}, ${goal.y.toFixed(1)}) · straight-line ${goal.dist.toFixed(1)}m, ${goal.blocked ? 'path BLOCKED by obstacles — needs planning' : 'roughly clear'}` : ''

  return `INCOMING ACTION: "${action}"${chatLines ? `

CHAT CONTEXT:
${chatLines}` : ''}${goalLine}

ROBOT TELEMETRY: ${telem}

NEARBY OBSTACLES:
${obs}

NEARBY ROBOTS:
${nearbyNpcs}

RECENT HISTORY:
${hist}

Decide now. Output only the JSON object.`
}

export async function classifyAction(ctx) {
  if (!hasAnyLLM()) {
    return regexFallback(ctx.action, 'llm-unavailable')
  }

  const { text, model, ms, error } = await chatOneShot({
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: buildUserPrompt(ctx) },
    ],
    temperature: 0.2,
    max_tokens: 300,
  })

  if (error || !text) {
    return regexFallback(ctx.action, error || 'empty-output', ms)
  }

  const parsed = extractJson(text)
  if (!parsed || !parsed.decision) {
    console.warn('[cognitiveLayer] bad json, falling back. raw:', text)
    return regexFallback(ctx.action, 'llm-bad-json', ms)
  }

  const decision = ['execute', 'modify', 'refuse'].includes(parsed.decision) ? parsed.decision : 'execute'
  const linear = decision === 'refuse' ? null : clampNumber(parsed.linear, LIN_MIN, LIN_MAX, 0)
  const angular = decision === 'refuse' ? null : clampNumber(parsed.angular, ANG_MIN, ANG_MAX, 0)
  const duration = decision === 'refuse' ? null : clampNumber(parsed.duration, DUR_MIN, DUR_MAX, 2)

  return {
    decision,
    linear,
    angular,
    duration,
    reason: String(parsed.reason || '').slice(0, 200) || '(no reason given)',
    model,
    ms,
    raw: ctx.action,
  }
}

function regexFallback(action, why, ms = 0) {
  const parsed = parseAction(action)
  if (!parsed) {
    return {
      decision: 'refuse',
      linear: null, angular: null, duration: null,
      reason: `Could not understand "${action}" (${why})`,
      model: 'regex',
      ms,
      raw: action,
    }
  }
  return {
    decision: 'execute',
    linear: clampNumber(parsed.linear ?? 0, LIN_MIN, LIN_MAX, 0),
    angular: clampNumber(parsed.angular ?? 0, ANG_MIN, ANG_MAX, 0),
    duration: clampNumber(parsed.duration ?? 2, DUR_MIN, DUR_MAX, 2),
    reason: parsed.stop ? `Emergency stop (${why})` : `Regex fallback parse (${why})`,
    model: 'regex',
    ms,
    raw: action,
  }
}
