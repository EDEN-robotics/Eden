// LLM-powered action classifier. Routes every incoming action (from chat
// or manual input) through EDEN's Cognitive Layer, which decides whether
// to execute, modify, or refuse — and emits a reason.
//
// Returns: { decision: 'execute' | 'modify' | 'refuse',
//            linear, angular, duration, reason, model, ms, raw }
// `linear`/`angular`/`duration` are null when decision === 'refuse'.

import { parseAction } from './simBridge'

const OPENROUTER_KEY = (import.meta.env.VITE_OPENROUTER_API_KEY || '').trim()
// Small & fast for a decision loop. Free-tier OpenRouter.
const CLASSIFIER_MODEL = 'google/gemini-2.0-flash-exp:free'
const FALLBACK_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'

const SYSTEM = `You are the Cognitive Layer of EDEN, a humanoid robotics architecture. You are evaluating an incoming action request before it reaches the Action Layer.

EDEN's values: be helpful, be safe, be curious. Prefer action over inaction when the request is reasonable. Refuse only when the action is clearly unsafe, ambiguous, or contradicts prior context.

You will receive:
- the raw action command
- the robot's current telemetry (position, heading, velocity)
- the list of obstacles and NPC robots nearby
- optionally recent action history

Output a JSON object with EXACTLY these keys:
{
  "decision": "execute" | "modify" | "refuse",
  "linear": <m/s, in [-0.6, 0.6]> or null,
  "angular": <rad/s, in [-1.5, 1.5]> or null,
  "duration": <seconds, in [0.1, 10]> or null,
  "reason": "<one short sentence explaining>"
}

- execute: do exactly what was asked
- modify: do something close to what was asked (e.g. slower, shorter) with a short reason why you adjusted
- refuse: return null velocities and explain briefly

ONLY output the JSON. No prose, no markdown fence.`

function extractJson(text) {
  if (!text) return null
  // Strip markdown fences if present
  const cleaned = text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim()
  // Find first { ... last }
  const first = cleaned.indexOf('{')
  const last = cleaned.lastIndexOf('}')
  if (first < 0 || last < 0) return null
  try { return JSON.parse(cleaned.slice(first, last + 1)) } catch { return null }
}

function clampNumber(v, min, max, fallback = null) {
  const n = typeof v === 'number' ? v : parseFloat(v)
  if (!isFinite(n)) return fallback
  return Math.max(min, Math.min(max, n))
}

function buildUserPrompt({ action, robot, obstacles = [], npcs = [], history = [] }) {
  const telem = [
    `position=(${robot.x.toFixed(2)}, ${robot.y.toFixed(2)})`,
    `heading=${(robot.heading * 180 / Math.PI).toFixed(0)}°`,
    `linear_vel=${robot.linVel.toFixed(2)} m/s`,
    `angular_vel=${robot.angVel.toFixed(2)} rad/s`,
  ].join(', ')

  const obs = obstacles.slice(0, 12).map((o) =>
    `  - ${o.label} at (${o.x.toFixed(1)}, ${o.y.toFixed(1)})  size=${o.w.toFixed(1)}×${o.h.toFixed(1)}`
  ).join('\n') || '  (none nearby)'

  const nearbyNpcs = npcs.slice(0, 6).map((n) =>
    `  - ${n.name} at (${n.x.toFixed(1)}, ${n.y.toFixed(1)})`
  ).join('\n') || '  (none)'

  const hist = history.slice(-4).map((h, i) =>
    `  ${i + 1}. ${h.source}: "${h.action}" → ${h.decision}`
  ).join('\n') || '  (none)'

  return `INCOMING ACTION: "${action}"

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
  if (!OPENROUTER_KEY) {
    // No LLM available — fall straight through to regex
    return regexFallback(ctx.action, 'llm-unavailable')
  }

  const body = {
    model: CLASSIFIER_MODEL,
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: buildUserPrompt(ctx) },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.2,
  }

  const started = performance.now()
  try {
    let res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_KEY}`,
        'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : '',
        'X-Title': 'EDEN Cognitive Layer',
      },
      body: JSON.stringify(body),
    })

    // Some free models don't accept response_format — retry without it
    if (!res.ok && res.status === 400) {
      const { response_format, ...noJson } = body
      res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENROUTER_KEY}`,
        },
        body: JSON.stringify({ ...noJson, model: FALLBACK_MODEL }),
      })
    }

    if (!res.ok) {
      console.warn('[cognitiveLayer] classifier non-ok:', res.status)
      return regexFallback(ctx.action, `llm-http-${res.status}`)
    }

    const json = await res.json()
    const text = json.choices?.[0]?.message?.content || ''
    const parsed = extractJson(text)
    const ms = Math.round(performance.now() - started)

    if (!parsed || !parsed.decision) {
      console.warn('[cognitiveLayer] bad json, falling back. raw:', text)
      return regexFallback(ctx.action, 'llm-bad-json', ms)
    }

    const decision = ['execute', 'modify', 'refuse'].includes(parsed.decision) ? parsed.decision : 'execute'
    const linear = decision === 'refuse' ? null : clampNumber(parsed.linear, -0.6, 0.6, 0)
    const angular = decision === 'refuse' ? null : clampNumber(parsed.angular, -1.5, 1.5, 0)
    const duration = decision === 'refuse' ? null : clampNumber(parsed.duration, 0.1, 10, 2)

    return {
      decision,
      linear,
      angular,
      duration,
      reason: String(parsed.reason || '').slice(0, 200) || '(no reason given)',
      model: CLASSIFIER_MODEL,
      ms,
      raw: ctx.action,
    }
  } catch (err) {
    console.warn('[cognitiveLayer] classifier threw:', err)
    return regexFallback(ctx.action, 'llm-threw', Math.round(performance.now() - started))
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
    decision: parsed.stop ? 'execute' : 'execute',
    linear: parsed.linear ?? 0,
    angular: parsed.angular ?? 0,
    duration: parsed.duration ?? 2,
    reason: `Regex fallback parse (${why})`,
    model: 'regex',
    ms,
    raw: action,
  }
}
