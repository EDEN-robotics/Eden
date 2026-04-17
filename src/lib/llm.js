// Unified LLM client. Tries a chain of providers in order, returns as soon as
// one produces non-empty output. Supports both streaming and one-shot calls.
//
// Chain (hard-coded for now — can be made configurable later):
//   1. Google Gemini direct (OpenAI-compatible endpoint) — fast, generous free tier
//   2. OpenRouter Gemini Flash free — same model family via OR
//   3. OpenRouter Llama-3.3-70B free
//   4. OpenRouter Llama-3.1-8B free
//
// All providers speak the OpenAI chat-completions protocol so the streaming
// parser below works against every one of them.

const GEMINI_KEY = (import.meta.env.VITE_GEMINI_API_KEY || '').trim()
const OPENROUTER_KEY = (import.meta.env.VITE_OPENROUTER_API_KEY || '').trim()

export const PROVIDERS = [
  GEMINI_KEY && {
    name: 'gemini-direct',
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    model: 'gemini-2.0-flash',
    key: GEMINI_KEY,
    referer: false,
  },
  GEMINI_KEY && {
    name: 'gemini-direct-1.5',
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    model: 'gemini-1.5-flash-latest',
    key: GEMINI_KEY,
    referer: false,
  },
  OPENROUTER_KEY && {
    name: 'or-gemini-flash',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'google/gemini-2.0-flash-exp:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-70b',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'meta-llama/llama-3.3-70b-instruct:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-8b',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'meta-llama/llama-3.1-8b-instruct:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
].filter(Boolean)

// Vision-capable chain (for image inputs)
export const VISION_PROVIDERS = [
  GEMINI_KEY && {
    name: 'gemini-vision',
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    model: 'gemini-2.0-flash',
    key: GEMINI_KEY,
    referer: false,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-vision',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'meta-llama/llama-3.2-90b-vision-instruct:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
].filter(Boolean)

function buildHeaders(p) {
  const h = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${p.key}`,
  }
  if (p.referer && typeof window !== 'undefined') {
    h['HTTP-Referer'] = window.location.origin
    h['X-Title'] = 'EDEN Robotics'
  }
  return h
}

// One-shot call — returns { text, model, ms, error }
export async function chatOneShot({ messages, temperature = 0.7, max_tokens = 600, vision = false, signal }) {
  const chain = vision ? VISION_PROVIDERS : PROVIDERS
  const started = performance.now()
  for (const p of chain) {
    try {
      const res = await fetch(p.url, {
        method: 'POST',
        headers: buildHeaders(p),
        body: JSON.stringify({ model: p.model, messages, temperature, max_tokens }),
        signal,
      })
      if (!res.ok) {
        console.warn(`[llm] ${p.name} status ${res.status}; next`)
        continue
      }
      const json = await res.json()
      const text = json.choices?.[0]?.message?.content || ''
      if (text.trim().length === 0) {
        console.warn(`[llm] ${p.name} empty; next`)
        continue
      }
      return { text, model: p.name, ms: Math.round(performance.now() - started), error: null }
    } catch (err) {
      console.warn(`[llm] ${p.name} threw: ${err.message}`)
    }
  }
  return { text: '', model: 'none', ms: Math.round(performance.now() - started), error: 'all_providers_failed' }
}

// Streaming call — yields content deltas via onDelta. Falls through to the
// next provider if the current one returns an empty stream. Returns the final
// assembled text + provider that actually served it.
export async function chatStreaming({ messages, temperature = 0.85, max_tokens = 600, vision = false, onDelta, onFirstToken, signal }) {
  const chain = vision ? VISION_PROVIDERS : PROVIDERS
  for (const p of chain) {
    try {
      const res = await fetch(p.url, {
        method: 'POST',
        headers: buildHeaders(p),
        body: JSON.stringify({ model: p.model, messages, temperature, max_tokens, stream: true }),
        signal,
      })
      if (!res.ok) {
        console.warn(`[llm/stream] ${p.name} status ${res.status}; next`)
        continue
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let text = ''
      let firedFirst = false
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        for (const line of chunk.split('\n').filter((l) => l.startsWith('data: '))) {
          const data = line.slice(6)
          if (data === '[DONE]') continue
          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta?.content
            if (delta) {
              if (!firedFirst) {
                firedFirst = true
                onFirstToken?.()
              }
              text += delta
              onDelta?.(delta, text)
            }
          } catch { /* ignore */ }
        }
      }
      if (text.trim().length > 0) {
        return { text, model: p.name, error: null }
      }
      console.warn(`[llm/stream] ${p.name} empty stream; next`)
    } catch (err) {
      console.warn(`[llm/stream] ${p.name} threw: ${err.message}`)
    }
  }
  return { text: '', model: 'none', error: 'all_providers_failed' }
}

export const hasAnyLLM = () => PROVIDERS.length > 0
