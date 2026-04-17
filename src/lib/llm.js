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
  // Native Gemini REST — guaranteed CORS-friendly via ?key=. Primary.
  GEMINI_KEY && {
    name: 'gemini-native-2.0',
    kind: 'gemini-native',
    model: 'gemini-2.0-flash',
    key: GEMINI_KEY,
  },
  GEMINI_KEY && {
    name: 'gemini-native-1.5',
    kind: 'gemini-native',
    model: 'gemini-1.5-flash-latest',
    key: GEMINI_KEY,
  },
  // OpenAI-compat Gemini endpoint (some environments; may trip CORS)
  GEMINI_KEY && {
    name: 'gemini-openai-compat',
    kind: 'openai',
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    model: 'gemini-2.0-flash',
    key: GEMINI_KEY,
    referer: false,
  },
  OPENROUTER_KEY && {
    name: 'or-gemini-flash',
    kind: 'openai',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'google/gemini-2.0-flash-exp:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-70b',
    kind: 'openai',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'meta-llama/llama-3.3-70b-instruct:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-8b',
    kind: 'openai',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'meta-llama/llama-3.1-8b-instruct:free',
    key: OPENROUTER_KEY,
    referer: true,
  },
].filter(Boolean)

// Vision-capable chain (for image inputs)
export const VISION_PROVIDERS = [
  GEMINI_KEY && {
    name: 'gemini-vision-native',
    kind: 'gemini-native',
    model: 'gemini-2.0-flash',
    key: GEMINI_KEY,
  },
  OPENROUTER_KEY && {
    name: 'or-llama-vision',
    kind: 'openai',
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

// Convert OpenAI-style messages → native Gemini contents + systemInstruction.
// Gemini's contract: user/model roles; a single optional systemInstruction.
function messagesToGemini(messages) {
  const sys = messages.filter((m) => m.role === 'system').map((m) => (typeof m.content === 'string' ? m.content : '')).join('\n\n').trim()
  const contents = []
  for (const m of messages) {
    if (m.role === 'system') continue
    const role = m.role === 'assistant' ? 'model' : 'user'
    if (typeof m.content === 'string') {
      contents.push({ role, parts: [{ text: m.content }] })
    } else if (Array.isArray(m.content)) {
      // OpenAI multipart: [{type:'text',text},{type:'image_url',image_url:{url}}]
      const parts = []
      for (const part of m.content) {
        if (part.type === 'text') parts.push({ text: part.text })
        else if (part.type === 'image_url' && part.image_url?.url) {
          const url = part.image_url.url
          if (url.startsWith('data:')) {
            const m2 = url.match(/^data:([^;]+);base64,(.+)$/)
            if (m2) parts.push({ inline_data: { mime_type: m2[1], data: m2[2] } })
          }
        }
      }
      contents.push({ role, parts })
    }
  }
  return { contents, systemInstruction: sys ? { parts: [{ text: sys }] } : undefined }
}

// Native Gemini one-shot. Returns { text, error }.
async function geminiNativeOneShot(p, { messages, temperature, max_tokens, signal }) {
  const { contents, systemInstruction } = messagesToGemini(messages)
  const body = {
    contents,
    ...(systemInstruction ? { systemInstruction } : {}),
    generationConfig: { temperature, maxOutputTokens: max_tokens },
  }
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${p.model}:generateContent?key=${encodeURIComponent(p.key)}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) return { text: '', error: `status_${res.status}` }
  const json = await res.json()
  const text = json.candidates?.[0]?.content?.parts?.map((pp) => pp.text).filter(Boolean).join('') || ''
  return { text, error: null }
}

// Native Gemini streaming. Server-Sent-Events via ?alt=sse. Emits deltas.
async function geminiNativeStream(p, { messages, temperature, max_tokens, onDelta, onFirstToken, signal }) {
  const { contents, systemInstruction } = messagesToGemini(messages)
  const body = {
    contents,
    ...(systemInstruction ? { systemInstruction } : {}),
    generationConfig: { temperature, maxOutputTokens: max_tokens },
  }
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${p.model}:streamGenerateContent?alt=sse&key=${encodeURIComponent(p.key)}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) return { text: '', error: `status_${res.status}` }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let text = ''
  let firedFirst = false
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // Split on blank lines — SSE event boundary
    const events = buffer.split(/\r?\n\r?\n/)
    buffer = events.pop() || ''
    for (const ev of events) {
      for (const line of ev.split(/\r?\n/)) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)
        if (data === '[DONE]') continue
        try {
          const parsed = JSON.parse(data)
          const delta = parsed.candidates?.[0]?.content?.parts?.map((pp) => pp.text).filter(Boolean).join('') || ''
          if (delta) {
            if (!firedFirst) { firedFirst = true; onFirstToken?.() }
            text += delta
            onDelta?.(delta, text)
          }
        } catch { /* ignore */ }
      }
    }
  }
  return { text, error: null }
}

// One-shot call — returns { text, model, ms, error }
export async function chatOneShot({ messages, temperature = 0.7, max_tokens = 600, vision = false, signal }) {
  const chain = vision ? VISION_PROVIDERS : PROVIDERS
  const started = performance.now()
  for (const p of chain) {
    try {
      let text = '', error = null
      if (p.kind === 'gemini-native') {
        ({ text, error } = await geminiNativeOneShot(p, { messages, temperature, max_tokens, signal }))
      } else {
        const res = await fetch(p.url, {
          method: 'POST',
          headers: buildHeaders(p),
          body: JSON.stringify({ model: p.model, messages, temperature, max_tokens }),
          signal,
        })
        if (!res.ok) { error = `status_${res.status}` }
        else {
          const json = await res.json()
          text = json.choices?.[0]?.message?.content || ''
        }
      }
      if (error) { console.warn(`[llm] ${p.name} ${error}; next`); continue }
      if (text.trim().length === 0) { console.warn(`[llm] ${p.name} empty; next`); continue }
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
      let text = '', error = null

      if (p.kind === 'gemini-native') {
        ({ text, error } = await geminiNativeStream(p, { messages, temperature, max_tokens, onDelta, onFirstToken, signal }))
      } else {
        const res = await fetch(p.url, {
          method: 'POST',
          headers: buildHeaders(p),
          body: JSON.stringify({ model: p.model, messages, temperature, max_tokens, stream: true }),
          signal,
        })
        if (!res.ok) {
          error = `status_${res.status}`
        } else {
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
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
                  if (!firedFirst) { firedFirst = true; onFirstToken?.() }
                  text += delta
                  onDelta?.(delta, text)
                }
              } catch { /* ignore */ }
            }
          }
        }
      }

      if (error) { console.warn(`[llm/stream] ${p.name} ${error}; next`); continue }
      if (text.trim().length > 0) return { text, model: p.name, error: null }
      console.warn(`[llm/stream] ${p.name} empty stream; next`)
    } catch (err) {
      console.warn(`[llm/stream] ${p.name} threw: ${err.message}`)
    }
  }
  return { text: '', model: 'none', error: 'all_providers_failed' }
}

export const hasAnyLLM = () => PROVIDERS.length > 0
