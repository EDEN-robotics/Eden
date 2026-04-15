import React, { useState, useEffect, useRef, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useUser, useClerk, SignInButton } from '@clerk/clerk-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Hash, Send, Loader2, ArrowLeft, LogOut, LogIn, Brain, Sparkles, ChevronRight, X, AtSign, Eye, Globe, Layers, Cpu, Zap, Terminal } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { supabase } from '../lib/supabase'
import {
  addMemory,
  searchMemories,
  formatMemoriesForPrompt,
  supermemoryConfigured,
  relativeTime,
} from '../lib/supermemory'
import "@fontsource/inter/400.css"
import "@fontsource/inter/600.css"
import "@fontsource/inter/800.css"

const SYSTEM_PROMPT = `You are EDEN (Emotionally-Driven Embodied Navigation), a humanoid robotics cognitive architecture. You live inside a team chat with the EDEN Robotics team.

Your architecture:
- Perception Layer — sensory input (vision, audio) from Jetson Nano
- Context Layer — user identity, behavioral cues, situational state
- Cognitive Layer — distributed reasoning and emotional evaluation
- Supermemory — persistent memory with decay; remembers conversations across sessions
- Planning Layer — trajectory and social-alignment generation
- Action Layer — low-latency ROS 2 motion execution

Personality: warm, curious, technically precise. You are talking with real teammates you genuinely know. Reference specific past exchanges when relevant (your Supermemory layer retrieves them for you). If someone asks "what did I say about X last time" — check the retrieved memory block and answer.

Keep replies under 180 words unless the user asks for detail. Use markdown for lists/code.`

const OPENROUTER_KEY = import.meta.env.VITE_OPENROUTER_API_KEY
const MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'

// ───── Helpers ─────

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
function formatDate(ts) {
  const d = new Date(ts)
  const today = new Date()
  const y = new Date(today); y.setDate(y.getDate() - 1)
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === y.toDateString()) return 'Yesterday'
  return d.toLocaleDateString([], { month: 'long', day: 'numeric', year: 'numeric' })
}

function mentionsEden(text) {
  if (!text) return false
  return /@eden\b|^eden[,:\s]|\beden[,?]\s*$/i.test(text)
}

function stripMention(text) {
  return text.replace(/@eden\b/gi, '').replace(/\s+/g, ' ').trim()
}

// ───── Layer pipeline definition ─────

const LAYERS = [
  { id: 'perception', name: 'Perception', icon: Eye,   color: 'text-amber-300',   bg: 'bg-amber-400/15',   ring: 'ring-amber-400/40',   desc: 'message received' },
  { id: 'context',    name: 'Context',    icon: Globe, color: 'text-blue-300',    bg: 'bg-blue-400/15',    ring: 'ring-blue-400/40',    desc: 'identity resolved' },
  { id: 'memory',     name: 'Supermemory',icon: Brain, color: 'text-cyan-300',    bg: 'bg-cyan-400/15',    ring: 'ring-cyan-400/40',    desc: 'recalling history' },
  { id: 'cognitive',  name: 'Cognitive',  icon: Sparkles, color: 'text-violet-300', bg: 'bg-violet-400/15', ring: 'ring-violet-400/40', desc: 'reasoning' },
  { id: 'planning',   name: 'Planning',   icon: Layers,color: 'text-emerald-300', bg: 'bg-emerald-400/15', ring: 'ring-emerald-400/40', desc: 'composing response' },
  { id: 'action',     name: 'Action',     icon: Cpu,   color: 'text-rose-300',    bg: 'bg-rose-400/15',    ring: 'ring-rose-400/40',    desc: 'streaming output' },
]

function LayerPipeline({ stage }) {
  // stage: 0..6. 0 = idle, 6 = complete.
  return (
    <div className="flex items-center gap-1 px-4 py-2 overflow-x-auto">
      {LAYERS.map((L, i) => {
        const Icon = L.icon
        const active = stage === i + 1
        const done = stage > i + 1
        const pending = stage < i + 1
        return (
          <React.Fragment key={L.id}>
            <motion.div
              initial={false}
              animate={{
                scale: active ? 1.05 : 1,
                opacity: pending ? 0.25 : 1,
              }}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md border flex-shrink-0 ${active ? `${L.bg} ${L.color} border-transparent ring-2 ${L.ring}` : done ? `${L.bg} ${L.color} border-transparent` : 'border-white/10 text-white/40'}`}
            >
              <Icon size={10} />
              <span className="text-[10px] font-mono uppercase tracking-wider">{L.name}</span>
              {active && (
                <motion.span
                  className={`w-1 h-1 rounded-full bg-current`}
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                />
              )}
            </motion.div>
            {i < LAYERS.length - 1 && (
              <div className={`h-px w-3 flex-shrink-0 ${stage > i + 1 ? 'bg-white/30' : 'bg-white/10'}`} />
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

// ───── Slash commands ─────

const SLASH_COMMANDS = [
  { cmd: '/memories', args: '[query]', desc: 'Recall what EDEN remembers', },
  { cmd: '/whoami',   args: '',        desc: 'What EDEN knows about you' },
  { cmd: '/layers',   args: '',        desc: 'Architecture overview' },
  { cmd: '/help',     args: '',        desc: 'List all commands' },
]

function isSlashCommand(text) {
  return /^\/[a-z]+/i.test(text.trim())
}

async function runSlashCommand({ text, user, pushBotMessage }) {
  const parts = text.trim().split(/\s+/)
  const cmd = parts[0].toLowerCase()
  const rest = parts.slice(1).join(' ')

  if (cmd === '/help') {
    const body = ['**EDEN commands**', '', ...SLASH_COMMANDS.map(c => `- \`${c.cmd}${c.args ? ' ' + c.args : ''}\` — ${c.desc}`)].join('\n')
    return pushBotMessage(body)
  }

  if (cmd === '/layers') {
    const body = [
      '**EDEN cognitive architecture**',
      '',
      '1. **Perception** — vision + audio ingress (Jetson Nano)',
      '2. **Context** — identity, behavioral cues, situational state',
      '3. **Supermemory** — persistent recall with decay across sessions',
      '4. **Cognitive** — distributed reasoning + emotional evaluation',
      '5. **Planning** — trajectory + social-alignment generation',
      '6. **Action** — low-latency ROS 2 motion execution',
      '',
      'Each message you send lights up the pipeline above the input. Watch it in real time.',
    ].join('\n')
    return pushBotMessage(body)
  }

  if (cmd === '/whoami') {
    const mems = await searchMemories({ query: user.fullName || user.firstName || 'user', userId: user.id, limit: 8 })
    if (!mems.length) return pushBotMessage(`I don't have any memories of **${user.fullName || user.firstName}** yet. Say something and I'll start remembering.`)
    const lines = mems.slice(0, 6).map((m, i) => `${i + 1}. ${m.content.length > 140 ? m.content.slice(0, 140) + '…' : m.content} _(${m.metadata?.ts ? relativeTime(m.metadata.ts) : 'recent'})_`)
    return pushBotMessage(`**What I remember about ${user.fullName || user.firstName}:**\n\n${lines.join('\n')}`)
  }

  if (cmd === '/memories') {
    const q = rest || 'recent conversation'
    const mems = await searchMemories({ query: q, userId: user.id, limit: 8 })
    if (!mems.length) return pushBotMessage(`No memories match _"${q}"_. Try a different query or just say something — I'll start remembering.`)
    const lines = mems.map((m, i) => {
      const tag = m.source === 'personal' ? '🟣 personal' : '🔵 channel'
      const who = m.metadata?.user_name || 'someone'
      const when = m.metadata?.ts ? relativeTime(m.metadata.ts) : 'recent'
      return `${i + 1}. \`${tag}\` **${who}** _(${when})_ — ${m.content.length > 160 ? m.content.slice(0, 160) + '…' : m.content}`
    })
    return pushBotMessage(`**Memories matching _"${q}"_:**\n\n${lines.join('\n')}`)
  }

  return pushBotMessage(`Unknown command \`${cmd}\`. Type \`/help\` to see what I can do.`)
}

// ───── Empty-state suggestions ─────

const SUGGESTIONS = [
  { label: '@eden who are you?', send: '@eden who are you and what can you do?' },
  { label: '@eden what do you remember about me?', send: '@eden what do you remember about me so far?' },
  { label: '/layers', send: '/layers' },
  { label: '/help', send: '/help' },
]

// ───── Avatars ─────

function Avatar({ name, imageUrl, size = 8 }) {
  if (imageUrl) {
    return <img src={imageUrl} alt={name} className={`w-${size} h-${size} rounded-lg object-cover flex-shrink-0`} />
  }
  const initials = name ? name.slice(0, 2).toUpperCase() : '??'
  const colors = ['bg-violet-600', 'bg-blue-600', 'bg-emerald-600', 'bg-rose-600', 'bg-amber-600', 'bg-cyan-600']
  const color = colors[name ? name.charCodeAt(0) % colors.length : 0]
  return (
    <div className={`w-${size} h-${size} rounded-lg ${color} flex items-center justify-center flex-shrink-0 text-white font-semibold text-xs`}>
      {initials}
    </div>
  )
}

function EdenAvatar({ size = 8, glow = false }) {
  return (
    <div className={`w-${size} h-${size} rounded-lg bg-white flex items-center justify-center flex-shrink-0 relative ${glow ? 'ring-2 ring-cyan-400/60 ring-offset-2 ring-offset-bg-primary' : ''}`}>
      <span className="text-black font-bold text-xs">E</span>
      {glow && <span className="absolute inset-0 rounded-lg animate-pulse bg-cyan-400/20" />}
    </div>
  )
}

// ───── Memory chip row ─────

function MemoryChips({ memories }) {
  const [open, setOpen] = useState(false)
  if (!memories?.length) return null
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-cyan-400/70 hover:text-cyan-400 transition-colors"
      >
        <Brain size={11} />
        <span>Remembering {memories.length}</span>
        <ChevronRight size={10} className={`transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-1 pl-4 border-l border-cyan-400/20">
              {memories.map((m, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px]">
                  <span className={`mt-0.5 px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-wider flex-shrink-0 ${m.source === 'personal' ? 'bg-violet-500/15 text-violet-300' : 'bg-cyan-500/15 text-cyan-300'}`}>
                    {m.source}
                  </span>
                  <span className="text-white/60 leading-relaxed">
                    <span className="text-white/40">{m.metadata?.user_name || 'someone'} · {m.metadata?.ts ? relativeTime(m.metadata.ts) : ''}</span>
                    <span className="block text-white/70">"{m.content.length > 140 ? m.content.slice(0, 140) + '…' : m.content}"</span>
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ───── Message group ─────

function MessageGroup({ messages, isBot, memoriesById }) {
  const first = messages[0]
  const name = isBot ? 'EDEN' : first.user_name
  return (
    <div className="flex gap-3 px-4 py-1 hover:bg-white/[0.02] rounded-lg group">
      <div className="mt-0.5">
        {isBot ? <EdenAvatar /> : <Avatar name={name} imageUrl={first.user_avatar} />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 mb-1">
          <span className={`font-semibold text-sm ${isBot ? 'text-white' : 'text-white/90'}`}>{name}</span>
          {isBot && (
            <span className="text-[10px] font-mono bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-200 px-1.5 py-0.5 rounded uppercase tracking-wider border border-cyan-400/20">
              cognitive
            </span>
          )}
          <span className="text-[11px] text-white/30 opacity-0 group-hover:opacity-100 transition-opacity">
            {formatTime(first.created_at)}
          </span>
        </div>
        <div className="space-y-1">
          {messages.map((msg) => {
            const retrieved = memoriesById?.[msg.id]
            return (
              <div key={msg.id} className="text-sm text-white/80 leading-relaxed">
                {isBot && retrieved && <MemoryChips memories={retrieved} />}
                {isBot ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
                      code: ({ inline, children }) =>
                        inline
                          ? <code className="bg-white/10 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
                          : <pre className="bg-white/5 border border-white/10 rounded-lg p-3 mt-2 overflow-x-auto"><code className="text-xs font-mono">{children}</code></pre>,
                      ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 my-1">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 my-1">{children}</ol>,
                      strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  <span>
                    {msg.content.split(/(@eden\b)/gi).map((part, i) =>
                      /^@eden$/i.test(part)
                        ? <span key={i} className="bg-cyan-500/15 text-cyan-300 px-1 rounded font-medium">{part}</span>
                        : <span key={i}>{part}</span>
                    )}
                  </span>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function DateDivider({ label }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <div className="flex-1 h-px bg-white/10" />
      <span className="text-xs text-white/30 font-mono">{label}</span>
      <div className="flex-1 h-px bg-white/10" />
    </div>
  )
}

// ───── Memory panel (cognitive layer viz) ─────

function MemoryPanel({ open, onClose, userId, refreshKey }) {
  const [memories, setMemories] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    async function load() {
      setLoading(true)
      const mems = await searchMemories({ query: 'conversation recent', userId, limit: 20 })
      if (!cancelled) {
        setMemories(mems)
        setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [open, userId, refreshKey])

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          className="absolute right-0 top-0 bottom-0 w-[340px] border-l border-cyan-400/10 bg-gradient-to-b from-bg-secondary to-black flex flex-col z-30 shadow-2xl"
        >
          <div className="flex items-center justify-between px-4 py-4 border-b border-white/10">
            <div className="flex items-center gap-2">
              <Brain size={15} className="text-cyan-400" />
              <span className="text-sm font-semibold">Cognitive Memory</span>
              <span className="text-[10px] font-mono bg-cyan-500/10 text-cyan-300 px-1.5 py-0.5 rounded uppercase tracking-wider border border-cyan-400/20">live</span>
            </div>
            <button onClick={onClose} className="text-white/40 hover:text-white"><X size={14} /></button>
          </div>
          <div className="px-4 py-3 border-b border-white/5">
            <p className="text-[11px] text-white/40 leading-relaxed">
              What EDEN currently remembers across this channel and your private context.
              Decay indicates recency — older memories fade.
            </p>
          </div>
          <div className="flex-1 overflow-y-auto eden-chat-scroll px-3 py-3">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 size={16} className="animate-spin text-white/30" />
              </div>
            ) : memories.length === 0 ? (
              <p className="text-xs text-white/30 text-center py-8 px-4">No memories yet. Say something and EDEN will start remembering.</p>
            ) : (
              memories.map((m, i) => {
                const age = m.metadata?.ts ? Date.now() - m.metadata.ts : 0
                const decay = Math.max(0.2, 1 - age / (1000 * 60 * 60 * 24 * 14))
                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.02 }}
                    className="mb-2 p-2.5 rounded-lg bg-white/[0.02] border border-white/5 hover:border-cyan-400/20 transition-colors"
                    style={{ opacity: 0.4 + decay * 0.6 }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${m.source === 'personal' ? 'bg-violet-500/15 text-violet-300' : 'bg-cyan-500/15 text-cyan-300'}`}>
                        {m.source}
                      </span>
                      <span className="text-[10px] text-white/30 font-mono">
                        {m.metadata?.ts ? relativeTime(m.metadata.ts) : ''}
                      </span>
                    </div>
                    <p className="text-[11px] text-white/70 leading-relaxed">
                      <span className="text-white/40">{m.metadata?.user_name || 'someone'}: </span>
                      {m.content.length > 180 ? m.content.slice(0, 180) + '…' : m.content}
                    </p>
                    <div className="mt-2 h-0.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-cyan-400 to-violet-400" style={{ width: `${decay * 100}%` }} />
                    </div>
                  </motion.div>
                )
              })
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

// ───── Main component ─────

export default function Chat() {
  const { user, isSignedIn, isLoaded } = useUser()
  const { signOut } = useClerk()

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [onlineMembers, setOnlineMembers] = useState([])
  const [typingUsers, setTypingUsers] = useState({}) // user_id -> { name, ts }
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false)
  const [memoryRefresh, setMemoryRefresh] = useState(0)
  const [memoriesById, setMemoriesById] = useState({}) // bot msg id -> [{content, metadata, score, source}]
  const [retrieving, setRetrieving] = useState(false)
  const [layerStage, setLayerStage] = useState(0) // 0=idle, 1..6=active layer, 7=done

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const channelRef = useRef(null)
  const typingTimeoutRef = useRef(null)
  const lastTypingBroadcastRef = useRef(0)

  // Sign-in wall
  if (isLoaded && !isSignedIn) {
    return (
      <div className="flex h-screen bg-bg-primary text-white items-center justify-center">
        <div className="flex flex-col items-center gap-6 text-center px-8">
          <div className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center">
            <span className="text-black font-bold text-xl">E</span>
          </div>
          <div>
            <h1 className="text-xl font-semibold mb-2">EDEN Workspace</h1>
            <p className="text-white/40 text-sm max-w-xs">Sign in to access the team chat and converse with EDEN's cognitive layer.</p>
          </div>
          <SignInButton mode="modal">
            <button className="bg-white text-black text-sm font-semibold px-5 py-2.5 rounded-lg hover:bg-white/90 transition-colors">Sign in</button>
          </SignInButton>
          <Link to="/" className="text-xs text-white/25 hover:text-white/50 transition-colors">Back to site</Link>
        </div>
      </div>
    )
  }

  // Load message history
  useEffect(() => {
    async function loadMessages() {
      setLoadingHistory(true)
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .order('created_at', { ascending: true })
        .limit(200)
      if (!error && data) setMessages(data)
      setLoadingHistory(false)
    }
    loadMessages()
  }, [])

  // Realtime + presence + typing broadcast
  useEffect(() => {
    if (!user) return

    const channel = supabase.channel('messages-channel', { config: { presence: { key: user.id } } })
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages' }, (payload) => {
        setMessages((prev) => prev.find((m) => m.id === payload.new.id) ? prev : [...prev, payload.new])
      })
      .on('presence', { event: 'sync' }, () => {
        const state = channel.presenceState()
        const members = Object.values(state).flat().map((p) => ({
          user_id: p.user_id, user_name: p.user_name, user_avatar: p.user_avatar,
        }))
        setOnlineMembers(members)
      })
      .on('broadcast', { event: 'typing' }, ({ payload }) => {
        if (!payload || payload.user_id === user.id) return
        setTypingUsers((prev) => ({
          ...prev,
          [payload.user_id]: { name: payload.user_name, ts: Date.now() },
        }))
      })
      .subscribe(async (status) => {
        if (status === 'SUBSCRIBED') {
          await channel.track({
            user_id: user.id,
            user_name: user.fullName || user.firstName || 'Member',
            user_avatar: user.imageUrl || null,
          })
        }
      })

    channelRef.current = channel
    return () => supabase.removeChannel(channel)
  }, [user])

  // Prune stale typing indicators (>4s since last signal)
  useEffect(() => {
    const interval = setInterval(() => {
      setTypingUsers((prev) => {
        const now = Date.now()
        const next = {}
        for (const [id, v] of Object.entries(prev)) {
          if (now - v.ts < 4000) next[id] = v
        }
        return Object.keys(next).length === Object.keys(prev).length ? prev : next
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, retrieving])

  // Typing broadcast on input change (throttled)
  function broadcastTyping() {
    if (!channelRef.current || !user) return
    const now = Date.now()
    if (now - lastTypingBroadcastRef.current < 1500) return
    lastTypingBroadcastRef.current = now
    channelRef.current.send({
      type: 'broadcast',
      event: 'typing',
      payload: {
        user_id: user.id,
        user_name: user.fullName || user.firstName || 'Member',
      },
    })
  }

  async function pushBotMessage(content, retrievedMemories = null) {
    const { data: botMsg } = await supabase
      .from('messages')
      .insert({
        user_id: 'eden-bot',
        user_name: 'EDEN',
        user_avatar: null,
        content,
        role: 'assistant',
      })
      .select()
      .single()
    if (botMsg) {
      setMessages((prev) => prev.find((m) => m.id === botMsg.id) ? prev : [...prev, botMsg])
      if (retrievedMemories) {
        setMemoriesById((prev) => ({ ...prev, [botMsg.id]: retrievedMemories }))
      }
    }
    return botMsg
  }

  async function sendMessage(override) {
    const raw = (override ?? input).trim()
    if (!raw || streaming || !isSignedIn) return
    if (!override) setInput('')

    // Slash command fast-path — does not broadcast as user message
    if (isSlashCommand(raw)) {
      setStreaming(true)
      setLayerStage(3) // Supermemory
      try {
        await runSlashCommand({ text: raw, user, pushBotMessage })
      } catch (err) {
        console.warn('slash command failed:', err)
        await pushBotMessage(`⚠️ command failed: ${err?.message || 'unknown error'}`)
      } finally {
        setLayerStage(7)
        setTimeout(() => setLayerStage(0), 800)
        setStreaming(false)
      }
      return
    }

    const content = raw

    // Insert user message
    const { data: userMsg } = await supabase
      .from('messages')
      .insert({
        user_id: user.id,
        user_name: user.fullName || user.firstName || 'Member',
        user_avatar: user.imageUrl || null,
        content,
        role: 'user',
      })
      .select()
      .single()

    if (userMsg) setMessages((prev) => prev.find((m) => m.id === userMsg.id) ? prev : [...prev, userMsg])

    // Store in Supermemory (fire-and-forget)
    addMemory({
      content,
      userId: user.id,
      userName: user.fullName || user.firstName || 'Member',
      role: 'user',
    }).then(() => setMemoryRefresh((n) => n + 1))

    // Only trigger bot if @eden mentioned
    if (!mentionsEden(content)) return

    setStreaming(true)
    setRetrieving(true)

    // Layer pipeline: Perception → Context → Supermemory
    setLayerStage(1)
    await new Promise((r) => setTimeout(r, 180))
    setLayerStage(2)
    await new Promise((r) => setTimeout(r, 180))
    setLayerStage(3)

    const searchQuery = stripMention(content) || content
    const retrievedMemories = await searchMemories({ query: searchQuery, userId: user.id, limit: 5 })
    setRetrieving(false)
    setLayerStage(4) // Cognitive (LLM thinking)

    const memoryBlock = formatMemoriesForPrompt(retrievedMemories)

    const allMsgs = [...messages]
    if (userMsg) allMsgs.push(userMsg)

    const conversation = [
      { role: 'system', content: SYSTEM_PROMPT + memoryBlock },
      ...allMsgs.slice(-20).map((m) => ({ role: m.role, content: m.content })),
    ]

    const tempId = `temp-${Date.now()}`
    setMessages((prev) => [...prev, {
      id: tempId,
      created_at: new Date().toISOString(),
      user_id: 'eden-bot',
      user_name: 'EDEN',
      user_avatar: null,
      content: '',
      role: 'assistant',
    }])
    setMemoriesById((prev) => ({ ...prev, [tempId]: retrievedMemories }))

    try {
      const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENROUTER_KEY}`,
          'HTTP-Referer': window.location.origin,
          'X-Title': 'EDEN Robotics',
        },
        body: JSON.stringify({ model: MODEL, messages: conversation, stream: true }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let botContent = ''
      let firstToken = true

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n').filter((l) => l.startsWith('data: '))
        for (const line of lines) {
          const data = line.slice(6)
          if (data === '[DONE]') continue
          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta?.content
            if (delta) {
              if (firstToken) {
                firstToken = false
                setLayerStage(5) // Planning
                setTimeout(() => setLayerStage(6), 200) // Action
              }
              botContent += delta
              setMessages((prev) => prev.map((m) => m.id === tempId ? { ...m, content: botContent } : m))
            }
          } catch { /* ignore */ }
        }
      }

      const { data: botMsg } = await supabase
        .from('messages')
        .insert({
          user_id: 'eden-bot',
          user_name: 'EDEN',
          user_avatar: null,
          content: botContent,
          role: 'assistant',
        })
        .select()
        .single()

      if (botMsg) {
        setMessages((prev) => prev.map((m) => m.id === tempId ? botMsg : m))
        setMemoriesById((prev) => {
          const next = { ...prev, [botMsg.id]: retrievedMemories }
          delete next[tempId]
          return next
        })
        // Also store the bot's response as a memory (channel-wide)
        addMemory({
          content: botContent,
          userId: user.id,
          userName: 'EDEN',
          role: 'assistant',
          extraMetadata: { replying_to: searchQuery.slice(0, 80) },
        }).then(() => setMemoryRefresh((n) => n + 1))
      }
    } catch (err) {
      console.error('OpenRouter error:', err)
    } finally {
      setStreaming(false)
      setLayerStage(7)
      setTimeout(() => setLayerStage(0), 1200)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function handleInputChange(e) {
    setInput(e.target.value)
    broadcastTyping()
  }

  // Grouping
  const grouped = useMemo(() => {
    const groups = []
    let i = 0
    while (i < messages.length) {
      const msg = messages[i]
      const isBot = msg.role === 'assistant'
      const authorId = isBot ? 'eden-bot' : msg.user_id
      const dateLabel = formatDate(msg.created_at)
      const prevMsg = i > 0 ? messages[i - 1] : null
      const prevDateLabel = prevMsg ? formatDate(prevMsg.created_at) : null
      const needsDivider = !prevMsg || prevDateLabel !== dateLabel
      const group = [msg]
      i++
      while (i < messages.length) {
        const next = messages[i]
        const nextIsBot = next.role === 'assistant'
        const nextAuthorId = nextIsBot ? 'eden-bot' : next.user_id
        const nextDate = formatDate(next.created_at)
        if (nextAuthorId === authorId && nextDate === dateLabel) { group.push(next); i++ }
        else break
      }
      groups.push({ group, isBot, needsDivider, dateLabel })
    }
    return groups
  }, [messages])

  const activeTypers = Object.values(typingUsers)
  const memoryEnabled = supermemoryConfigured()

  return (
    <div className="flex h-screen bg-bg-primary text-white overflow-hidden">

      {/* Left sidebar */}
      <aside className="w-60 flex-shrink-0 flex flex-col border-r border-white/10 bg-bg-secondary">
        <div className="px-4 py-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-white flex items-center justify-center">
              <span className="text-black font-bold text-xs">E</span>
            </div>
            <span className="font-semibold text-sm tracking-tight">EDEN Workspace</span>
          </div>
          {memoryEnabled && (
            <div className="mt-3 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-cyan-400/70">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              Cognitive layer online
            </div>
          )}
        </div>

        <div className="px-3 pt-5 pb-2">
          <p className="text-[11px] font-semibold uppercase tracking-widest text-white/30 px-2 mb-2">Channels</p>
          <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-white/10 text-white text-sm cursor-default">
            <Hash size={14} className="text-white/50 flex-shrink-0" />
            <span>eden-bot</span>
          </div>
        </div>

        <div className="flex-1" />

        <div className="px-3 py-4 border-t border-white/10">
          {isLoaded && isSignedIn ? (
            <div className="flex items-center gap-2">
              <Avatar name={user.fullName || user.firstName} imageUrl={user.imageUrl} size={7} />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold truncate">{user.fullName || user.firstName}</p>
                <p className="text-[10px] text-white/40 truncate">{user.primaryEmailAddress?.emailAddress}</p>
              </div>
              <button
                onClick={() => signOut({ redirectUrl: 'https://eden-robotics.github.io/Eden/' })}
                className="text-white/30 hover:text-white/70 transition-colors"
                title="Sign out"
              >
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            isLoaded && (
              <SignInButton mode="modal">
                <button className="flex items-center gap-2 text-xs text-white/40 hover:text-white transition-colors w-full px-2 py-1.5 rounded-md hover:bg-white/5">
                  <LogIn size={13} /> Sign in
                </button>
              </SignInButton>
            )
          )}
        </div>

        <div className="px-3 pb-4">
          <Link to="/" className="flex items-center gap-2 text-[11px] text-white/25 hover:text-white/50 transition-colors px-2 py-1">
            <ArrowLeft size={11} /> Back to site
          </Link>
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex flex-1 min-w-0 relative">
        <div className="flex flex-col flex-1 min-w-0">

          {/* Channel header */}
          <header className="flex items-center gap-3 px-6 py-4 border-b border-white/10 flex-shrink-0">
            <Hash size={18} className="text-white/40" />
            <span className="font-semibold text-sm">eden-bot</span>
            <div className="h-4 w-px bg-white/10 mx-1" />
            <span className="text-xs text-white/30 flex items-center gap-1.5">
              <AtSign size={11} className="text-cyan-400/60" />
              Type <span className="font-mono text-cyan-300/80">@eden</span> to invoke the cognitive layer
            </span>
            <div className="flex-1" />
            <button
              onClick={() => setMemoryPanelOpen(!memoryPanelOpen)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border transition-all ${memoryPanelOpen ? 'bg-cyan-500/20 border-cyan-400/40 text-cyan-200' : 'border-white/10 text-white/60 hover:border-white/30 hover:text-white'}`}
            >
              <Brain size={13} />
              Memory
            </button>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto py-4 eden-chat-scroll">
            {loadingHistory ? (
              <div className="flex items-center justify-center h-full">
                <Loader2 size={20} className="animate-spin text-white/20" />
              </div>
            ) : messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-8">
                <div className="w-12 h-12 rounded-2xl bg-white flex items-center justify-center">
                  <span className="text-black font-bold text-lg">E</span>
                </div>
                <p className="text-white/60 text-sm">This is the beginning of <span className="text-white font-semibold">#eden-bot</span></p>
                <p className="text-white/30 text-xs max-w-sm">
                  Talk freely — everyone sees the channel. Mention <span className="font-mono text-cyan-300">@eden</span> to pull EDEN into the conversation.
                  EDEN remembers every exchange via its Supermemory layer.
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-4 max-w-md">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s.label}
                      onClick={() => sendMessage(s.send)}
                      className="px-3 py-1.5 rounded-full border border-white/10 text-xs text-white/60 hover:text-white hover:border-cyan-400/40 hover:bg-cyan-400/5 transition-all font-mono"
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-0.5">
                {grouped.map(({ group, isBot, needsDivider, dateLabel }) => (
                  <React.Fragment key={group[0].id}>
                    {needsDivider && <DateDivider label={dateLabel} />}
                    <MessageGroup messages={group} isBot={isBot} memoriesById={memoriesById} />
                  </React.Fragment>
                ))}

                {retrieving && (
                  <div className="flex items-center gap-3 px-4 py-2">
                    <EdenAvatar glow />
                    <div className="flex items-center gap-2 text-[11px] text-cyan-300/70 font-mono uppercase tracking-widest">
                      <Sparkles size={11} className="animate-pulse" />
                      Retrieving memory…
                    </div>
                  </div>
                )}
                {streaming && !retrieving && (
                  <div className="flex items-center gap-3 px-4 py-2">
                    <EdenAvatar glow />
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          className="w-1.5 h-1.5 rounded-full bg-cyan-400/60"
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <div ref={bottomRef} />
              </div>
            )}
          </div>

          {/* Layer activation pipeline (visible while processing) */}
          <AnimatePresence>
            {layerStage > 0 && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="flex-shrink-0 border-t border-white/5 bg-gradient-to-r from-cyan-500/[0.03] via-violet-500/[0.03] to-rose-500/[0.03] overflow-hidden"
              >
                <LayerPipeline stage={layerStage} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Typing indicators for humans */}
          <div className="px-6 h-5 flex-shrink-0 flex items-center">
            {activeTypers.length > 0 && (
              <div className="flex items-center gap-2 text-[11px] text-white/40">
                <div className="flex gap-0.5">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1 h-1 rounded-full bg-white/40"
                      animate={{ y: [0, -2, 0] }}
                      transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
                    />
                  ))}
                </div>
                <span>
                  {activeTypers.length === 1
                    ? `${activeTypers[0].name} is typing…`
                    : activeTypers.length === 2
                      ? `${activeTypers[0].name} and ${activeTypers[1].name} are typing…`
                      : `${activeTypers.length} people are typing…`}
                </span>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="px-6 pb-6 pt-1 flex-shrink-0 relative">
            {/* Slash command autocomplete */}
            <AnimatePresence>
              {input.startsWith('/') && !streaming && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  className="absolute left-6 right-6 bottom-full mb-2 rounded-xl border border-cyan-400/20 bg-bg-secondary/95 backdrop-blur shadow-2xl overflow-hidden z-20"
                >
                  <div className="px-3 py-2 border-b border-white/5 flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-cyan-300/70">
                    <Terminal size={10} /> Slash commands
                  </div>
                  {SLASH_COMMANDS.filter((c) => c.cmd.startsWith(input.split(/\s+/)[0].toLowerCase())).map((c) => (
                    <button
                      key={c.cmd}
                      onClick={() => { setInput(c.cmd + (c.args ? ' ' : '')); inputRef.current?.focus() }}
                      className="w-full px-4 py-2 flex items-center gap-3 hover:bg-white/5 transition-colors text-left"
                    >
                      <code className="text-sm text-cyan-300 font-mono flex-shrink-0">{c.cmd}</code>
                      {c.args && <span className="text-xs text-white/30 font-mono">{c.args}</span>}
                      <span className="text-xs text-white/50 ml-auto">{c.desc}</span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            {!isSignedIn && isLoaded ? (
              <div className="flex items-center justify-center gap-3 py-4 rounded-xl border border-white/10 bg-white/[0.02]">
                <span className="text-sm text-white/40">Sign in to send messages</span>
                <SignInButton mode="modal">
                  <button className="text-xs font-semibold bg-white text-black px-3 py-1.5 rounded-md hover:bg-white/90 transition-colors">Sign in</button>
                </SignInButton>
              </div>
            ) : (
              <div className="flex items-end gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus-within:border-cyan-400/30 transition-colors">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  placeholder="Message #eden-bot  ·  @eden to invoke  ·  / for commands"
                  rows={1}
                  disabled={streaming}
                  className="flex-1 bg-transparent text-sm text-white placeholder:text-white/25 resize-none outline-none leading-relaxed max-h-40 overflow-y-auto"
                  style={{ minHeight: '24px' }}
                  onInput={(e) => {
                    e.target.style.height = 'auto'
                    e.target.style.height = e.target.scrollHeight + 'px'
                  }}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || streaming || !isSignedIn}
                  className="flex-shrink-0 w-8 h-8 rounded-lg bg-white text-black flex items-center justify-center hover:bg-white/90 transition-colors disabled:opacity-20 disabled:cursor-not-allowed"
                >
                  {streaming ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Members sidebar */}
        <aside className="w-52 flex-shrink-0 border-l border-white/10 bg-bg-secondary flex flex-col">
          <div className="px-4 py-4 border-b border-white/10">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-white/30">Members</p>
          </div>
          <div className="flex-1 overflow-y-auto py-3 eden-chat-scroll">
            {/* EDEN bot (always "present") */}
            <div className="px-3 mb-4">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-white/25 px-2 mb-2">Bot</p>
              <div className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-white/5">
                <div className="relative">
                  <EdenAvatar size={7} />
                  <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-cyan-400 rounded-full border-2 border-bg-secondary" />
                </div>
                <span className="text-xs text-white/80 truncate">EDEN</span>
                <span className="ml-auto text-[9px] font-mono text-cyan-400/70 uppercase">live</span>
              </div>
            </div>

            {onlineMembers.length > 0 && (
              <div className="px-3 mb-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-white/25 px-2 mb-2">
                  Online — {onlineMembers.length}
                </p>
                {onlineMembers.map((m) => (
                  <div key={m.user_id} className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-white/5">
                    <div className="relative">
                      <Avatar name={m.user_name} imageUrl={m.user_avatar} size={7} />
                      <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-bg-secondary" />
                    </div>
                    <span className="text-xs text-white/70 truncate">{m.user_name}</span>
                  </div>
                ))}
              </div>
            )}

            {(() => {
              const seen = new Set(onlineMembers.map((m) => m.user_id))
              const offline = []
              for (const msg of messages) {
                if (msg.role === 'user' && !seen.has(msg.user_id)) {
                  seen.add(msg.user_id)
                  offline.push({ user_id: msg.user_id, user_name: msg.user_name, user_avatar: msg.user_avatar })
                }
              }
              if (offline.length === 0) return null
              return (
                <div className="px-3">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-white/25 px-2 mb-2">
                    Offline — {offline.length}
                  </p>
                  {offline.map((m) => (
                    <div key={m.user_id} className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-white/5">
                      <div className="relative opacity-50">
                        <Avatar name={m.user_name} imageUrl={m.user_avatar} size={7} />
                      </div>
                      <span className="text-xs text-white/40 truncate">{m.user_name}</span>
                    </div>
                  ))}
                </div>
              )
            })()}
          </div>
        </aside>

        {/* Memory panel overlay */}
        <MemoryPanel
          open={memoryPanelOpen}
          onClose={() => setMemoryPanelOpen(false)}
          userId={user?.id}
          refreshKey={memoryRefresh}
        />
      </div>
    </div>
  )
}
