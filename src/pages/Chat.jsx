import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useUser, useClerk, SignInButton } from '@clerk/clerk-react'
import { motion } from 'framer-motion'
import { Hash, Send, Loader2, ArrowLeft, LogOut, LogIn } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { supabase } from '../lib/supabase'
import "@fontsource/inter/400.css"
import "@fontsource/inter/600.css"
import "@fontsource/inter/800.css"

const SYSTEM_PROMPT = `You are EDEN (Emotionally-Driven Embodied Navigation), a humanoid robotics cognitive architecture assistant. You help users understand EDEN's layered architecture including the Perception Layer, Context Layer, Cognitive Layer, Supermemory, Planning Layer, and Action Layer. You are friendly, concise, and technically knowledgeable about robotics, AI, ROS 2, and human-robot interaction. Keep responses focused and under 200 words unless the user asks for detail.`

const OPENROUTER_KEY = import.meta.env.VITE_OPENROUTER_API_KEY

function formatTime(ts) {
  const d = new Date(ts)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatDate(ts) {
  const d = new Date(ts)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return d.toLocaleDateString([], { month: 'long', day: 'numeric', year: 'numeric' })
}

function Avatar({ name, imageUrl, size = 8 }) {
  if (imageUrl) {
    return (
      <img
        src={imageUrl}
        alt={name}
        className={`w-${size} h-${size} rounded-lg object-cover flex-shrink-0`}
      />
    )
  }
  const initials = name ? name.slice(0, 2).toUpperCase() : '??'
  const colors = [
    'bg-violet-600', 'bg-blue-600', 'bg-emerald-600',
    'bg-rose-600', 'bg-amber-600', 'bg-cyan-600'
  ]
  const color = colors[name ? name.charCodeAt(0) % colors.length : 0]
  return (
    <div className={`w-${size} h-${size} rounded-lg ${color} flex items-center justify-center flex-shrink-0 text-white font-semibold text-xs`}>
      {initials}
    </div>
  )
}

function EdenAvatar({ size = 8 }) {
  return (
    <div className={`w-${size} h-${size} rounded-lg bg-white flex items-center justify-center flex-shrink-0 flex-shrink-0`}>
      <span className="text-black font-bold text-xs">E</span>
    </div>
  )
}

function MessageGroup({ messages, isBot }) {
  const first = messages[0]
  const name = isBot ? 'EDEN' : first.user_name

  return (
    <div className="flex gap-3 px-4 py-1 hover:bg-white/[0.02] rounded-lg group">
      <div className="mt-0.5">
        {isBot ? <EdenAvatar /> : <Avatar name={name} imageUrl={first.user_avatar} />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 mb-1">
          <span className={`font-semibold text-sm ${isBot ? 'text-white' : 'text-white/90'}`}>
            {name}
          </span>
          {isBot && (
            <span className="text-[10px] font-mono bg-white/10 text-white/50 px-1.5 py-0.5 rounded uppercase tracking-wider">
              bot
            </span>
          )}
          <span className="text-[11px] text-white/30 opacity-0 group-hover:opacity-100 transition-opacity">
            {formatTime(first.created_at)}
          </span>
        </div>
        <div className="space-y-1">
          {messages.map((msg) => (
            <div key={msg.id} className="text-sm text-white/80 leading-relaxed">
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
                <span>{msg.content}</span>
              )}
            </div>
          ))}
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

export default function Chat() {
  const { user, isSignedIn, isLoaded } = useUser()
  const { signOut } = useClerk()

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [onlineMembers, setOnlineMembers] = useState([])

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const channelRef = useRef(null)

  // Gate: show sign-in wall if not authenticated
  if (isLoaded && !isSignedIn) {
    return (
      <div className="flex h-screen bg-bg-primary text-white items-center justify-center">
        <div className="flex flex-col items-center gap-6 text-center px-8">
          <div className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center">
            <span className="text-black font-bold text-xl">E</span>
          </div>
          <div>
            <h1 className="text-xl font-semibold mb-2">EDEN Workspace</h1>
            <p className="text-white/40 text-sm max-w-xs">Sign in to access the team chat and interact with EDEN.</p>
          </div>
          <SignInButton mode="modal">
            <button className="bg-white text-black text-sm font-semibold px-5 py-2.5 rounded-lg hover:bg-white/90 transition-colors">
              Sign in
            </button>
          </SignInButton>
          <Link to="/" className="text-xs text-white/25 hover:text-white/50 transition-colors">
            Back to site
          </Link>
        </div>
      </div>
    )
  }

  // Load messages from Supabase
  useEffect(() => {
    async function loadMessages() {
      setLoadingHistory(true)
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .order('created_at', { ascending: true })
      if (!error && data) setMessages(data)
      setLoadingHistory(false)
    }
    loadMessages()
  }, [])

  // Realtime subscription + Presence
  useEffect(() => {
    if (!user) return

    const channel = supabase.channel('messages-channel', { config: { presence: { key: user.id } } })
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'messages' }, (payload) => {
        setMessages((prev) => {
          if (prev.find((m) => m.id === payload.new.id)) return prev
          return [...prev, payload.new]
        })
      })
      .on('presence', { event: 'sync' }, () => {
        const state = channel.presenceState()
        const members = Object.values(state).flat().map((p) => ({
          user_id: p.user_id,
          user_name: p.user_name,
          user_avatar: p.user_avatar,
        }))
        setOnlineMembers(members)
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

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage() {
    if (!input.trim() || streaming || !isSignedIn) return

    const content = input.trim()
    setInput('')
    setStreaming(true)

    // Insert user message to Supabase
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

    if (userMsg) {
      setMessages((prev) => [...prev.filter((m) => m.id !== userMsg.id), userMsg])
    }

    // Build conversation history for LLM
    const allMsgs = [...messages]
    if (userMsg) allMsgs.push(userMsg)

    const conversation = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...allMsgs.map((m) => ({ role: m.role, content: m.content })),
    ]

    // Stream from OpenRouter
    try {
      const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENROUTER_KEY}`,
          'HTTP-Referer': window.location.origin,
          'X-Title': 'EDEN Robotics',
        },
        body: JSON.stringify({
          model: 'nvidia/nemotron-3-super-120b-a12b:free',
          messages: conversation,
          stream: true,
        }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let botContent = ''

      // Optimistically add bot message to local state
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
              botContent += delta
              setMessages((prev) =>
                prev.map((m) => m.id === tempId ? { ...m, content: botContent } : m)
              )
            }
          } catch { /* ignore parse errors */ }
        }
      }

      // Save final bot message to Supabase
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

      // Replace temp message with real one
      if (botMsg) {
        setMessages((prev) => prev.map((m) => m.id === tempId ? botMsg : m))
      }
    } catch (err) {
      console.error('OpenRouter error:', err)
    }

    setStreaming(false)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Group messages by author + consecutive run + date
  function groupMessages(msgs) {
    const groups = []
    let i = 0
    while (i < msgs.length) {
      const msg = msgs[i]
      const isBot = msg.role === 'assistant'
      const authorId = isBot ? 'eden-bot' : msg.user_id
      const dateLabel = formatDate(msg.created_at)

      // Check if we need a date divider
      const prevMsg = i > 0 ? msgs[i - 1] : null
      const prevDateLabel = prevMsg ? formatDate(prevMsg.created_at) : null
      const needsDivider = !prevMsg || prevDateLabel !== dateLabel

      // Collect consecutive messages from same author
      const group = [msg]
      i++
      while (i < msgs.length) {
        const next = msgs[i]
        const nextIsBot = next.role === 'assistant'
        const nextAuthorId = nextIsBot ? 'eden-bot' : next.user_id
        const nextDate = formatDate(next.created_at)
        if (nextAuthorId === authorId && nextDate === dateLabel) {
          group.push(next)
          i++
        } else break
      }

      groups.push({ group, isBot, needsDivider, dateLabel })
    }
    return groups
  }

  const grouped = groupMessages(messages)

  return (
    <div className="flex h-screen bg-bg-primary text-white overflow-hidden">

      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 flex flex-col border-r border-white/10 bg-bg-secondary">
        {/* Workspace header */}
        <div className="px-4 py-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-white flex items-center justify-center">
              <span className="text-black font-bold text-xs">E</span>
            </div>
            <span className="font-semibold text-sm tracking-tight">EDEN Workspace</span>
          </div>
        </div>

        {/* Channels */}
        <div className="px-3 pt-5 pb-2">
          <p className="text-[11px] font-semibold uppercase tracking-widest text-white/30 px-2 mb-2">Channels</p>
          <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-white/10 text-white text-sm cursor-default">
            <Hash size={14} className="text-white/50 flex-shrink-0" />
            <span>eden-bot</span>
          </div>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* User section */}
        <div className="px-3 py-4 border-t border-white/10">
          {isLoaded && isSignedIn ? (
            <div className="flex items-center gap-2">
              <Avatar name={user.fullName || user.firstName} imageUrl={user.imageUrl} size={7} />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold truncate">{user.fullName || user.firstName}</p>
                <p className="text-[10px] text-white/40 truncate">{user.primaryEmailAddress?.emailAddress}</p>
              </div>
              <button
                onClick={() => signOut()}
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
                  <LogIn size={13} />
                  Sign in
                </button>
              </SignInButton>
            )
          )}
        </div>

        {/* Back to site */}
        <div className="px-3 pb-4">
          <Link
            to="/"
            className="flex items-center gap-2 text-[11px] text-white/25 hover:text-white/50 transition-colors px-2 py-1"
          >
            <ArrowLeft size={11} />
            Back to site
          </Link>
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex flex-1 min-w-0">
        <div className="flex flex-col flex-1 min-w-0">

        {/* Channel header */}
        <header className="flex items-center gap-3 px-6 py-4 border-b border-white/10 flex-shrink-0">
          <Hash size={18} className="text-white/40" />
          <span className="font-semibold text-sm">eden-bot</span>
          <div className="h-4 w-px bg-white/10 mx-1" />
          <span className="text-xs text-white/30">Ask EDEN anything about the robotics architecture</span>
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
              <p className="text-white/30 text-xs max-w-sm">Ask EDEN about the robotics architecture, how layers communicate, or anything about the project.</p>
            </div>
          ) : (
            <div className="space-y-0.5">
              {grouped.map(({ group, isBot, needsDivider, dateLabel }) => (
                <React.Fragment key={group[0].id}>
                  {needsDivider && <DateDivider label={dateLabel} />}
                  <MessageGroup messages={group} isBot={isBot} />
                </React.Fragment>
              ))}
              {streaming && (
                <div className="flex items-center gap-3 px-4 py-2">
                  <EdenAvatar />
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-1.5 h-1.5 rounded-full bg-white/40"
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

        {/* Input area */}
        <div className="px-6 pb-6 pt-2 flex-shrink-0">
          {!isSignedIn && isLoaded ? (
            <div className="flex items-center justify-center gap-3 py-4 rounded-xl border border-white/10 bg-white/[0.02]">
              <span className="text-sm text-white/40">Sign in to send messages</span>
              <SignInButton mode="modal">
                <button className="text-xs font-semibold bg-white text-black px-3 py-1.5 rounded-md hover:bg-white/90 transition-colors">
                  Sign in
                </button>
              </SignInButton>
            </div>
          ) : (
            <div className="flex items-end gap-3 bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus-within:border-white/20 transition-colors">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Message #eden-bot`}
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
        </div>{/* end flex-col chat area */}

        {/* Members sidebar */}
        <aside className="w-52 flex-shrink-0 border-l border-white/10 bg-bg-secondary flex flex-col">
          <div className="px-4 py-4 border-b border-white/10">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-white/30">Members</p>
          </div>
          <div className="flex-1 overflow-y-auto py-3 eden-chat-scroll">
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

      </div>
    </div>
  )
}
