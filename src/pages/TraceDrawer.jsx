import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X, Copy, ChevronRight, Eye, Globe, Brain, Sparkles, Layers, Cpu, Database, Activity,
} from 'lucide-react'

// ───── Helpers ─────
const fmt = (n, d = 0) => (n == null ? '—' : Number(n).toFixed(d))
const ms = (n) => (n == null ? '—' : `${Math.round(n)}ms`)
function copy(txt) {
  try { navigator.clipboard.writeText(typeof txt === 'string' ? txt : JSON.stringify(txt, null, 2)) } catch {}
}

// Rough token estimate — GPT-style heuristic, ~4 chars/token
function tokenEstimate(s) { return s ? Math.ceil(s.length / 4) : 0 }

// ───── JSON viewer (simple, collapsible) ─────
function JsonView({ data, depth = 0 }) {
  if (data == null) return <span className="text-white/30">null</span>
  if (typeof data === 'string') return <span className="text-emerald-300">"{data}"</span>
  if (typeof data === 'number') return <span className="text-amber-300">{data}</span>
  if (typeof data === 'boolean') return <span className="text-rose-300">{String(data)}</span>
  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="text-white/30">[]</span>
    return (
      <div className="pl-3 border-l border-white/10">
        {data.map((v, i) => (
          <div key={i} className="flex gap-1 text-[11px]">
            <span className="text-white/30 font-mono">{i}:</span>
            <JsonView data={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }
  if (typeof data === 'object') {
    const keys = Object.keys(data)
    if (keys.length === 0) return <span className="text-white/30">{'{}'}</span>
    return (
      <div className={depth === 0 ? '' : 'pl-3 border-l border-white/10'}>
        {keys.map((k) => (
          <div key={k} className="flex gap-1 text-[11px] leading-tight py-0.5">
            <span className="text-cyan-300 font-mono">{k}:</span>
            <JsonView data={data[k]} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }
  return <span className="text-white/60">{String(data)}</span>
}

// ───── Section wrapper ─────
function Section({ icon: Icon, title, subtitle, color = 'cyan', children, defaultOpen = true, copyable }) {
  const [open, setOpen] = useState(defaultOpen)
  const accent = {
    amber:   'text-amber-300 border-amber-400/20 bg-amber-500/[0.03]',
    blue:    'text-blue-300 border-blue-400/20 bg-blue-500/[0.03]',
    cyan:    'text-cyan-300 border-cyan-400/20 bg-cyan-500/[0.03]',
    violet:  'text-violet-300 border-violet-400/20 bg-violet-500/[0.03]',
    emerald: 'text-emerald-300 border-emerald-400/20 bg-emerald-500/[0.03]',
    rose:    'text-rose-300 border-rose-400/20 bg-rose-500/[0.03]',
    white:   'text-white/70 border-white/10 bg-white/[0.02]',
  }[color]
  return (
    <div className={`border rounded-lg ${accent} overflow-hidden`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-3 py-2 flex items-center gap-2 text-left hover:bg-white/[0.02]"
      >
        <Icon size={12} />
        <span className="text-[10px] font-mono uppercase tracking-widest font-semibold">{title}</span>
        {subtitle && <span className="text-[10px] font-mono text-white/40 ml-1">{subtitle}</span>}
        <div className="flex-1" />
        {copyable && (
          <span
            onClick={(e) => { e.stopPropagation(); copy(copyable) }}
            className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white cursor-pointer"
            title="Copy"
          ><Copy size={10} /></span>
        )}
        <ChevronRight size={11} className={`transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 border-t border-white/5">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function Row({ k, v, mono = false }) {
  return (
    <div className="flex gap-3 text-[11px] py-0.5">
      <span className="text-white/40 font-mono uppercase tracking-widest w-32 flex-shrink-0">{k}</span>
      <span className={`text-white/80 ${mono ? 'font-mono' : ''} break-all flex-1`}>{v == null ? '—' : v}</span>
    </div>
  )
}

// ───── Timing bar chart ─────
function TimingBars({ trace }) {
  const total = trace?.cognition?.latencyTotal || 1
  const bars = [
    { label: 'retrieval',         ms: trace?.retrieval?.durationMs || 0, color: 'bg-cyan-400' },
    { label: 'cognition (total)', ms: trace?.cognition?.latencyTotal || 0, color: 'bg-violet-400' },
    { label: '└ to first token',  ms: trace?.cognition?.latencyToFirstToken || 0, color: 'bg-violet-400/60' },
    { label: 'action dispatch',   ms: trace?.action?.dispatchMs || 0, color: 'bg-rose-400' },
  ]
  const scale = Math.max(total, ...bars.map((b) => b.ms), 100)
  return (
    <div className="space-y-1.5">
      {bars.map((b) => (
        <div key={b.label} className="flex items-center gap-2 text-[10px] font-mono">
          <span className="w-32 text-white/50 uppercase tracking-widest flex-shrink-0">{b.label}</span>
          <div className="flex-1 h-3 bg-white/5 rounded-sm overflow-hidden relative">
            <div className={`h-full ${b.color} rounded-sm`} style={{ width: `${Math.min(100, (b.ms / scale) * 100)}%` }} />
          </div>
          <span className="text-white/70 w-14 text-right">{ms(b.ms)}</span>
        </div>
      ))}
    </div>
  )
}

// ───── Pipeline diagram (6-stage graph) ─────
const STAGES = [
  { id: 'perception', name: 'Perception',  icon: Eye,       color: 'amber',   key: 'input' },
  { id: 'context',    name: 'Context',     icon: Globe,     color: 'blue',    key: 'context' },
  { id: 'memory',     name: 'Supermemory', icon: Database,  color: 'cyan',    key: 'retrieval' },
  { id: 'cognitive',  name: 'Cognitive',   icon: Sparkles,  color: 'violet',  key: 'cognition' },
  { id: 'planning',   name: 'Planning',    icon: Layers,    color: 'emerald', key: 'planning' },
  { id: 'action',     name: 'Action',      icon: Cpu,       color: 'rose',    key: 'action' },
]

function PipelineDiagram({ trace }) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1">
      {STAGES.map((s, i) => {
        const Icon = s.icon
        const active = !!trace?.[s.key]
        const cls = {
          amber:   active ? 'bg-amber-500/15 text-amber-200 border-amber-400/40' : '',
          blue:    active ? 'bg-blue-500/15 text-blue-200 border-blue-400/40' : '',
          cyan:    active ? 'bg-cyan-500/15 text-cyan-200 border-cyan-400/40' : '',
          violet:  active ? 'bg-violet-500/15 text-violet-200 border-violet-400/40' : '',
          emerald: active ? 'bg-emerald-500/15 text-emerald-200 border-emerald-400/40' : '',
          rose:    active ? 'bg-rose-500/15 text-rose-200 border-rose-400/40' : '',
        }[s.color]
        return (
          <React.Fragment key={s.id}>
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md border flex-shrink-0 ${active ? cls : 'border-white/10 text-white/30'}`}>
              <Icon size={10} />
              <span className="text-[9px] font-mono uppercase tracking-widest">{s.name}</span>
            </div>
            {i < STAGES.length - 1 && <div className={`h-px w-3 flex-shrink-0 ${active ? 'bg-white/30' : 'bg-white/10'}`} />}
          </React.Fragment>
        )
      })}
    </div>
  )
}

// ───── Main drawer ─────
export default function TraceDrawer({ trace, onClose }) {
  if (!trace) return null

  const env = trace.cognition?.envelope || {}
  const mems = trace.retrieval?.memories || []
  const people = trace.retrieval?.people || []
  const vibe = trace.retrieval?.vibe

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <motion.aside
        initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 30, stiffness: 260 }}
        className="fixed right-0 top-0 bottom-0 w-[620px] max-w-full z-50 bg-bg-secondary border-l border-white/10 flex flex-col shadow-2xl"
      >
        <header className="px-5 py-4 border-b border-white/10 flex items-center gap-3 flex-shrink-0">
          <Activity size={14} className="text-cyan-400" />
          <div>
            <div className="text-sm font-semibold">Pipeline Trace</div>
            <div className="text-[10px] font-mono text-white/40">
              turn_id={trace.msgId?.slice(0, 12) || '—'} · t0={new Date(trace.turnStartTs).toLocaleTimeString([], { hour12: false })}
            </div>
          </div>
          <div className="flex-1" />
          <button
            onClick={() => copy(trace)}
            className="text-[10px] font-mono uppercase tracking-widest px-2 py-1 rounded border border-white/10 text-white/60 hover:text-white hover:border-white/30"
            title="Copy full trace as JSON"
          >
            <Copy size={11} className="inline mr-1" /> copy
          </button>
          <button onClick={onClose} className="text-white/40 hover:text-white p-1"><X size={16} /></button>
        </header>

        <div className="flex-1 overflow-y-auto eden-chat-scroll p-4 space-y-3">
          {/* Pipeline diagram */}
          <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
            <div className="text-[10px] font-mono uppercase tracking-widest text-white/40 mb-2">stage graph</div>
            <PipelineDiagram trace={trace} />
          </div>

          {/* Timing */}
          <Section icon={Activity} title="Timings" color="cyan" defaultOpen>
            <TimingBars trace={trace} />
          </Section>

          {/* 1. Perception / Input */}
          <Section icon={Eye} title="1. Perception Layer" subtitle="raw input frame" color="amber">
            <Row k="role"       v={trace.input?.role || 'user'} />
            <Row k="user_id"    v={trace.input?.userId} mono />
            <Row k="user_name"  v={trace.input?.userName} />
            <Row k="has_image"  v={String(!!trace.input?.image)} />
            <Row k="mentions"   v={trace.input?.hasMention ? 'true (bot invoked)' : 'false'} />
            <Row k="text"       v={<span className="text-white/80">{trace.input?.text || '—'}</span>} />
            {trace.input?.imageInfo && (
              <Row k="image"     v={`${trace.input.imageInfo.w}×${trace.input.imageInfo.h} · ${Math.round(trace.input.imageInfo.size / 1024)}KB`} mono />
            )}
          </Section>

          {/* 2. Context */}
          <Section icon={Globe} title="2. Context Layer" subtitle={`people_known=${people.length}`} color="blue">
            <Row k="speaker"    v={trace.input?.userName} />
            <Row k="email"      v={trace.input?.email} mono />
            <Row k="identified" v={trace.context?.identityKnown ? 'yes (Perception record exists)' : 'no (first seen)'} />
            {people.length > 0 && (
              <div className="mt-2">
                <div className="text-[10px] font-mono uppercase tracking-widest text-white/40 mb-1">known people injected</div>
                <div className="space-y-0.5 font-mono text-[11px] text-white/70">
                  {people.slice(0, 8).map((p, i) => (
                    <div key={i}>· {p.metadata?.user_name || '?'} {p.metadata?.email ? <span className="text-white/30">({p.metadata.email})</span> : null}</div>
                  ))}
                </div>
              </div>
            )}
          </Section>

          {/* 3. Supermemory retrieval */}
          <Section icon={Database} title="3. Supermemory Retrieval" subtitle={`${mems.length} hit${mems.length === 1 ? '' : 's'} · hybrid search`} color="cyan">
            <Row k="query"         v={<code className="text-cyan-200">{trace.retrieval?.query || '—'}</code>} mono />
            <Row k="containers"    v="eden-user-{id} ∪ eden-channel-eden-bot" mono />
            <Row k="search_mode"   v="hybrid (vector + keyword)" />
            <Row k="limit"         v={trace.retrieval?.limit ?? 5} mono />
            <Row k="returned"      v={mems.length} mono />
            <Row k="latency"       v={ms(trace.retrieval?.durationMs)} mono />
            {mems.length > 0 && (
              <div className="mt-2">
                <div className="text-[10px] font-mono uppercase tracking-widest text-white/40 mb-1">top memories</div>
                <div className="space-y-1">
                  {mems.map((m, i) => (
                    <div key={i} className="flex items-start gap-2 text-[11px] rounded bg-white/[0.02] px-2 py-1.5 border border-white/5">
                      <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded ${m.source === 'personal' ? 'bg-violet-500/15 text-violet-300' : 'bg-cyan-500/15 text-cyan-300'}`}>{m.source}</span>
                      <span className="text-[9px] font-mono text-white/40 flex-shrink-0">score={fmt(m.score, 3)}</span>
                      <span className="text-white/70 flex-1">{m.content.length > 140 ? m.content.slice(0, 140) + '…' : m.content}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {vibe && (
              <div className="mt-2">
                <div className="text-[10px] font-mono uppercase tracking-widest text-white/40 mb-1">relationship state</div>
                <div className="text-[11px] text-white/70 font-mono">
                  total={vibe.total} · count={vibe.count} · recent=[{(vibe.recent || []).slice(0, 3).map((r) => `${r.delta > 0 ? '+' : ''}${r.delta}`).join(', ')}]
                </div>
              </div>
            )}
          </Section>

          {/* 4. Prompt assembly */}
          <Section icon={Sparkles} title="4. Prompt Assembly" subtitle="system + memory + people + vibe + history" color="violet"
            copyable={trace.prompt?.systemPreview}>
            <Row k="system_chars"    v={trace.prompt?.systemChars ?? '—'} mono />
            <Row k="system_tokens~"  v={trace.prompt?.systemTokens ?? '—'} mono />
            <Row k="history_turns"   v={trace.prompt?.historyCount ?? '—'} mono />
            <Row k="total_chars"     v={trace.prompt?.totalChars ?? '—'} mono />
            <Row k="total_tokens~"   v={trace.prompt?.totalTokens ?? '—'} mono />
            <Row k="contains"        v="SYSTEM · PEOPLE · RELATIONSHIP · MEMORY · history[-20]" mono />
          </Section>

          {/* 5. Cognition / inference */}
          <Section icon={Brain} title="5. Cognitive Layer" subtitle={`${trace.cognition?.model || '?'} · stream`} color="violet">
            <Row k="endpoint"      v="POST https://openrouter.ai/api/v1/chat/completions" mono />
            <Row k="model"         v={trace.cognition?.model} mono />
            <Row k="stream"        v="true" mono />
            <Row k="first_token"   v={ms(trace.cognition?.latencyToFirstToken)} mono />
            <Row k="total_latency" v={ms(trace.cognition?.latencyTotal)} mono />
            <Row k="output_chars"  v={trace.cognition?.outputChars ?? '—'} mono />
            <Row k="output_tokens~"v={trace.cognition?.outputTokens ?? '—'} mono />
          </Section>

          {/* 6. Planning (envelope) */}
          <Section icon={Layers} title="6. Planning Layer" subtitle="parsed envelope" color="emerald">
            <Row k="plan"   v={env.plan ? <span className="text-white/80">{env.plan}</span> : <span className="text-white/30">(none — no plan emitted)</span>} />
            <Row k="tone"   v={env.tone || <span className="text-white/30">—</span>} mono />
            <Row k="vibe_delta"  v={env.vibe ? `${env.vibe.delta > 0 ? '+' : ''}${env.vibe.delta}` : '—'} mono />
            <Row k="vibe_reason" v={env.vibe?.reason || '—'} />
          </Section>

          {/* 7. Action */}
          <Section icon={Cpu} title="7. Action Layer" subtitle={env.action ? `raw="${env.action}"` : 'none'} color="rose">
            <Row k="raw"         v={env.action || <span className="text-white/30">none</span>} mono />
            <Row k="dispatched"  v={trace.action?.dispatched ? 'true → eden-action-bus' : 'false'} mono />
            <Row k="transports"  v="localStorage (primary) + supabase broadcast (secondary)" mono />
            {trace.action?.reason && <Row k="reason" v={trace.action.reason} />}
          </Section>

          {/* Raw trace JSON */}
          <Section icon={Database} title="Raw trace JSON" color="white" defaultOpen={false} copyable={trace}>
            <div className="max-h-72 overflow-y-auto eden-chat-scroll">
              <JsonView data={trace} />
            </div>
          </Section>
        </div>
      </motion.aside>
    </AnimatePresence>
  )
}
