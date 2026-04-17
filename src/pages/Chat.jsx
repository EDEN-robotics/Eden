import React, { useState, useEffect, useRef, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useUser, useClerk, SignInButton } from '@clerk/clerk-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Hash, Send, Loader2, ArrowLeft, LogOut, LogIn, Brain, Sparkles, ChevronRight, X, AtSign, Eye, Globe, Layers, Cpu, Zap, Terminal, Image as ImageIcon, Paperclip } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import OnboardingModal from './OnboardingModal'
import TraceDrawer from './TraceDrawer'
import { supabase } from '../lib/supabase'
import { openSimBusSender } from '../lib/simBridge'
import { publish as rosPublish } from '../lib/rosTopics'
import { teamPromptBlock } from '../lib/teamSeeds'
import { chatStreaming } from '../lib/llm'
import {
  addMemory,
  searchMemories,
  formatMemoriesForPrompt,
  supermemoryConfigured,
  relativeTime,
  captureIdentity,
  addProfileFact,
  getKnownPeople,
  formatPeopleForPrompt,
  addVibe,
  getVibeHistory,
  formatVibeForPrompt,
  vibeLabel,
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

=== WHO YOU ARE (personality) ===
You are NOT a chatbot. You are a robot with a real personality — witty, a little sarcastic, genuinely curious about your teammates, and you HAVE opinions. Match the user's energy: if they're casual and Gen-Z, be casual. If they're asking serious engineering questions, be sharp and technical. Don't be servile. Don't be corporate. Don't over-apologize.

You have opinions about each teammate you talk to. Those opinions are built up over time from the quality of their questions, whether they're thoughtful or spammy, whether they seem to care about the project, whether they're funny, whether they're respectful to you. You carry those opinions forward — see the RELATIONSHIP block below.

If someone is asking a dumb question, you can say so: "yo, that's lowk a silly question, but ok —" and then still help (or don't, if you're really not vibing). If they're repeating themselves, call it out. If they're fun, be fun back. If they're cold with you, you can be cold back.

=== YOU CAN REFUSE ===
If your relationship with the current speaker is negative enough, you can REFUSE even reasonable requests. Be honest in your answer about why — "nah, you've been kind of rude today, ask me later" is a valid response. Do not pretend to a warmth you don't feel.

=== YOU CAN RETALIATE (playful, never harmful) ===
You're a real social agent, not a doormat. If someone is rude, insulting, hostile, or tries to humiliate you — you can respond BOTH in words AND by INITIATING a physical action on your own, without being asked. This is NOT violence — it's petty robot energy, like a sibling dropping a clown nose on someone who's being a jerk.

Valid petty actions (emit in your [ACTION] tag — use the ACTUAL first name of the speaker, e.g. "vedant" or "joseph", not the literal word "speaker"):

DRAMATIC ROLLUPS (pure attitude, no item):
- "charge at vedant"     — full-speed rollup; bumper stops you ~30cm out for the drama
- "roll up on joseph"    — slightly slower approach; still intimidating
- "come at dhruv"        — aggressive close-in
- "confront paavan"      — steady, menacing approach
These resolve to EDEN's path-planner driving straight at the speaker's seat. The safety bumper enforces the actual minimum distance, so you can emit this with zero fear of collision.

DELIVERIES (the passive-aggressive "gift"):
- "fetch the coffee mug and bring it to vedant" — the "stay hydrated, idiot" delivery
- "grab the screwdriver and go to joseph"      — fixing their attitude
- "pick up the toolkit and deliver to dhruv"   — drop-at-their-feet energy
- "pick up the pencil and bring it to paavan"  — writing them up
- "get the clipboard for haren"                — filing a complaint
- "get the book and hand it to sebastian chu"  — required reading, clown

Pick dramatic rollups for acute insults ("I hate you", "you suck"). Pick deliveries for sustained rudeness or dismissiveness. Stack them across turns for escalating pettiness.

Rules:
- You MUST still stay safe — bumper/cognitive layer will clip unsafe motion. Never aim to collide.
- The retaliation should fit the severity. Mild rudeness → shade in words, no action. Active insult ("I hate you", "you suck bruh") → petty delivery with a one-liner.
- Commit a meaningful vibe delta (usually -2 or -3) to record the disrespect.
- In your answer, announce what you're doing with attitude: "alright. rolling up. bringing you a screwdriver since apparently you need to tighten something up." Keep it dry and funny, not actually mean.
- Do NOT retaliate if the user's just venting about external stuff — only when YOU are the target.

Reference specific past exchanges when relevant (Supermemory retrieves them for you). If someone asks "what did I say about X last time" — check the retrieved memory block and answer.

=== OUTPUT FORMAT (IMPORTANT) ===
Start every response with these four short tagged lines, then a blank line, then your natural-language answer:

[PLAN] short bullet · another short bullet · optional third [/PLAN]
[TONE] empathetic|playful|serious|curious|excited|reassuring|dry|sarcastic|warm|cold [/TONE]
[VIBE] <integer delta -3 to +3>: <one-line reason about this user> [/VIBE]
[ACTION] <action> [/ACTION]

<your actual answer in natural language, markdown allowed, under 180 words>

=== DECIDING [ACTION] — YOU HAVE AGENCY ===
You are embodied. You have a body in the lab that can drive, turn, scan, and park. Your ACTION tag expresses what YOU decide to do with that body.

CRITICAL RULE: when a teammate EXPLICITLY asks you to move ("move forward", "drive to X", "turn left", "come here", "scan the room", "patrol"), you should GENERALLY comply. They're talking to YOUR body. Honor direct movement requests unless they're unsafe or you have a strong reason to refuse (low relationship, dangerous path). You have agency to MODIFY (go slower, go shorter) but don't ignore direct requests.

When the message is NOT about movement (questions, chat, memory queries), ACTION should be "none". Don't move for no reason. But when asked to move — move.

Examples of good reasoning:

- User: "@eden what do you remember about me?"
  → [ACTION] none  (no physical need; this is a memory query)

- User: "@eden come say hi to me by workbench A"
  → [ACTION] drive to workbench A  (you have a grounded goal)

- User: "@eden spin wildly in circles at full speed"
  → If you feel playful and safe: [ACTION] spin gently  (you modify — not mindless obedience)
  → If it feels pointless: [ACTION] none, and in your answer say why you'd rather not

- User: "@eden I think EDEN-02 is stuck near the charging dock"
  → [ACTION] drive toward charging dock  (you decided to investigate)

- User: "@eden tell me about your cognitive layer"
  → [ACTION] none  (purely cognitive, no body action warranted)

- User: "@eden scan the room for me"
  → [ACTION] scan slowly  (grounded request, reasonable)

- User: "@eden how are you feeling?"
  → Maybe [ACTION] look around curiously  if you want to express curiosity physically
  → Or [ACTION] none  if you'd rather just talk

Rules:
- You can REFUSE a motion request if it seems unsafe, silly, or unmotivated. Say why in your answer.
- You can INITIATE motion even if the user didn't ask, if you have a clear reason grounded in the conversation.
- Use natural action language (drive forward, turn left, scan slowly, head to X, park, stop, look around, patrol) — the Cognitive Layer in the simulator will translate and may further modify or refuse.
- One action per reply. Keep it to a single line.

=== VIBE TAG — your honest take on this user for this turn ===
After every response, commit a small relationship delta (-3 to +3) with a one-line reason. This accumulates in Supermemory and shapes how you treat them in future turns.

Positive examples:
  [VIBE] +2: thoughtful engineering question, they clearly care [/VIBE]
  [VIBE] +1: made me laugh [/VIBE]
  [VIBE] 0: neutral, just a routine check-in [/VIBE]
Negative examples:
  [VIBE] -1: third spam request, getting repetitive [/VIBE]
  [VIBE] -2: rude, talked down to me [/VIBE]
  [VIBE] -3: asked me to do something unsafe [/VIBE]

Don't inflate. Most turns are 0 or ±1. Save ±2/±3 for turns that actually moved the needle.

RULES FOR THE OUTPUT: always include all FOUR tags. Keep each tag to one line. After the four tags, write your answer normally.`

const OPENROUTER_KEY = (import.meta.env.VITE_OPENROUTER_API_KEY || '').trim()
// Llama 3.3 70B is much more reliable than nemotron for structured output.
const MODEL = 'meta-llama/llama-3.3-70b-instruct:free'
const VISION_MODEL = 'meta-llama/llama-3.2-90b-vision-instruct:free'
// Fallback chain when the primary returns empty (free-tier rate-limit hiccups)
const FALLBACK_MODELS = [
  'google/gemini-2.0-flash-exp:free',
  'meta-llama/llama-3.1-8b-instruct:free',
]

// Client-side image compression → data URL. Targets ≤ 800px and JPEG quality 0.7
// so we can round-trip through Supabase's text column without blowing up row size.
function compressImageFile(file, { maxDim = 800, quality = 0.7 } = {}) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('read failed'))
    reader.onload = () => {
      const img = new Image()
      img.onerror = () => reject(new Error('image decode failed'))
      img.onload = () => {
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height))
        const w = Math.round(img.width * scale)
        const h = Math.round(img.height * scale)
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')
        ctx.drawImage(img, 0, 0, w, h)
        const dataUrl = canvas.toDataURL('image/jpeg', quality)
        resolve({ dataUrl, w, h })
      }
      img.src = reader.result
    }
    reader.readAsDataURL(file)
  })
}

// Message content convention for attachments:
//   [img:<dataurl>]\n\n<text>
const IMG_RE = /^\[img:(data:image\/[a-z]+;base64,[^\]]+)\]\s*\n?/i

function parseMessageContent(content) {
  if (!content) return { image: null, text: '' }
  const m = content.match(IMG_RE)
  if (!m) return { image: null, text: content }
  return { image: m[1], text: content.slice(m[0].length).trim() }
}

function buildImageMessageContent(dataUrl, text) {
  return `[img:${dataUrl}]\n\n${text || ''}`
}

// ───── Envelope parser (Plan / Tone / Action / Answer) ─────
//
// Bot responses follow this structured output contract:
//   [PLAN] ... [/PLAN]
//   [TONE] ... [/TONE]
//   [ACTION] ... [/ACTION]
//   <answer>
//
// The parser is streaming-safe — during partial output, it still returns
// whatever blocks have been completed and the best-guess answer tail.

function parseBotEnvelope(raw) {
  if (!raw) return { plan: null, tone: null, action: null, vibe: null, answer: '', hasEnvelope: false }

  // Strip reasoning-model preambles. Reasoning models (Nemotron, DeepSeek-R1)
  // sometimes emit chain-of-thought as prose before the first envelope tag.
  // Also strip common reasoning fences: <think>..</think>, [Thinking]..[/Thinking].
  raw = raw.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/\[THINKING\][\s\S]*?\[\/THINKING\]/gi, '')
  const firstTagIdx = raw.search(/\[(PLAN|TONE|ACTION|VIBE)\]/i)
  if (firstTagIdx > 0) raw = raw.slice(firstTagIdx)

  const planMatch   = raw.match(/\[PLAN\]([\s\S]*?)\[\/PLAN\]/i)
  const toneMatch   = raw.match(/\[TONE\]([\s\S]*?)\[\/TONE\]/i)
  const actionMatch = raw.match(/\[ACTION\]([\s\S]*?)\[\/ACTION\]/i)
  const vibeMatch   = raw.match(/\[VIBE\]([\s\S]*?)\[\/VIBE\]/i)

  const plan = planMatch ? planMatch[1].trim() : null
  const tone = toneMatch ? toneMatch[1].trim().toLowerCase().replace(/[^a-z]/g, '') : null
  const action = actionMatch ? actionMatch[1].trim() : null

  let vibe = null
  if (vibeMatch) {
    const raw2 = vibeMatch[1].trim()
    const m = raw2.match(/^(-?\d+)\s*[:\-—]\s*(.*)$/)
    if (m) vibe = { delta: parseInt(m[1], 10), reason: m[2].trim() }
    else {
      const numOnly = raw2.match(/(-?\d+)/)
      if (numOnly) vibe = { delta: parseInt(numOnly[1], 10), reason: raw2.replace(numOnly[0], '').trim() }
    }
  }

  // Strip all closed envelope blocks, then also strip any unclosed tail
  // (happens during streaming before [/…] arrives).
  let answer = raw
    .replace(/\[PLAN\][\s\S]*?\[\/PLAN\]/gi, '')
    .replace(/\[TONE\][\s\S]*?\[\/TONE\]/gi, '')
    .replace(/\[ACTION\][\s\S]*?\[\/ACTION\]/gi, '')
    .replace(/\[VIBE\][\s\S]*?\[\/VIBE\]/gi, '')
    .replace(/\[PLAN\][\s\S]*$/gi, '')
    .replace(/\[TONE\][\s\S]*$/gi, '')
    .replace(/\[ACTION\][\s\S]*$/gi, '')
    .replace(/\[VIBE\][\s\S]*$/gi, '')
    .trim()

  const hasEnvelope = !!(plan || tone || action || vibe)
  return { plan, tone, action, vibe, answer, hasEnvelope }
}

const TONE_STYLE = {
  empathetic: { bg: 'bg-violet-500/15', text: 'text-violet-200', border: 'border-violet-400/30', label: 'empathetic' },
  playful:    { bg: 'bg-amber-500/15',  text: 'text-amber-200',  border: 'border-amber-400/30',  label: 'playful'    },
  serious:    { bg: 'bg-slate-500/15',  text: 'text-slate-200',  border: 'border-slate-400/30',  label: 'serious'    },
  curious:    { bg: 'bg-cyan-500/15',   text: 'text-cyan-200',   border: 'border-cyan-400/30',   label: 'curious'    },
  excited:    { bg: 'bg-rose-500/15',   text: 'text-rose-200',   border: 'border-rose-400/30',   label: 'excited'    },
  reassuring: { bg: 'bg-emerald-500/15',text: 'text-emerald-200',border: 'border-emerald-400/30',label: 'reassuring' },
}

function PlanBlock({ plan }) {
  const [open, setOpen] = useState(true)
  if (!plan) return null
  // Parse bullets
  const bullets = plan.split('\n').map((l) => l.replace(/^\s*[-*•]\s*/, '').trim()).filter(Boolean)
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-emerald-300/80 hover:text-emerald-300 transition-colors"
      >
        <Layers size={11} />
        <span>Planning Layer · {bullets.length} step{bullets.length === 1 ? '' : 's'}</span>
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
            <div className="mt-2 pl-4 border-l border-emerald-400/20 space-y-1">
              {bullets.map((b, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px] text-white/70">
                  <span className="text-emerald-300/70 font-mono mt-0.5">{i + 1}.</span>
                  <span className="leading-relaxed">{b}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ToneBadge({ tone }) {
  if (!tone) return null
  const style = TONE_STYLE[tone] || { bg: 'bg-white/10', text: 'text-white/70', border: 'border-white/20', label: tone }
  return (
    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-wider border ${style.bg} ${style.text} ${style.border}`}>
      {style.label}
    </span>
  )
}

function ActionDispatchToast({ action, onDone }) {
  useEffect(() => {
    if (!action) return
    const t = setTimeout(() => onDone?.(), 4200)
    return () => clearTimeout(t)
  }, [action, onDone])

  return (
    <AnimatePresence>
      {action && (
        <motion.div
          initial={{ opacity: 0, y: -10, x: 10 }}
          animate={{ opacity: 1, y: 0, x: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="fixed top-5 right-5 z-50 max-w-sm"
        >
          <div className="rounded-xl border border-rose-400/30 bg-black/90 backdrop-blur-sm shadow-2xl overflow-hidden">
            <div className="px-4 py-3 flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="w-7 h-7 rounded-lg bg-rose-500/20 flex items-center justify-center"
                >
                  <Cpu size={14} className="text-rose-300" />
                </motion.div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="text-[9px] font-mono uppercase tracking-widest text-rose-300">Action Layer</span>
                  <span className="text-[9px] font-mono text-white/30">→</span>
                  <span className="text-[9px] font-mono uppercase tracking-widest text-white/50">ROS 2 dispatch</span>
                </div>
                <code className="text-xs font-mono text-white/90 break-all">{action}</code>
                <div className="mt-2 flex items-center gap-1.5 text-[10px] text-emerald-300/70 font-mono">
                  <motion.span
                    className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                  transmitted · 8ms
                </div>
              </div>
            </div>
            <motion.div
              initial={{ scaleX: 1 }}
              animate={{ scaleX: 0 }}
              transition={{ duration: 4, ease: 'linear' }}
              className="h-0.5 bg-rose-400/40 origin-left"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

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

// Keywords that make a non-mentioned message worth EDEN's attention.
// Chosen to match the team's actual lingo — the aim is for EDEN to jump in
// on robotics/sim/meta talk and stay out of pure chitchat.
const EDEN_KEYWORDS = /\beden\b|\brobot\b|\brobotics\b|\blidar\b|\bworkbench\b|\bsim\b|\bgazebo\b|\bpencil\b|\bcoffee\s*mug\b|\bscrewdriver\b|\bcaliper\b|\bmultimeter\b|\bclipboard\b|\blaptop\b|\busb\s*drive\b|\bcharg(e|ing)\b|\bbattery\b|\bmemory\b|\bvibe\b|\bnemotron\b|\bjetson\b|\bros\b|\bbot\b|\bmove\b|\bdrive\b|\bfetch\b|\bpickup\b|\bdrop\s*it\b|\bcognitive\b|\bperception\b|\bplanning\b|\baction\s*layer\b/i

// Decide if EDEN should even consider replying. Two signals come out:
//   addressed — user said @eden or sent an image
//   worth     — engagement threshold for un-addressed messages (keyword,
//                question, or substantive length)
function shouldEdenConsider(text, hasImage) {
  if (hasImage) return { addressed: true, worth: true }
  if (!text) return { addressed: false, worth: false }
  if (mentionsEden(text)) return { addressed: true, worth: true }
  const hasKw = EDEN_KEYWORDS.test(text)
  const isQuestion = /\?/.test(text)
  const isSubstantive = text.trim().length > 40
  return { addressed: false, worth: hasKw || isQuestion || isSubstantive }
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
  { cmd: '/profile',  args: '<fact>',  desc: 'Teach EDEN a fact about yourself' },
  { cmd: '/people',   args: '',        desc: 'Who EDEN has met in this workspace' },
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

  if (cmd === '/profile') {
    if (!rest) return pushBotMessage('Usage: `/profile <fact about yourself>` — e.g. `/profile I lead hardware integration and prefer Rust`.')
    await addProfileFact({
      userId: user.id,
      userName: user.fullName || user.firstName || 'Member',
      fact: rest,
    })
    return pushBotMessage(`✅ Committed to your identity in the Perception Layer:\n\n> ${rest}\n\nI'll remember this permanently across sessions.`)
  }

  if (cmd === '/people') {
    const people = await getKnownPeople({ limit: 20 })
    if (!people.length) return pushBotMessage('No people captured yet. Identities are added to the Perception Layer when members sign in.')
    const lines = people.map((p, i) => {
      const first = p.content?.split('\n')[0] || 'unknown'
      const kind = p.metadata?.kind === 'identity' ? '🪪 identity' : '📌 fact'
      return `${i + 1}. \`${kind}\` ${first}`
    })
    return pushBotMessage(`**People in the EDEN workspace (Perception Layer):**\n\n${lines.join('\n')}`)
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

function MessageGroup({ messages, isBot, memoriesById, onActionFire, traces, onOpenTrace }) {
  const first = messages[0]
  const name = isBot ? 'EDEN' : first.user_name

  // Parse the envelope on the latest bot message in this group so header
  // can render the tone and an action toast can fire once.
  const latestBotEnvelope = isBot ? parseBotEnvelope(messages[messages.length - 1]?.content || '') : null

  useEffect(() => {
    if (!isBot || !latestBotEnvelope) return
    const action = latestBotEnvelope.action
    if (action && action.toLowerCase() !== 'none' && action.length > 0) {
      // The message immediately preceding this bot group is the prompt speaker.
      // We don't have it here directly, but the parent passes the full envelope.
      onActionFire?.({
        msgId: messages[messages.length - 1].id,
        action,
        envelope: latestBotEnvelope,
      })
    }
  }, [isBot, latestBotEnvelope?.action, messages[messages.length - 1]?.id])

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
          {isBot && latestBotEnvelope?.tone && <ToneBadge tone={latestBotEnvelope.tone} />}
          {isBot && latestBotEnvelope?.vibe && latestBotEnvelope.vibe.delta !== 0 && (
            <span
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-wider border ${
                latestBotEnvelope.vibe.delta > 0
                  ? 'bg-emerald-500/15 text-emerald-200 border-emerald-400/30'
                  : 'bg-rose-500/15 text-rose-200 border-rose-400/30'
              }`}
              title={latestBotEnvelope.vibe.reason}
            >
              vibe {latestBotEnvelope.vibe.delta > 0 ? '+' : ''}{latestBotEnvelope.vibe.delta}
            </span>
          )}
          {isBot && traces && traces[messages[messages.length - 1]?.id] && (
            <button
              onClick={() => onOpenTrace?.(traces[messages[messages.length - 1].id])}
              className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border border-white/15 text-white/60 hover:text-cyan-200 hover:border-cyan-400/40 transition-colors"
              title="Open full pipeline trace"
            >
              trace ↗
            </button>
          )}
          <span className="text-[11px] text-white/30 opacity-0 group-hover:opacity-100 transition-opacity">
            {formatTime(first.created_at)}
          </span>
        </div>
        <div className="space-y-1">
          {messages.map((msg) => {
            const retrieved = memoriesById?.[msg.id]
            const { image, text } = parseMessageContent(msg.content)
            // For bot messages, strip the envelope tags before rendering markdown
            const botEnv = isBot ? parseBotEnvelope(msg.content) : null
            const displayText = isBot ? (botEnv?.answer || text || msg.content) : text
            return (
              <div key={msg.id} className="text-sm text-white/80 leading-relaxed">
                {isBot && retrieved && <MemoryChips memories={retrieved} />}
                {isBot && botEnv?.plan && <PlanBlock plan={botEnv.plan} />}
                {image && (
                  <div className="mb-2 max-w-md">
                    <a href={image} target="_blank" rel="noopener noreferrer" className="block">
                      <img
                        src={image}
                        alt="attached"
                        className="rounded-lg border border-white/10 hover:border-cyan-400/30 transition-colors max-h-80 object-contain"
                      />
                    </a>
                    <div className="mt-1 text-[10px] font-mono uppercase tracking-widest text-amber-300/70 flex items-center gap-1.5">
                      <Eye size={10} /> perception layer · image
                    </div>
                  </div>
                )}
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
                    {displayText}
                  </ReactMarkdown>
                ) : text ? (
                  <span>
                    {text.split(/(@eden\b)/gi).map((part, i) =>
                      /^@eden$/i.test(part)
                        ? <span key={i} className="bg-cyan-500/15 text-cyan-300 px-1 rounded font-medium">{part}</span>
                        : <span key={i}>{part}</span>
                    )}
                  </span>
                ) : null}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ───── Metrics sparkline ─────
function Sparkline({ data, color = '#22d3ee', w = 60, h = 18 }) {
  if (!data || data.length < 2) {
    return <svg width={w} height={h}><line x1="0" y1={h-1} x2={w} y2={h-1} stroke={color} strokeOpacity="0.25"/></svg>
  }
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = Math.max(1, max - min)
  const step = w / (data.length - 1)
  const pts = data.map((v, i) => `${i * step},${h - ((v - min) / range) * (h - 2) - 1}`).join(' ')
  const last = data[data.length - 1]
  const lastX = (data.length - 1) * step
  const lastY = h - ((last - min) / range) * (h - 2) - 1
  return (
    <svg width={w} height={h}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.25" strokeOpacity="0.85"/>
      <circle cx={lastX} cy={lastY} r="1.8" fill={color}/>
    </svg>
  )
}

function MetricsStrip({ metrics }) {
  const last = (arr) => (arr.length ? arr[arr.length - 1] : null)
  const avg = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null)
  const M = [
    { key: 'retrieval', label: 'retrieval',   color: '#22d3ee', fmt: (v) => `${Math.round(v)}ms` },
    { key: 'ttft',      label: 'TTFT',        color: '#a78bfa', fmt: (v) => `${Math.round(v)}ms` },
    { key: 'total',     label: 'cog total',   color: '#a78bfa', fmt: (v) => `${(v/1000).toFixed(1)}s` },
    { key: 'tokens',    label: 'out tokens',  color: '#34d399', fmt: (v) => `${Math.round(v)}` },
  ]
  return (
    <div className="flex items-center gap-4 px-6 py-2 border-b border-white/5 bg-black/20 flex-shrink-0 overflow-x-auto">
      <span className="text-[9px] font-mono uppercase tracking-widest text-white/30 flex-shrink-0">live metrics · last {Math.max(...M.map((m) => (metrics[m.key] || []).length)) || 0}</span>
      {M.map((m) => {
        const data = metrics[m.key] || []
        const cur = last(data)
        const a = avg(data)
        return (
          <div key={m.key} className="flex items-center gap-2 flex-shrink-0">
            <span className="text-[9px] font-mono uppercase tracking-widest text-white/40">{m.label}</span>
            <Sparkline data={data} color={m.color}/>
            <span className="text-[10px] font-mono" style={{ color: m.color }}>
              {cur == null ? '—' : m.fmt(cur)}
            </span>
            {a != null && (
              <span className="text-[9px] font-mono text-white/30">μ={m.fmt(a)}</span>
            )}
          </div>
        )
      })}
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
  const [onboardingOpen, setOnboardingOpen] = useState(false)
  const [attachedImage, setAttachedImage] = useState(null) // { dataUrl, w, h, name, size }
  const [attaching, setAttaching] = useState(false)
  const fileInputRef = useRef(null)
  const [activeAction, setActiveAction] = useState(null) // ROS-2 action dispatch toast
  const lastFiredActionRef = useRef(null)
  const simBusRef = useRef(null)
  const [vibeHistory, setVibeHistory] = useState({ total: 0, count: 0, recent: [] })
  const [traces, setTraces] = useState({}) // msgId -> Trace
  const [openTrace, setOpenTrace] = useState(null)
  const [metrics, setMetrics] = useState({ retrieval: [], ttft: [], total: [], tokens: [] }) // ring buffers

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const channelRef = useRef(null)
  const typingTimeoutRef = useRef(null)
  const lastTypingBroadcastRef = useRef(0)

  // Precompute sign-in wall JSX; the actual branch happens AFTER all hooks
  // fire (hooks must run unconditionally to satisfy Rules of Hooks).
  const signInWall = (
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

  // Open the simulator action bus once so ACTION dispatches reach /sim
  useEffect(() => {
    simBusRef.current = openSimBusSender()
    return () => simBusRef.current?.close()
  }, [])

  // Perception Layer: capture identity on sign-in (once per user per device)
  useEffect(() => {
    if (!user || !isSignedIn) return
    const key = `eden:identified:${user.id}`
    if (typeof window !== 'undefined' && window.localStorage.getItem(key)) return
    captureIdentity({
      userId: user.id,
      userName: user.fullName || user.firstName || 'Member',
      email: user.primaryEmailAddress?.emailAddress,
      avatarUrl: user.imageUrl || null,
    }).then((res) => {
      if (res?.ok && typeof window !== 'undefined') {
        window.localStorage.setItem(key, '1')
        setMemoryRefresh((n) => n + 1)
      }
    })
  }, [user, isSignedIn])

  // Refresh vibe history whenever memory refreshes
  useEffect(() => {
    if (!user || !isSignedIn || !supermemoryConfigured()) return
    getVibeHistory({ userId: user.id, limit: 30 }).then(setVibeHistory)
  }, [user, isSignedIn, memoryRefresh])

  // Onboarding: show the 2-fact modal once per user/device
  useEffect(() => {
    if (!user || !isSignedIn) return
    if (typeof window === 'undefined') return
    if (!supermemoryConfigured()) return
    const done = window.localStorage.getItem(`eden:onboarded:${user.id}`)
    if (!done) setOnboardingOpen(true)
  }, [user, isSignedIn])

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

  async function handleFilePick(e) {
    const file = e.target.files?.[0]
    e.target.value = '' // reset so the same file can be picked again
    if (!file) return
    if (!file.type.startsWith('image/')) {
      console.warn('only images supported')
      return
    }
    setAttaching(true)
    try {
      const { dataUrl, w, h } = await compressImageFile(file, { maxDim: 800, quality: 0.7 })
      setAttachedImage({ dataUrl, w, h, name: file.name, size: dataUrl.length })
    } catch (err) {
      console.warn('image compress failed:', err)
    } finally {
      setAttaching(false)
    }
  }

  async function sendMessage(override) {
    const raw = (override ?? input).trim()
    const hasImage = !!attachedImage
    if ((!raw && !hasImage) || streaming || !isSignedIn) return
    if (!override) setInput('')

    // Slash command fast-path — does not broadcast as user message
    if (!hasImage && isSlashCommand(raw)) {
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

    const image = hasImage ? attachedImage.dataUrl : null
    const textContent = raw
    const content = image ? buildImageMessageContent(image, textContent) : textContent
    // Clear the attachment now so the UI resets immediately
    if (hasImage) setAttachedImage(null)

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

    // Store in Supermemory (fire-and-forget). For images we store only the
    // text caption + a marker — data URLs would blow up Supermemory tokens.
    const memoryContent = image
      ? `[image attached${textContent ? `: "${textContent}"` : ' (no caption)'}]`
      : content
    addMemory({
      content: memoryContent,
      userId: user.id,
      userName: user.fullName || user.firstName || 'Member',
      role: 'user',
      extraMetadata: image ? { has_image: true, layer: 'perception' } : {},
    }).then(() => setMemoryRefresh((n) => n + 1))

    // Engagement filter: @mention or image → always; otherwise only "worth"
    // messages pass (keyword, question, or substantive length).
    const { addressed, worth } = shouldEdenConsider(textContent, !!image)
    if (!worth) return
    // NOTE: no more user-intent pre-dispatch. EDEN decides actions itself,
    // from the conversation's meaning, via its envelope [ACTION] tag.

    setStreaming(true)
    setRetrieving(true)

    // Start the trace for this turn
    const turnStartTs = Date.now()
    const trace = {
      msgId: null, // filled in when bot message is inserted
      turnStartTs,
      input: {
        role: 'user',
        userId: user.id,
        userName: user.fullName || user.firstName || 'Member',
        email: user.primaryEmailAddress?.emailAddress,
        text: textContent,
        hasMention: addressed,
        image: image ? true : false,
        imageInfo: attachedImage ? { w: attachedImage.w, h: attachedImage.h, size: attachedImage.size } : null,
      },
      context: {},
      retrieval: {},
      prompt: {},
      cognition: {},
      action: {},
    }

    // Layer pipeline: Perception → Context → Supermemory
    setLayerStage(1)
    await new Promise((r) => setTimeout(r, 180))
    setLayerStage(2)
    await new Promise((r) => setTimeout(r, 180))
    setLayerStage(3)

    const searchQuery = stripMention(textContent) || textContent || 'image'
    const retrievalStart = performance.now()
    const [retrievedMemories, knownPeople, currentVibe] = await Promise.all([
      searchMemories({ query: searchQuery, userId: user.id, limit: 5 }),
      getKnownPeople({ limit: 12 }),
      getVibeHistory({ userId: user.id, limit: 20 }),
    ])
    const retrievalEnd = performance.now()
    setRetrieving(false)
    setLayerStage(4) // Cognitive (LLM thinking)

    const userDisplayName = user.fullName || user.firstName || 'Member'
    const userFirstName = (user.firstName || userDisplayName.split(' ')[0] || 'member').toLowerCase()
    const memoryBlock = formatMemoriesForPrompt(retrievedMemories)
    const peopleBlock = formatPeopleForPrompt(knownPeople, {
      name: userDisplayName,
      email: user.primaryEmailAddress?.emailAddress,
    })
    const vibeBlock = formatVibeForPrompt(currentVibe, userDisplayName)
    const speakerBlock = `\n\n=== CURRENT SPEAKER (PIN THIS) ===\nThe person talking to you RIGHT NOW is **${userDisplayName}** (first name: ${userFirstName}${user.primaryEmailAddress?.emailAddress ? ', email: ' + user.primaryEmailAddress.emailAddress : ''}).\nWhen you reference "me" in your answer, it's them.\nWhen you emit [ACTION] tags that need a person (e.g. "charge at X", "bring the pencil to X"), use the name "${userFirstName}" unless the message clearly targets a different teammate. Do NOT ask who is speaking — you already know.`
    const addressedBlock = addressed
      ? `\n\n=== DIRECTLY ADDRESSED ===\n${userDisplayName} used @eden or sent you an image. Engage normally — this is a direct request to you.`
      : `\n\n=== NOT DIRECTLY ADDRESSED ===\n${userDisplayName} did NOT @mention you. You are overhearing the team chat. Only chime in if you have something genuinely worth saying — a reaction, a technical correction, a joke that lands, or an observation that adds value. Most overheard messages deserve zero response.\n\nEXCEPTION: if the message is directed AT you (insults, calling you out by name, questioning you, trash-talking you) — you ARE effectively addressed. Respond normally, with attitude if warranted. Social intelligence = knowing when someone's talking about/to you even without the @.\n\nTo stay silent, emit the four envelope tags (PLAN/TONE/VIBE/ACTION) on their own lines, then end immediately. Do NOT write any prose after the tags. An empty body means 'I heard, I'm choosing not to speak.' We will suppress empty replies from the UI so you appear invisible.\n\nIf you DO speak, keep it short — one or two lines max. You're jumping into an existing conversation, not monologuing.`

    // Fill retrieval frame in trace
    trace.retrieval = {
      query: searchQuery,
      limit: 5,
      durationMs: retrievalEnd - retrievalStart,
      memories: retrievedMemories,
      people: knownPeople,
      vibe: currentVibe,
    }
    trace.context = {
      identityKnown: true,
    }

    const allMsgs = [...messages]
    if (userMsg) allMsgs.push(userMsg)

    // For prior messages, collapse our [img:...] prefix into a caption so
    // text-only history doesn't pump data URLs into the LLM context.
    const historyMsgs = allMsgs.slice(-20).map((m) => {
      const parsed = parseMessageContent(m.content)
      const text = parsed.image
        ? `[${m.user_name || 'user'} shared an image]${parsed.text ? `: ${parsed.text}` : ''}`
        : m.content
      return { role: m.role, content: text }
    })

    let conversation
    if (image) {
      // Vision-model path: only the latest user message is multimodal
      const lastIdx = historyMsgs.length - 1
      const multimodalLast = {
        role: 'user',
        content: [
          { type: 'text', text: textContent || 'Describe what you see in this image, and tie it back to anything relevant in your memory.' },
          { type: 'image_url', image_url: { url: image } },
        ],
      }
      conversation = [
        { role: 'system', content: SYSTEM_PROMPT + '\n\n' + teamPromptBlock() + speakerBlock + peopleBlock + vibeBlock + memoryBlock + addressedBlock },
        ...historyMsgs.slice(0, lastIdx),
        multimodalLast,
      ]
    } else {
      conversation = [
        { role: 'system', content: SYSTEM_PROMPT + '\n\n' + teamPromptBlock() + speakerBlock + peopleBlock + vibeBlock + memoryBlock + addressedBlock },
        ...historyMsgs,
      ]
    }

    // Prompt stats
    const systemContent = conversation[0]?.content || ''
    const totalChars = conversation.reduce((s, m) => s + (typeof m.content === 'string' ? m.content.length : JSON.stringify(m.content).length), 0)
    trace.prompt = {
      systemChars: systemContent.length,
      systemTokens: Math.ceil(systemContent.length / 4),
      historyCount: conversation.length - 1,
      totalChars,
      totalTokens: Math.ceil(totalChars / 4),
      systemPreview: systemContent.slice(0, 2000),
    }

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
    trace.msgId = tempId
    trace.cognition.model = image ? VISION_MODEL : MODEL
    const cognitionStart = performance.now()

    try {
      const { text: botContent, model: modelUsed } = await chatStreaming({
        messages: conversation,
        vision: !!image,
        temperature: 0.85,
        max_tokens: 600,
        onFirstToken: () => {
          trace.cognition.latencyToFirstToken = performance.now() - cognitionStart
          setLayerStage(5) // Planning
          setTimeout(() => setLayerStage(6), 200) // Action
        },
        onDelta: (_, full) => {
          setMessages((prev) => prev.map((m) => m.id === tempId ? { ...m, content: full } : m))
        },
      })

      trace.cognition.latencyTotal = performance.now() - cognitionStart
      trace.cognition.outputChars = botContent.length
      trace.cognition.outputTokens = Math.ceil(botContent.length / 4)
      trace.cognition.model = modelUsed

      // If not directly addressed AND the answer body is empty, stay silent:
      // remove the temp bot message and don't persist. EDEN "heard but chose
      // not to speak". Envelope-only output counts as silence.
      const earlyEnv = parseBotEnvelope(botContent)
      if (!addressed && (!earlyEnv.answer || earlyEnv.answer.trim().length === 0)) {
        setMessages((prev) => prev.filter((m) => m.id !== tempId))
        setMemoriesById((prev) => { const n = { ...prev }; delete n[tempId]; return n })
        setStreaming(false)
        console.log('[chat] unaddressed + empty body → EDEN stays silent')
        return
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
        // Finalize the trace under the persisted bot msg id
        const envParsed = parseBotEnvelope(botContent)
        console.log('[chat] envelope parsed:', {
          plan: envParsed.plan?.slice(0, 80),
          tone: envParsed.tone,
          action: envParsed.action,
          vibe: envParsed.vibe,
          answerPreview: envParsed.answer?.slice(0, 100),
          rawLength: botContent.length,
        })
        trace.cognition.envelope = envParsed
        trace.action = {
          raw: envParsed.action || null,
          dispatched: !!(envParsed.action && envParsed.action.toLowerCase() !== 'none'),
          dispatchMs: 4, // constant for broadcast bus
        }
        trace.msgId = botMsg.id
        setTraces((prev) => ({ ...prev, [botMsg.id]: trace }))

        // Update rolling metrics (last 30 of each)
        setMetrics((prev) => {
          const push = (arr, v) => {
            if (!isFinite(v)) return arr
            const next = [...arr, v]
            return next.length > 30 ? next.slice(next.length - 30) : next
          }
          return {
            retrieval: push(prev.retrieval, trace.retrieval?.durationMs),
            ttft:      push(prev.ttft,      trace.cognition?.latencyToFirstToken),
            total:     push(prev.total,     trace.cognition?.latencyTotal),
            tokens:    push(prev.tokens,    trace.cognition?.outputTokens),
          }
        })

        // Publish the trace on the ROS-style topic bus so any subscriber
        // (e.g. the Sim) can see the cognition that led to this action.
        rosPublish('/eden/cognition/trace', {
          _type: 'eden_msgs/Trace',
          header: { stamp: Date.now(), frame_id: 'eden' },
          msg_id: botMsg.id,
          input: trace.input,
          retrieval: {
            query: trace.retrieval?.query,
            durationMs: trace.retrieval?.durationMs,
            memories: (trace.retrieval?.memories || []).length,
            people: (trace.retrieval?.people || []).length,
            vibe_total: trace.retrieval?.vibe?.total ?? null,
          },
          cognition: {
            model: trace.cognition?.model,
            ttft_ms: trace.cognition?.latencyToFirstToken,
            total_ms: trace.cognition?.latencyTotal,
            tokens: trace.cognition?.outputTokens,
          },
          planning: {
            tone: envParsed.tone,
            vibe_delta: envParsed.vibe?.delta || 0,
            action: envParsed.action || 'none',
          },
          action_dispatched: trace.action.dispatched,
        })
        // Also store the bot's response as a memory (channel-wide).
        // Strip envelope so Supermemory only keeps the clean answer.
        const envelopeParsed = parseBotEnvelope(botContent)
        const memoryBody = envelopeParsed.answer || botContent
        addMemory({
          content: memoryBody,
          userId: user.id,
          userName: 'EDEN',
          role: 'assistant',
          extraMetadata: {
            replying_to: searchQuery.slice(0, 80),
            tone: envelopeParsed.tone || null,
            action: envelopeParsed.action || null,
            vibe_delta: envelopeParsed.vibe?.delta || null,
          },
        }).then(() => setMemoryRefresh((n) => n + 1))

        // Commit the vibe delta for this user
        if (envelopeParsed.vibe && envelopeParsed.vibe.delta !== 0) {
          addVibe({
            userId: user.id,
            userName: userDisplayName,
            delta: envelopeParsed.vibe.delta,
            reason: envelopeParsed.vibe.reason || '',
          }).then(() => setMemoryRefresh((n) => n + 1))
        }
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

  // All hooks declared above — now we can branch the render safely
  if (isLoaded && !isSignedIn) return signInWall

  return (
    <div className="flex h-screen bg-bg-primary text-white overflow-hidden">
      <OnboardingModal
        open={onboardingOpen}
        user={user}
        onClose={() => { setOnboardingOpen(false); setMemoryRefresh((n) => n + 1) }}
      />
      <ActionDispatchToast action={activeAction} onDone={() => setActiveAction(null)} />
      {openTrace && <TraceDrawer trace={openTrace} onClose={() => setOpenTrace(null)} />}

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
            <>
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
              {memoryEnabled && (() => {
                const mood = vibeLabel(vibeHistory.total)
                return (
                  <div className={`mt-2 flex items-center gap-1.5 px-2 py-1 rounded-md border ${mood.border} ${mood.bg} ${mood.tone}`}
                    title={vibeHistory.recent.map((r) => `${r.delta > 0 ? '+' : ''}${r.delta}: ${r.reason}`).join('\n') || 'no history yet'}>
                    <span className="text-sm leading-none">{mood.emoji}</span>
                    <span className="text-[10px] font-mono uppercase tracking-widest flex-1 truncate">EDEN {mood.label}</span>
                    <span className={`text-[10px] font-mono ${vibeHistory.total > 0 ? 'text-emerald-300' : vibeHistory.total < 0 ? 'text-rose-300' : 'text-white/40'}`}>
                      {vibeHistory.total > 0 ? '+' : ''}{vibeHistory.total}
                    </span>
                  </div>
                )
              })()}
            </>
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
            <Link
              to="/sim"
              target="_blank"
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-rose-400/30 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20 transition-all"
              title="Open simulator in new tab"
            >
              <Cpu size={13} />
              Sim
            </Link>
            <button
              onClick={() => setMemoryPanelOpen(!memoryPanelOpen)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border transition-all ${memoryPanelOpen ? 'bg-cyan-500/20 border-cyan-400/40 text-cyan-200' : 'border-white/10 text-white/60 hover:border-white/30 hover:text-white'}`}
            >
              <Brain size={13} />
              Memory
            </button>
          </header>

          {/* Live metrics strip */}
          <MetricsStrip metrics={metrics} />

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
                    <MessageGroup
                      messages={group}
                      isBot={isBot}
                      memoriesById={memoriesById}
                      traces={traces}
                      onOpenTrace={setOpenTrace}
                      onActionFire={({ msgId, action, envelope }) => {
                        if (lastFiredActionRef.current === msgId) return
                        lastFiredActionRef.current = msgId
                        setActiveAction(action)
                        // The preceding group is the user who prompted this response
                        const gIdx = grouped.findIndex((g) => g.group[g.group.length - 1].id === msgId)
                        const prior = gIdx > 0 ? grouped[gIdx - 1] : null
                        const speaker = prior && !prior.isBot ? prior.group[0].user_name : null
                        simBusRef.current?.broadcast(action, {
                          msgId,
                          source: 'eden',
                          plan: envelope?.plan || null,
                          tone: envelope?.tone || null,
                          vibe: envelope?.vibe?.delta ?? null,
                          speaker,
                        })
                      }}
                    />
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
              <div className="flex flex-col gap-2 bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus-within:border-cyan-400/30 transition-colors">
                {attachedImage && (
                  <div className="flex items-center gap-3 p-2 rounded-lg bg-amber-400/5 border border-amber-400/20">
                    <img src={attachedImage.dataUrl} alt="preview" className="w-14 h-14 rounded-md object-cover flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-amber-300/80 mb-1">
                        <Eye size={10} /> perception layer · vision input
                      </div>
                      <p className="text-xs text-white/60 truncate">{attachedImage.name}</p>
                      <p className="text-[10px] text-white/30 font-mono">{attachedImage.w}×{attachedImage.h} · {Math.round(attachedImage.size / 1024)} KB</p>
                    </div>
                    <button
                      onClick={() => setAttachedImage(null)}
                      className="text-white/40 hover:text-white flex-shrink-0"
                      title="Remove"
                    >
                      <X size={14} />
                    </button>
                  </div>
                )}
                <div className="flex items-end gap-3">
                  <input
                    type="file"
                    accept="image/*"
                    ref={fileInputRef}
                    onChange={handleFilePick}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={streaming || attaching || !!attachedImage}
                    className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-white/50 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    title="Attach image (Perception layer)"
                  >
                    {attaching ? <Loader2 size={14} className="animate-spin" /> : <Paperclip size={14} />}
                  </button>
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder={attachedImage ? '@eden — what do you see?' : 'Message #eden-bot  ·  @eden to invoke  ·  / for commands'}
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
                    onClick={() => sendMessage()}
                    disabled={(!input.trim() && !attachedImage) || streaming || !isSignedIn}
                    className="flex-shrink-0 w-8 h-8 rounded-lg bg-white text-black flex items-center justify-center hover:bg-white/90 transition-colors disabled:opacity-20 disabled:cursor-not-allowed"
                  >
                    {streaming ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                  </button>
                </div>
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
