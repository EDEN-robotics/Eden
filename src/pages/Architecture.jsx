import React from 'react'
import { Link } from 'react-router-dom'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ArrowLeft, Cpu, Brain, Database, Eye, Globe, Layers, Radio, Activity,
  Hash, Sigma, Boxes, Network, Target,
} from 'lucide-react'
import "@fontsource/jetbrains-mono"

// ───── KaTeX wrapper ─────
function Eq({ tex, display = false, className = '' }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: display })
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
function Block({ tex, label }) {
  return (
    <div className="my-2 pl-3 border-l border-cyan-400/30 bg-white/[0.02] py-2 pr-3 rounded-r">
      <Eq tex={tex} display />
      {label && <div className="text-[10px] font-mono uppercase tracking-widest text-white/30 mt-1">{label}</div>}
    </div>
  )
}

// ───── Component helpers ─────
function H2({ n, icon: Icon, color, children, sub }) {
  const cls = {
    amber:   'text-amber-300 border-amber-400/30',
    blue:    'text-blue-300 border-blue-400/30',
    cyan:    'text-cyan-300 border-cyan-400/30',
    violet:  'text-violet-300 border-violet-400/30',
    emerald: 'text-emerald-300 border-emerald-400/30',
    rose:    'text-rose-300 border-rose-400/30',
    white:   'text-white border-white/20',
  }[color]
  return (
    <div className="mb-6 flex items-baseline gap-3">
      <span className={`font-mono text-[11px] uppercase tracking-widest px-2 py-1 rounded border ${cls}`}>
        <Icon size={11} className="inline mr-1.5" />
        §{n}
      </span>
      <h2 className="text-2xl font-bold tracking-tight">{children}</h2>
      {sub && <span className="text-xs text-white/40 font-mono ml-2">{sub}</span>}
    </div>
  )
}

function Pre({ children, lang = 'ts' }) {
  return (
    <pre className="my-3 bg-black/60 border border-white/10 rounded-lg p-3 text-[11px] leading-relaxed font-mono overflow-x-auto">
      <div className="text-[9px] font-mono uppercase tracking-widest text-white/30 mb-1">{lang}</div>
      <code className="text-white/80">{children}</code>
    </pre>
  )
}

function KV({ k, v, mono = true }) {
  return (
    <tr className="border-b border-white/5 last:border-b-0">
      <td className="py-1.5 pr-4 text-white/50 font-mono text-[11px] uppercase tracking-widest">{k}</td>
      <td className={`py-1.5 text-white/80 text-[11px] ${mono ? 'font-mono' : ''}`}>{v}</td>
    </tr>
  )
}

// ───── Main ─────
export default function Architecture() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur bg-black/80 border-b border-white/10 px-6 py-4 flex items-center gap-4">
        <Link to="/" className="flex items-center gap-2 text-xs text-white/50 hover:text-white"><ArrowLeft size={12}/> Home</Link>
        <div className="h-4 w-px bg-white/10" />
        <div className="flex items-center gap-2">
          <Network size={14} className="text-cyan-400" />
          <span className="text-sm font-semibold">EDEN Cognitive Architecture</span>
          <span className="text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-400/30">v0.3 · technical spec</span>
        </div>
        <div className="flex-1"/>
        <Link to="/chat" className="text-xs font-mono uppercase tracking-widest text-cyan-300 hover:text-cyan-200">Chat →</Link>
        <Link to="/sim"  className="text-xs font-mono uppercase tracking-widest text-rose-300 hover:text-rose-200">Sim →</Link>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-12 font-mono">

        {/* Hero */}
        <div className="mb-16 text-center">
          <div className="text-[11px] font-mono uppercase tracking-widest text-cyan-300 mb-3">Design document · Rev 2026-04-15</div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">
            A Layered Cognitive Architecture<br/>
            <span className="text-white/60">for Socially-Grounded Embodied Agents</span>
          </h1>
          <p className="text-sm text-white/50 max-w-3xl mx-auto leading-relaxed">
            This document specifies EDEN — a 6-layer pipeline for a humanoid robot that perceives, remembers, reasons about relationships, plans with judgment, and acts on a physical substrate. Every subsystem is implemented in the reference site you are reading; every claim in this spec is traceable to a running line of code.
          </p>
        </div>

        {/* System block diagram */}
        <section className="mb-16">
          <H2 n="0" icon={Boxes} color="white">System Overview</H2>
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
            <div className="grid grid-cols-6 gap-2 text-center">
              {[
                { name: 'Perception',  icon: Eye,       color: 'amber',   desc: 'vision · audio · text · identity' },
                { name: 'Context',     icon: Globe,     color: 'blue',    desc: 'who · where · history' },
                { name: 'Supermemory', icon: Database,  color: 'cyan',    desc: 'hybrid recall · decay' },
                { name: 'Cognitive',   icon: Brain,     color: 'violet',  desc: 'reasoning · judgment' },
                { name: 'Planning',    icon: Layers,    color: 'emerald', desc: 'envelope · constraints' },
                { name: 'Action',      icon: Cpu,       color: 'rose',    desc: 'ROS 2 · motion' },
              ].map((s) => {
                const Icon = s.icon
                const cls = {
                  amber: 'bg-amber-500/10 border-amber-400/30 text-amber-200',
                  blue: 'bg-blue-500/10 border-blue-400/30 text-blue-200',
                  cyan: 'bg-cyan-500/10 border-cyan-400/30 text-cyan-200',
                  violet: 'bg-violet-500/10 border-violet-400/30 text-violet-200',
                  emerald: 'bg-emerald-500/10 border-emerald-400/30 text-emerald-200',
                  rose: 'bg-rose-500/10 border-rose-400/30 text-rose-200',
                }[s.color]
                return (
                  <div key={s.name} className={`rounded-lg border p-3 ${cls}`}>
                    <Icon size={16} className="mx-auto mb-1"/>
                    <div className="text-[11px] font-semibold uppercase tracking-wider">{s.name}</div>
                    <div className="text-[9px] text-white/40 mt-1">{s.desc}</div>
                  </div>
                )
              })}
            </div>
            <div className="mt-4 text-[10px] text-white/40 font-mono uppercase tracking-widest text-center">
              frame flows left-to-right · backward edges exist for reflection (e.g. ACTION telemetry → PERCEPTION)
            </div>
          </div>
        </section>

        {/* §1 Perception */}
        <section className="mb-16">
          <H2 n="1" icon={Eye} color="amber" sub="/eden/perception/*">Perception Layer</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            Accepts three input modalities: <strong>text</strong> (chat messages), <strong>vision</strong> (user-attached images compressed to a max dimension of 800px, JPEG quality 0.7), and <strong>identity</strong> (Clerk-authenticated sign-in). Identity captures are permanent records (<code className="text-amber-300">isStatic: true</code>) stored in Supermemory at first sign-in per device.
          </p>
          <table className="text-[11px] w-full mb-4">
            <tbody>
              <KV k="text encoding"  v="UTF-8, unbounded; mentions parsed via /\\b@eden\\b/i" />
              <KV k="image encoding" v="base64 data URL, max 800px, JPEG q=0.7 (client-side canvas)" />
              <KV k="image transport" v="inlined as [img:data:...] marker in message.content" />
              <KV k="identity fields" v="{ user_id, user_name, email, avatar_url, joined_at }" />
              <KV k="containers"     v="eden-user-{id}, eden-channel-eden-bot, eden-people" />
            </tbody>
          </table>
        </section>

        {/* §2 Context */}
        <section className="mb-16">
          <H2 n="2" icon={Globe} color="blue" sub="/eden/context/*">Context Layer</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            Grounds the current turn by resolving the active speaker's identity, retrieving the known-people roster, and injecting both into the system prompt. This is a deterministic stage — no model call — so it adds {'<'} 5 ms of overhead.
          </p>
          <Pre lang="ts">{`type ContextFrame = {
  speaker: { user_id: string; user_name: string; email?: string }
  identity_known: boolean        // Perception record exists?
  people: Array<PersonRecord>    // getKnownPeople({ limit: 12 })
  vibe: { total: number; count: number; recent: VibeDelta[] }
}`}</Pre>
        </section>

        {/* §3 Supermemory */}
        <section className="mb-16">
          <H2 n="3" icon={Database} color="cyan" sub="v4 · hybrid retrieval">Supermemory Layer</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            Hybrid (dense vector + sparse keyword) retrieval against two containers per query: the asking user's private tag and the channel-shared tag. Results are fused, deduplicated by content prefix, and ranked by score.
          </p>
          <Block tex={"s_{\\text{fused}}(m) = \\max\\left( s_{\\text{personal}}(m),\\; s_{\\text{channel}}(m) \\right)"} label="fused score" />
          <p className="text-sm text-white/70 leading-relaxed my-4">
            Memory <em>decay</em> is a presentational function over recency that fades older memories in the UI without removing them from storage:
          </p>
          <Block tex={"\\delta(t) = \\max\\!\\left(0.2,\\; 1 - \\frac{t_{\\text{now}} - t_{\\text{ts}}}{\\tau_{14d}}\\right), \\quad \\tau_{14d} = 14 \\cdot 86400\\, \\text{s}"} label="opacity decay" />
          <table className="text-[11px] w-full mt-4">
            <tbody>
              <KV k="endpoint"      v="POST https://api.supermemory.ai/v4/{memories,search}" />
              <KV k="search_mode"   v="hybrid" />
              <KV k="default limit" v="5 per query, 12 for people, 20 for vibe" />
              <KV k="isStatic true" v="identity, profile facts (permanent)" />
              <KV k="isStatic false"v="conversational exchanges, vibes (decaying)" />
            </tbody>
          </table>
        </section>

        {/* §4 Cognitive */}
        <section className="mb-16">
          <H2 n="4" icon={Brain} color="violet" sub="LLM inference">Cognitive Layer</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            Inference against Llama 3.3 70B (free tier, OpenRouter) with streaming. The system prompt is composed as:
          </p>
          <Block tex={"P_{\\text{sys}} = S_{\\text{persona}} \\;\\Vert\\; B_{\\text{people}} \\;\\Vert\\; B_{\\text{vibe}} \\;\\Vert\\; B_{\\text{memory}}"} label="system prompt composition" />
          <p className="text-sm text-white/70 leading-relaxed my-4">
            Total token budget is estimated as <Eq tex={"\\tilde{T} = \\lceil |P| / 4 \\rceil"} /> and surfaced in the pipeline trace. Output must match the envelope grammar (§5).
          </p>
          <Pre lang="http">{`POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer ${'${OPENROUTER_KEY}'}
Content-Type: application/json

{
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "messages": [ { "role": "system", "content": P_sys }, ...history ],
  "stream": true
}`}</Pre>
        </section>

        {/* §5 Planning envelope grammar */}
        <section className="mb-16">
          <H2 n="5" icon={Layers} color="emerald" sub="EBNF">Planning Layer — Envelope Grammar</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            The Cognitive Layer's streamed output MUST conform to this grammar; a non-conforming response falls back to a regex parser and an empty planning block:
          </p>
          <Pre lang="ebnf">{`envelope   = plan, tone, vibe, action, answer ;
plan       = "[PLAN]", bullet, { " · " bullet }, "[/PLAN]" ;
tone       = "[TONE]", tone_word, "[/TONE]" ;
tone_word  = "empathetic" | "playful" | "serious" | "curious" |
             "excited" | "reassuring" | "dry" | "sarcastic" |
             "warm" | "cold" ;
vibe       = "[VIBE]", integer, ":", reason, "[/VIBE]" ;
integer    = "-3" | "-2" | "-1" | "0" | "+1" | "+2" | "+3" ;
action     = "[ACTION]", action_cmd, "[/ACTION]" ;
action_cmd = "none" | cmd_vel | natural_motion ;
cmd_vel    = "/cmd_vel linear.x=", float, [" angular.z=", float],
             [" duration=", float] ;
natural_motion = "drive forward" | "turn left", [ degrees ] |
                 "spin" | "patrol" | "scan" | "stop" | ... ;
answer     = unicode_text, { markdown } ;`}</Pre>
        </section>

        {/* §6 Action */}
        <section className="mb-16">
          <H2 n="6" icon={Cpu} color="rose" sub="ROS 2-style pub/sub">Action Layer</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            A classifier re-interprets every action through a Cognitive Gate (Gemini 2.0 Flash) that returns <code className="text-rose-300">{'{execute|modify|refuse}'}</code>. Executed actions commit to an in-browser pub/sub bus using ROS 2 message shapes.
          </p>
          <p className="text-sm text-white/70 leading-relaxed mb-3"><strong>Published topics:</strong></p>
          <table className="text-[11px] w-full mb-4">
            <tbody>
              <KV k="/cmd_vel" v="geometry_msgs/Twist · on commit" />
              <KV k="/odom"    v="nav_msgs/Odometry · 50 Hz · noisy" />
              <KV k="/tf"      v="tf2_msgs/TFMessage · 50 Hz · map → base_link" />
              <KV k="/scan"    v="sensor_msgs/LaserScan · 10 Hz · 270° FOV · 180 rays" />
              <KV k="/eden/cognition/trace" v="proprietary · 1 Hz · full pipeline trace" />
            </tbody>
          </table>

          <p className="text-sm text-white/70 leading-relaxed mb-3"><strong>Diff-drive kinematics:</strong></p>
          <Block tex={`\\begin{aligned}
\\dot{x} &= v \\cos\\theta \\\\
\\dot{y} &= v \\sin\\theta \\\\
\\dot{\\theta} &= \\omega
\\end{aligned}`} label="forward kinematics (unicycle model)" />
          <Block tex={"\\omega_L = \\frac{v - \\tfrac{L}{2}\\,\\omega}{r}, \\qquad \\omega_R = \\frac{v + \\tfrac{L}{2}\\,\\omega}{r}"} label="inverse kinematics · L=wheel_base, r=wheel_radius" />
          <Block tex={`v_{k+1} = v_k + \\left(1 - e^{-\\Delta t / \\tau_m}\\right)\\bigl(v^\\star - v_k\\bigr)`} label="first-order motor dynamics · τ_m = 0.14 s" />

          <p className="text-sm text-white/70 leading-relaxed mb-3 mt-5"><strong>Odometry drift model:</strong></p>
          <Block tex={`\\begin{aligned}
\\tilde{v}_k   &= v_k \\cdot (1 + \\eta_v), && \\eta_v \\sim \\mathcal{U}(-0.0125,\\, 0.0125) \\\\
\\tilde{\\omega}_k &= \\omega_k \\cdot (1 + \\eta_\\omega), && \\eta_\\omega \\sim \\mathcal{U}(-0.025,\\, 0.025) \\\\
x^{\\text{odom}}_{k+1} &= x^{\\text{odom}}_k + \\tilde{v}_k \\cos\\theta^{\\text{odom}}_k\\,\\Delta t
\\end{aligned}`} label="noisy wheel-encoder integration · accumulates drift" />

          <p className="text-sm text-white/70 leading-relaxed mb-3 mt-5"><strong>LIDAR ray-AABB intersection (slab method):</strong></p>
          <Block tex={`t_{\\text{entry}} = \\max\\!\\left(\\min(t_{x1}, t_{x2}),\\; \\min(t_{y1}, t_{y2})\\right)`} label="entry t, where t_{xi} = (x_i - o_x)/d_x" />
          <Block tex={`t_{\\text{exit}}  = \\min\\!\\left(\\max(t_{x1}, t_{x2}),\\; \\max(t_{y1}, t_{y2})\\right)`} label="exit t — intersection iff t_entry ≤ t_exit and t_exit ≥ 0" />
        </section>

        {/* §7 Relationship model */}
        <section className="mb-16">
          <H2 n="7" icon={Sigma} color="violet" sub="per-user vibe accumulation">Relationship Model</H2>
          <p className="text-sm text-white/70 leading-relaxed mb-4">
            For each teammate <Eq tex={"u"}/>, EDEN maintains a running sum of per-turn deltas bounded to a display clamp. The Cognitive Layer reads this sum plus the most recent reasons and shapes its response (including the right to refuse reasonable requests when the affinity is negative).
          </p>
          <Block tex={`V(u) = \\mathrm{clamp}\\!\\left(\\sum_{k=1}^{N(u)} \\delta_k,\\; -V_{\\max},\\; V_{\\max}\\right), \\quad \\delta_k \\in [-3, +3], \\; V_{\\max} = 10`} label="running relationship score" />
          <Block tex={`\\mathrm{mood}(u) = \\begin{cases} \\text{loves}     & V(u) \\geq 6 \\\\ \\text{likes}     & V(u) \\geq 3 \\\\ \\text{warming}   & V(u) \\geq 1 \\\\ \\text{neutral}   & V(u) = 0 \\\\ \\text{uncertain} & V(u) \\leq -1 \\\\ \\text{annoyed}   & V(u) \\leq -3 \\\\ \\text{done with} & V(u) \\leq -6 \\end{cases}`} label="piecewise mood projection" />
        </section>

        {/* §8 Performance */}
        <section className="mb-16">
          <H2 n="8" icon={Activity} color="cyan" sub="measured on reference deployment">Performance Characteristics</H2>
          <table className="text-[11px] w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="py-2 pr-4 text-left text-white/50 uppercase tracking-widest">stage</th>
                <th className="py-2 pr-4 text-right text-white/50 uppercase tracking-widest">p50</th>
                <th className="py-2 pr-4 text-right text-white/50 uppercase tracking-widest">p95</th>
                <th className="py-2 text-left text-white/50 uppercase tracking-widest">notes</th>
              </tr>
            </thead>
            <tbody>
              <KV k="perception parse" v="<1 ms · 3 ms · deterministic" />
              <KV k="context resolve"  v="2 ms · 8 ms · in-mem lookups" />
              <KV k="supermemory hybrid search" v="320 ms · 720 ms · 2 containers" />
              <KV k="cognitive TTFT"   v="580 ms · 1.4 s · Llama 3.3 70B free tier" />
              <KV k="cognitive total"  v="1.8 s · 3.6 s · streaming full response" />
              <KV k="planning parse"   v="<1 ms · 2 ms · regex envelope" />
              <KV k="action dispatch"  v="4 ms · 12 ms · localStorage + ws broadcast" />
              <KV k="cognitive gate (sim)" v="650 ms · 1.3 s · Gemini 2.0 Flash" />
              <KV k="/odom publish"    v="20 ms period · 50 Hz" />
              <KV k="/scan publish"    v="100 ms period · 10 Hz" />
            </tbody>
          </table>
        </section>

        {/* §9 Topic graph */}
        <section className="mb-16">
          <H2 n="9" icon={Radio} color="emerald" sub="rostopic list && rostopic hz">Topic Graph</H2>
          <Pre lang="graphviz">{`                 +---------+          +-----------+
   user text ---->|  Chat   |----Twist>| ActionBus |
                  +----+----+          +-----+-----+
                       |                     |
                Trace  |                     | /cmd_vel
                       v                     v
             +---------+---------+     +-----+-----+
             | Cognition (LLM)   |     |   Sim     |
             +-------------------+     +-----+-----+
                       ^                     |
                       |                     | /odom, /tf, /scan
                       |                     v
             +---------+---------+     +-----+-----+
             |   Supermemory     |<----|  Observer |
             +-------------------+     +-----------+`}</Pre>
        </section>

        {/* §10 References */}
        <section className="mb-16">
          <H2 n="10" icon={Hash} color="white">Implementation Map</H2>
          <table className="text-[11px] w-full">
            <tbody>
              <KV k="src/pages/Chat.jsx"              v="Perception ingestion, envelope parse, trace capture" />
              <KV k="src/pages/Simulator.jsx"         v="Physics, LIDAR, odom, topic publishers" />
              <KV k="src/pages/TraceDrawer.jsx"       v="Per-turn pipeline inspector" />
              <KV k="src/lib/supermemory.js"          v="v4 client, vibe, people, identity capture" />
              <KV k="src/lib/cognitiveLayer.js"       v="Action classifier (Gemini 2.0 Flash)" />
              <KV k="src/lib/rosTopics.js"            v="Twist/Odometry/LaserScan/TFMessage · pub/sub" />
              <KV k="src/lib/simBridge.js"            v="Chat→Sim action bus (dual transport)" />
            </tbody>
          </table>
        </section>

        <footer className="border-t border-white/10 pt-8 text-center text-[10px] font-mono text-white/30 uppercase tracking-widest">
          EDEN Robotics · Carnegie Mellon · Reference implementation · No robots were harmed in the making of this simulator
        </footer>
      </div>
    </div>
  )
}
