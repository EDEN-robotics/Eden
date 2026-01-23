import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import {
  Brain,
  Cpu,
  Database,
  Eye,
  Layers,
  ArrowRight,
  Github,
  ChevronRight,
  Activity,
  Zap,
  Globe,
  Clock,
  ShieldCheck,
  Code,
  Linkedin,
  Twitter,
  Mail,
  Star,
  GitFork,
  Users,
  TrendingUp
} from 'lucide-react';
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/800.css";
import "@fontsource/jetbrains-mono";

// --- Components ---

const Nav = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 flex justify-end items-center px-8 py-6 backdrop-blur-sm bg-black/10 transition-all border-b border-white/5">
    <div className="flex gap-8 text-xs font-mono uppercase tracking-widest text-text-dim">
      <a href="#architecture" className="hover:text-white transition-colors">Architecture</a>
      <a href="#walkthrough" className="hover:text-white transition-colors">How it works</a>
      <a href="#roadmap" className="hover:text-white transition-colors">Roadmap</a>
      <a href="#github" className="flex items-center gap-2 hover:text-white transition-colors border-l border-white/10 pl-8">
        <Github size={14} /> GitHub
      </a>
    </div>

  </nav>
);

const SectionHeading = ({ number, title, subtitle }) => (
  <div className="mb-16">
    <div className="flex items-center gap-4 mb-2">
      <span className="font-mono text-accent text-sm tracking-tighter">[{number}]</span>
      <div className="h-px flex-1 bg-white/10" />
    </div>
    <h2 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">{title}</h2>
    {subtitle && <p className="text-text-dim text-lg max-w-2xl">{subtitle}</p>}
  </div>
);

const AnimatedArchitecture = () => {
  const [activeLayer, setActiveLayer] = useState(0);
  const layers = [
    { name: "Input Layer", desc: "Sensory data from Jetson Nano (Vision, Audio)", icon: <Eye size={20} /> },
    { name: "Context Layer", desc: "User identity, behavioral cues, situational state", icon: <Globe size={20} /> },
    { name: "Cognitive Layer", desc: "Distributed reasoning and emotional evaluation", icon: <Brain size={20} /> },
    { name: "Planning Layer", desc: "Trajectory and social-alignment generation", icon: <Layers size={20} /> },
    { name: "Action Layer", desc: "Low-latency ROS 2 motion execution", icon: <Cpu size={20} /> }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center glass-panel p-8 md:p-12 mb-32">
      <div className="flex flex-col gap-4">
        {layers.map((layer, i) => (
          <div
            key={i}
            onMouseEnter={() => setActiveLayer(i)}
            className={`p-6 rounded-xl transition-all cursor-default border ${activeLayer === i ? 'bg-accent/10 border-accent/30 translate-x-2' : 'hover:bg-white/5 border-transparent'}`}
          >
            <div className={`flex items-center gap-4 mb-2 ${activeLayer === i ? 'text-accent' : 'text-text-dim'}`}>
              {layer.icon}
              <span className="font-mono text-sm uppercase italic">{layer.name}</span>
            </div>
            <p className="text-sm text-text-dim leading-relaxed">{layer.desc}</p>
          </div>
        ))}
      </div>
      <div className="relative aspect-square flex items-center justify-center p-8 bg-black/40 rounded-2xl border border-white/5 overflow-hidden">
        <div className="absolute inset-0 subtle-grid opacity-20" />
        <svg viewBox="0 0 100 100" className="w-full h-full relative z-10">
          {layers.map((_, i) => (
            <motion.circle
              key={i}
              cx="50" cy="50"
              r={15 + i * 8}
              fill="none"
              stroke={activeLayer === i ? '#f8fafc' : '#ffffff20'}
              strokeWidth="0.5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />
          ))}
          <motion.circle
            cx="50" cy="50" r="10"
            fill="#f8fafc"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
            transition={{ repeat: Infinity, duration: 3 }}
          />
        </svg>
      </div>
    </div>
  );
};

const Walkthrough = () => {
  const steps = [
    {
      id: "perception",
      title: "Perception & Input",
      text: "EDEN processes live video and sensor streams on the Jetson Nano, performing real-time human action recognition and environment mapping.",
      icon: <Eye className="text-white" />
    },
    {
      id: "context",
      title: "Context Gathering",
      text: "The system identifies the user and gathers situational markers—cues like voice tone, spatial distance, and previous engagement level.",
      icon: <Globe className="text-slate-300" />
    },
    {
      id: "cognition",
      title: "Cognitive Layer",
      text: "Running on the Host PC, EDEN evaluates the current emotional state and compares it with the accessed goal-state for the interaction.",
      icon: <Brain className="text-slate-400" />
    },
    {
      id: "supermemory",
      title: "Supermemory Layer",
      text: "EDEN's unique advantage: a long-term memory graph that stores past decisions, user behavioral history, and outcomes to shape future responses.",
      icon: <Database className="text-white" />
    },
    {
      id: "planning",
      title: "Planning Layer",
      text: "The planner optimizes for both task success and human alignment, ensuring motion is socially appropriate and predictive of human needs.",
      icon: <Layers className="text-slate-300" />
    },
    {
      id: "action",
      title: "Action Layer",
      text: "Finally, the decision is streamed back to the Jetson via WebSockets, triggering precise hardware actuation through ROS 2.",
      icon: <Activity className="text-white" />
    }
  ];

  return (
    <div id="walkthrough" className="relative py-32">
      <div className="absolute left-[39px] top-0 bottom-0 w-px bg-white/10 hidden md:block" />
      {steps.map((step, i) => (
        <motion.div
          key={step.id}
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: i * 0.1 }}
          viewport={{ once: true }}
          className="relative pl-0 md:pl-24 mb-24 last:mb-0"
        >
          <div className="absolute left-[-10px] md:left-[30px] top-0 w-5 h-5 rounded-full bg-accent border-4 border-bg-primary hidden md:block z-10" />
          <div className="glass-panel p-8 md:p-10 hover:border-accent/20 transition-all group">
            <div className="flex items-center gap-6 mb-6">
              <div className="p-4 rounded-xl bg-white/5 group-hover:bg-accent/10 transition-colors">
                {React.cloneElement(step.icon, { size: 32 })}
              </div>
              <div>
                <span className="font-mono text-xs text-accent uppercase tracking-widest mb-1 block">Phase 0{i + 1}</span>
                <h3 className="text-2xl font-bold">{step.title}</h3>
              </div>
            </div>
            <p className="text-text-dim text-lg leading-relaxed max-w-3xl">{step.text}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

const GitHubStats = () => {
  const [stats, setStats] = useState({
    stars: 0,
    forks: 0,
    watchers: 0,
    loading: true
  });

  useEffect(() => {
    // Replace 'EDEN-robotics/Eden' with your actual GitHub repo
    fetch('https://api.github.com/repos/EDEN-robotics/Eden')
      .then(res => res.json())
      .then(data => {
        setStats({
          stars: data.stargazers_count || 0,
          forks: data.forks_count || 0,
          watchers: data.subscribers_count || 0,
          loading: false
        });
      })
      .catch(() => {
        setStats(prev => ({ ...prev, loading: false }));
      });
  }, []);

  const statItems = [
    { label: 'Stars', value: stats.stars, icon: <Star size={24} />, color: 'text-yellow-400' },
    { label: 'Forks', value: stats.forks, icon: <GitFork size={24} />, color: 'text-blue-400' },
    { label: 'Watchers', value: stats.watchers, icon: <Users size={24} />, color: 'text-green-400' }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
      {statItems.map((item, i) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: i * 0.1 }}
          viewport={{ once: true }}
          className="glass-panel p-8 text-center group hover:border-white/20 transition-all"
        >
          <div className={`${item.color} mb-4 flex justify-center group-hover:scale-110 transition-transform`}>
            {item.icon}
          </div>
          <div className="text-4xl font-bold mb-2">
            {stats.loading ? '...' : item.value.toLocaleString()}
          </div>
          <div className="text-text-dim text-sm font-mono uppercase tracking-widest">{item.label}</div>
        </motion.div>
      ))}
    </div>
  );
};

const UseCases = () => {
  const cases = [
    {
      title: "Long-Term Care Facilities",
      description: "EDEN learns patient preferences over weeks, adapting its interaction style and assistance patterns based on individual needs and emotional states.",
      icon: <Activity className="text-blue-400" />,
      tags: ["Healthcare", "Personalization"]
    },
    {
      title: "Educational Research Labs",
      description: "Universities can study human-robot interaction with a system that maintains consistent personality while adapting to different researchers and experimental contexts.",
      icon: <Brain className="text-purple-400" />,
      tags: ["Research", "Academia"]
    },
    {
      title: "Household Assistants",
      description: "Unlike traditional home robots, EDEN remembers family routines, preferences, and social dynamics—improving its helpfulness over months of interaction.",
      icon: <Globe className="text-green-400" />,
      tags: ["Consumer", "Daily Life"]
    },
    {
      title: "Social Robotics Studies",
      description: "Researchers investigating empathy, trust, and long-term HRI can leverage EDEN's memory-driven reasoning to study relationship formation.",
      icon: <TrendingUp className="text-amber-400" />,
      tags: ["HRI", "Social Science"]
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      {cases.map((useCase, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: i * 0.1 }}
          viewport={{ once: true }}
          className="glass-panel p-8 group hover:border-white/20 transition-all"
        >
          <div className="flex items-start gap-4 mb-4">
            <div className="p-3 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors">
              {React.cloneElement(useCase.icon, { size: 28 })}
            </div>
            <div className="flex-1">
              <h3 className="text-2xl font-bold mb-2">{useCase.title}</h3>
              <div className="flex gap-2 mb-4">
                {useCase.tags.map((tag, j) => (
                  <span key={j} className="text-xs font-mono px-2 py-1 rounded bg-white/5 text-text-dim">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <p className="text-text-dim leading-relaxed">{useCase.description}</p>
        </motion.div>
      ))}
    </div>
  );
};

const Team = () => {
  const teamMembers = [
    {
      name: "Vedant Soni",
      role: "Project Lead",
      image: "/team/vedant-soni.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/vedantsonimech/"
      }
    },
    {
      name: "Sebastian Chu",
      role: "Electrical Engineer",
      image: "/team/sebastian-chu.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/sebcchu/"
      }
    },
    {
      name: "Sebastian Dayer",
      role: "Mechanical Engineer",
      image: "/team/sebastian-dayer.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/sebastian-dayer/"
      }
    },
    {
      name: "Paavan Bagla",
      role: "Software Engineer",
      image: "/team/paavan-bagla.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/paavanbagla/"
      }
    },
    {
      name: "Haren Thorat",
      role: "Software Engineer",
      image: "/team/haren-thorat.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/haren-thorat/"
      }
    },
    {
      name: "William Lam",
      role: "Mechanical Engineer",
      image: "/team/william-lam.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/swwilliamlam/"
      }
    },
    {
      name: "Dillon Markentell",
      role: "Mechanical Engineer",
      image: "/team/dillon-markentell.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/dillon-markentell-0630b5350/"
      }
    },
    {
      name: "Krish Singh",
      role: "Software Engineer",
      image: "/team/krish-singh.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/krish-singh2007/"
      }
    },
    {
      name: "Juan Gomez Sandoval",
      role: "Electrical Engineer",
      image: "/team/juan-gomez-sandoval.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/juan-gomez-sandoval-24aa55239/"
      }
    },
    {
      name: "Andrew Zheng",
      role: "Software Engineer",
      image: "/team/andrew-zheng.jpg",
      socials: {
        linkedin: "#"
      }
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
      {teamMembers.map((member, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: i * 0.1 }}
          viewport={{ once: true }}
          className="group relative overflow-hidden rounded-2xl"
        >
          {/* Image */}
          <div className="aspect-square overflow-hidden bg-white/5">
            <img
              src={`${import.meta.env.BASE_URL}${member.image.replace(/^\//, '')}`}
              alt={member.name}
              className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500 group-hover:scale-110"
            />
          </div>

          {/* Overlay with socials - appears on hover */}
          <motion.div
            initial={{ opacity: 0 }}
            whileHover={{ opacity: 1 }}
            className="absolute inset-0 bg-black/90 backdrop-blur-sm flex flex-col items-center justify-center gap-4 opacity-0 group-hover:opacity-100 transition-all duration-300"
          >
            <h3 className="text-xl font-bold">{member.name}</h3>
            <p className="text-text-dim text-sm font-mono uppercase tracking-widest">{member.role}</p>
            <div className="flex gap-4 mt-4">
              <a
                href={member.socials.linkedin}
                className="w-10 h-10 rounded-full bg-white/10 hover:bg-white hover:text-black flex items-center justify-center transition-all"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Linkedin size={18} />
              </a>
            </div>
          </motion.div>
        </motion.div>
      ))}
    </div>
  );
};

export default function App() {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95]);

  return (
    <div className="min-h-screen bg-bg-primary">
      <Nav />

      {/* 1. Hero Section */}
      <section className="relative min-h-screen flex flex-col justify-center items-center px-8 pt-20 overflow-hidden">
        <div className="absolute inset-0 subtle-grid pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-accent/5 rounded-full blur-[120px] pointer-events-none" />

        <motion.div
          style={{ opacity, scale }}
          className="relative z-10 text-center max-w-5xl"
        >

          <motion.h1
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              duration: 1.2,
              delay: 0.3,
              ease: [0.16, 1, 0.3, 1] // Custom easing for dramatic effect
            }}
            className="text-7xl md:text-[14rem] font-black mb-10 leading-[0.9] tracking-[-0.04em] gradient-text"
          >
            EDEN
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.5 }}
            className="text-xl md:text-2xl text-text-dim max-w-2xl mx-auto mb-12 font-light leading-relaxed"
          >
            A humanoid robotics framework for adaptive reasoning, emotional context, and long-term memory.
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 2.0 }}
            className="flex flex-col items-center gap-4"
          >
            <div className="flex gap-4 mb-20">
              <button className="px-8 py-4 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-all flex items-center gap-3">
                Read Poster <ArrowRight size={18} />
              </button>
              <button className="px-8 py-4 bg-transparent border border-white/10 text-white font-semibold rounded-full hover:bg-white/5 transition-all">
                View Documentation
              </button>
            </div>


          </motion.div>
        </motion.div>
      </section>

      <main className="max-w-7xl mx-auto px-8 py-32">
        {/* 2. The Problem */}
        <section id="problem" className="py-32 border-t border-white/5">
          <SectionHeading
            number="01"
            title="The Problem"
            subtitle="Most humanoids today are locked in a deterministic cycle—they react, but they do not learn from the social fabric."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { title: "No Continuity", desc: "Robots treat every interaction as if it's the first.", icon: <Clock /> },
              { title: "Deterministic", desc: "Same input always leads to the exact same behavior.", icon: <Zap /> },
              { title: "Context Blind", desc: "They ignore user behavior and emotional cues.", icon: <ShieldCheck /> },
              { title: "Task Only", desc: "Optimizing for physics, not human alignment.", icon: <Code /> }
            ].map((p, i) => (
              <div key={i} className="glass-panel p-8 group">
                <div className="text-accent mb-6 bg-accent/10 w-fit p-3 rounded-xl group-hover:bg-accent group-hover:text-black transition-all">
                  {p.icon}
                </div>
                <h4 className="text-xl font-bold mb-3">{p.title}</h4>
                <p className="text-text-dim text-sm leading-relaxed">{p.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 3. The EDEN Difference */}
        <section id="difference" className="py-32">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-20 items-center">
            <div>
              <SectionHeading
                number="02"
                title="The EDEN Difference"
                subtitle="EDEN is designed to change how it responds over time, mirroring human social learning."
              />
              <div className="space-y-12">
                <div className="flex gap-6">
                  <div className="w-12 h-12 rounded-xl bg-accent/20 flex items-center justify-center shrink-0">
                    <Brain className="text-accent" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold mb-2">Evaluates Experiences</h4>
                    <p className="text-text-dim leading-relaxed">Instead of just executing, EDEN reflects on past interactions to weight its current priority list.</p>
                  </div>
                </div>
                <div className="flex gap-6">
                  <div className="w-12 h-12 rounded-xl bg-accent/10 flex items-center justify-center shrink-0">
                    <Activity className="text-accent" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold mb-2">Personality Consistency</h4>
                    <p className="text-text-dim leading-relaxed">Adaptive responses ensure the robot maintains a consistent social persona that evolves organically.</p>
                  </div>
                </div>
              </div>
              <div className="mt-12 p-6 rounded-2xl bg-white/5 border border-white/10 italic text-text-dim text-sm">
                "A user who repeatedly breaks rules may receive polite refusals, while positive interactions increase cooperation."
              </div>
            </div>
            <div className="glass-panel p-10 bg-accent/5 overflow-hidden relative min-h-[500px] flex items-center justify-center">
              {/* Knowledge Graph Visualization */}
              <div className="absolute inset-0 subtle-grid" />
              <svg viewBox="0 0 400 400" className="w-full h-full max-w-[500px]">
                {/* Connection Lines */}
                <g className="connections" stroke="#ffffff" strokeWidth="1" opacity="0.2">
                  <motion.line
                    x1="200" y1="200" x2="120" y2="100"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, repeat: Infinity, repeatType: "reverse" }}
                  />
                  <motion.line
                    x1="200" y1="200" x2="280" y2="100"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: 0.2, repeat: Infinity, repeatType: "reverse" }}
                  />
                  <motion.line
                    x1="200" y1="200" x2="320" y2="200"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: 0.4, repeat: Infinity, repeatType: "reverse" }}
                  />
                  <motion.line
                    x1="200" y1="200" x2="280" y2="300"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: 0.6, repeat: Infinity, repeatType: "reverse" }}
                  />
                  <motion.line
                    x1="200" y1="200" x2="120" y2="300"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: 0.8, repeat: Infinity, repeatType: "reverse" }}
                  />
                  <motion.line
                    x1="200" y1="200" x2="80" y2="200"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: 1.0, repeat: Infinity, repeatType: "reverse" }}
                  />
                  {/* Inter-node connections */}
                  <line x1="120" y1="100" x2="280" y2="100" opacity="0.1" />
                  <line x1="280" y1="100" x2="320" y2="200" opacity="0.1" />
                  <line x1="120" y1="300" x2="80" y2="200" opacity="0.1" />
                </g>

                {/* Memory Nodes */}
                {[
                  { x: 200, y: 200, label: "Core", delay: 0 },
                  { x: 120, y: 100, label: "User", delay: 0.2 },
                  { x: 280, y: 100, label: "Task", delay: 0.4 },
                  { x: 320, y: 200, label: "Context", delay: 0.6 },
                  { x: 280, y: 300, label: "Emotion", delay: 0.8 },
                  { x: 120, y: 300, label: "Action", delay: 1.0 },
                  { x: 80, y: 200, label: "History", delay: 1.2 }
                ].map((node, i) => (
                  <g key={i}>
                    {/* Outer pulse ring */}
                    <motion.circle
                      cx={node.x}
                      cy={node.y}
                      r="20"
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth="1"
                      opacity="0.1"
                      animate={{
                        r: [20, 30, 20],
                        opacity: [0.1, 0.3, 0.1]
                      }}
                      transition={{
                        duration: 3,
                        delay: node.delay,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />

                    {/* Node circle */}
                    <motion.circle
                      cx={node.x}
                      cy={node.y}
                      r="12"
                      fill={i === 0 ? "#ffffff" : "#0a0a0a"}
                      stroke="#ffffff"
                      strokeWidth="2"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: node.delay }}
                    />

                    {/* Label */}
                    <text
                      x={node.x}
                      y={node.y + 35}
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="10"
                      fontFamily="monospace"
                      opacity="0.6"
                    >
                      {node.label}
                    </text>
                  </g>
                ))}

                {/* Data flow particles */}
                {[0, 1, 2].map((i) => (
                  <motion.circle
                    key={`particle-${i}`}
                    r="2"
                    fill="#ffffff"
                    initial={{ opacity: 0 }}
                    animate={{
                      opacity: [0, 1, 0],
                      x: [200, 120, 200],
                      y: [200, 100, 200]
                    }}
                    transition={{
                      duration: 2,
                      delay: i * 0.7,
                      repeat: Infinity,
                      ease: "linear"
                    }}
                  />
                ))}
              </svg>

              {/* Title overlay */}
              <div className="absolute bottom-8 left-0 right-0 text-center">
                <span className="font-mono text-2xl font-bold gradient-text block mb-1">Memory Graph</span>
                <span className="font-mono text-text-dim uppercase tracking-widest text-xs">Live Knowledge Network</span>
              </div>
            </div>
          </div>
        </section>

        {/* 4. High-Level Architecture */}
        <section id="architecture" className="py-32">
          <SectionHeading
            number="03"
            title="System Architecture"
            subtitle="A distributed multi-processing pipeline connecting local perception to remote cognition."
          />
          <AnimatedArchitecture />
        </section>

        {/* 5. Deep Dive Walkthrough */}
        <section id="walkthrough-section" className="py-32">
          <SectionHeading
            number="04"
            title="How It Works"
            subtitle="An end-to-end walkthrough of a single cognitive loop."
          />
          <Walkthrough />
        </section>

        {/* 6. Why This Matters */}
        <section id="relevance" className="py-32 bg-white/5 -mx-8 px-16 rounded-3xl border border-white/10">
          <div className="max-w-4xl">
            <h2 className="text-5xl font-bold mb-10 gradient-text leading-tight">Beyond Motion: The Future <br /> of Human-Robot Alignment.</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-12 font-light text-lg text-text-dim leading-relaxed">
              <p>Emotionally adaptive humanoids allow for robots that don't just work in human spaces, but fit into human social structures.</p>
              <p>Socially informed decision-making ensures that safety is prioritized not just physically, but psychologically in shared environments.</p>
              <p>By establishing a foundation for long-term memory, we enable genuine human-robot partnerships that grow over years.</p>
            </div>
          </div>
        </section>

        {/* 7. Use Cases */}
        <section id="use-cases" className="py-32">
          <SectionHeading
            number="06"
            title="Real-World Applications"
            subtitle="Practical scenarios where EDEN's memory-driven cognition creates meaningful impact."
          />
          <UseCases />
        </section>

        {/* 8. Roadmap */}
        <section id="roadmap" className="py-32">
          <SectionHeading
            number="05"
            title="Release Roadmap"
            subtitle="Our current development objectives and future milestones."
          />
          <div className="space-y-4">
            {[
              { title: "Jetson Orin Migration", state: "In Progress", date: "Q1 2026" },
              { title: "Emotional Display V2", state: "Planning", date: "Q2 2026" },
              { title: "Bipedal Prototype Alpha", state: "Research", date: "Q3 2026" },
              { title: "Multi-user Supermemory", state: "Backlog", date: "Q4 2026" }
            ].map((job, i) => (
              <div key={i} className="flex justify-between items-center p-8 border border-white/5 rounded-2xl hover:bg-white/5 transition-all">
                <div className="flex items-center gap-8">
                  <span className="font-mono text-text-dim text-xs">{job.date}</span>
                  <h4 className="text-xl font-semibold">{job.title}</h4>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${job.state === 'In Progress' ? 'bg-accent animate-pulse' : 'bg-white/20'}`} />
                    <span className="font-mono text-xs uppercase text-text-dim">{job.state}</span>
                  </div>
                  <ChevronRight size={16} className="text-white/20" />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 9. Open Source */}
        <section id="github" className="py-32 border-t border-white/5">
          <div className="glass-panel p-16 text-center bg-accent/5">
            <SectionHeading
              number="07"
              title="Open Source & Community"
              subtitle="EDEN is built for the research community. We believe in open standards and developer-first documentation."
            />
            <GitHubStats />
            <div className="flex justify-center gap-6 mt-12">
              <a
                href="https://github.com/EDEN-robotics"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-4 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-all flex items-center gap-3"
              >
                <Github size={18} /> Join Organization
              </a>
              <button className="px-8 py-4 bg-transparent border border-white/10 text-white font-semibold rounded-full hover:bg-white/5 transition-all">
                Browse Documentation
              </button>
            </div>
          </div>
        </section>

        {/* 10. Team Section */}
        <section id="team" className="py-32">
          <SectionHeading
            number="08"
            title="The Team"
            subtitle="Meet the people building the future of humanoid robotics."
          />
          <Team />
        </section>

        {/* Contact for Contributions */}
        <section className="py-32">
          <div className="glass-panel p-12 text-center">
            <h3 className="text-3xl font-bold mb-4">Interested in Contributing?</h3>
            <p className="text-text-dim text-lg mb-8 max-w-2xl mx-auto">
              We're always looking for passionate researchers and developers to join the EDEN project.
            </p>
            <a
              href="mailto:ved.soni@tamu.edu"
              className="inline-flex items-center gap-3 px-8 py-4 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-all"
            >
              <Mail size={18} />
              ved.soni@tamu.edu
            </a>
          </div>
        </section>
      </main>

      <footer className="py-20 border-t border-white/5 px-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center flex-wrap gap-8">
          <div className="flex items-center gap-3 opacity-50">
            <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center font-bold text-white italic">E</div>
            <span className="font-mono tracking-tighter text-sm uppercase">Eden Robotics</span>
          </div>
          <p className="text-text-dim text-xs font-mono uppercase tracking-widest italic">
            Developed in research partnership with humanoid laboratories worldwide.
          </p>
          <div className="flex gap-8 text-xs font-mono text-text-dim uppercase tracking-widest">
            <a href="#" className="hover:text-white transition-colors">Twitter</a>
            <a href="#" className="hover:text-white transition-colors">LinkedIn</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
