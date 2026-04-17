import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import {
  Brain,
  ArrowRight,
  Github,
  ChevronRight,
  Linkedin,
  Mail,
  Star,
  GitFork,
  Users,
} from 'lucide-react';
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/800.css";
import "@fontsource/jetbrains-mono";

// --- Components ---

const Nav = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 flex justify-end items-center px-8 py-6 backdrop-blur-sm bg-black/10 transition-all border-b border-white/5">
    <div className="flex gap-8 text-xs font-mono uppercase tracking-widest text-text-dim">
      <a href="#demos" className="hover:text-white transition-colors">Demos</a>
      <a href="#slides" className="hover:text-white transition-colors">Deck</a>
      <a href="#roadmap" className="hover:text-white transition-colors">Roadmap</a>
      <a href="#team" className="hover:text-white transition-colors">Team</a>
      <Link to="/architecture" className="hover:text-white transition-colors">Spec</Link>
      <Link to="/chat" className="hover:text-white transition-colors">Chat</Link>
      <Link to="/sim" className="hover:text-white transition-colors">Sim</Link>
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

const SlideGrid = () => {
  const BASE = import.meta.env.BASE_URL
  const slides = [
    { title: 'What we built',        tag: 'Fall 2025',        img: `${BASE}team/team.jpg`,                     fallback: null,                        section: 'intro' },
    { title: 'Making it perfect',    tag: 'North Star',       img: `${BASE}robot_body.png`,                    fallback: null,                        section: 'vision' },
    { title: 'The 5-Layer Pipeline', tag: 'Architecture',     img: `${BASE}supermemory_graph.png`,             fallback: null,                        section: 'architecture' },
    { title: 'Cognitive memory',     tag: 'Supermemory',      img: `${BASE}supermemory_extraction.png`,        fallback: null,                        section: 'cognitive' },
    { title: 'Memory graph',         tag: 'Retrieval + decay',img: `${BASE}supermemory_graph.png`,             fallback: null,                        section: 'memory' },
    { title: 'Gazebo simulation',    tag: 'Sim · Physics',    img: `${BASE}gazebo_1.png`,                      fallback: null,                        section: 'sim' },
    { title: 'Sim environment',      tag: 'Gazebo world',     img: `${BASE}gazebo_2.png`,                      fallback: null,                        section: 'sim' },
    { title: 'Navigation trial',     tag: 'Sim · Planning',   img: `${BASE}gazebo_3.png`,                      fallback: null,                        section: 'sim' },
    { title: '4-Wheeled chassis',    tag: 'Hardware',         img: `${BASE}robot_ackermann.png`,               fallback: null,                        section: 'hardware' },
    { title: 'Robot internals',      tag: 'Hardware',         img: `${BASE}robot_internal.png`,                fallback: null,                        section: 'hardware' },
    { title: 'ESP-CAM vision',       tag: 'Perception',       img: `${BASE}robot_espcam.png`,                  fallback: null,                        section: 'perception' },
    { title: 'Discord bot logs',     tag: 'Cognitive loop',   img: `${BASE}discord_bot_logs.png`,              fallback: null,                        section: 'agent' },
    { title: 'Discord interactions', tag: 'Agent demo',       img: `${BASE}discord_bot_chat.png`,              fallback: null,                        section: 'agent' },
  ]
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {slides.map((s, i) => (
        <motion.a
          key={i}
          href={`${BASE}dr3.html`}
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.45, delay: i * 0.03 }}
          whileHover={{ y: -4 }}
          className="group relative aspect-[4/3] rounded-xl overflow-hidden border border-white/10 bg-black hover:border-cyan-400/40 transition-colors"
        >
          <img
            src={s.img}
            alt={s.title}
            loading="lazy"
            onError={(e) => { e.currentTarget.style.opacity = '0.15' }}
            className="absolute inset-0 w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent" />
          <div className="absolute top-2 left-2">
            <span className="text-[9px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-md bg-black/60 backdrop-blur border border-white/20 text-white/80">
              {s.section}
            </span>
          </div>
          <div className="absolute bottom-0 left-0 right-0 p-3">
            <div className="text-[10px] font-mono uppercase tracking-widest text-cyan-300/90 mb-1">{s.tag}</div>
            <div className="text-sm font-semibold text-white group-hover:text-cyan-200 transition-colors leading-tight">
              {s.title}
            </div>
          </div>
          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="w-6 h-6 rounded-full bg-white/90 text-black flex items-center justify-center">
              <ArrowRight size={12} />
            </div>
          </div>
        </motion.a>
      ))}
    </div>
  )
}

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

const TeamMemberCard = ({ member, index }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay: index * 0.1 }}
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
);

const Team = () => {
  const currentMembers = [
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
      name: "Andrew Zheng",
      role: "Software Engineer",
      image: "/team/andrew-zheng.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/andrew--zheng/"
      }
    },
    {
      name: "Dhruv Bhambhani",
      role: "Software Engineer",
      image: "/team/dhruv-bhambhani.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/dhruvbhambhani05/"
      }
    },
    {
      name: "Sphoorthi Gurram",
      role: "Computer Engineer",
      image: "/team/sphoorthi-gurram.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/sphoorthi-gurram/"
      }
    }
  ];

  const previousMembers = [
    {
      name: "Juan Gomez Sandoval",
      role: "Electrical Engineer",
      image: "/team/juan-gomez-sandoval.jpg",
      socials: {
        linkedin: "https://www.linkedin.com/in/juan-gomez-sandoval-24aa55239/"
      }
    }
  ];

  return (
    <div className="space-y-16">
      {/* Current Members */}
      <div>
        <h3 className="text-2xl font-bold mb-8 text-white/80">Current Members</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {currentMembers.map((member, i) => (
            <TeamMemberCard key={i} member={member} index={i} />
          ))}
        </div>
      </div>

      {/* Previous Members */}
      <div>
        <h3 className="text-2xl font-bold mb-8 text-white/80">Previous Members</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {previousMembers.map((member, i) => (
            <TeamMemberCard key={i} member={member} index={i} />
          ))}
        </div>
      </div>
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
              <a
                href={`${import.meta.env.BASE_URL}poster.pdf`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-4 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-all flex items-center gap-3"
              >
                Read Poster <ArrowRight size={18} />
              </a>
              <Link
                to="/docs"
                className="px-8 py-4 bg-transparent border border-white/10 text-white font-semibold rounded-full hover:bg-white/5 transition-all"
              >
                View Documentation
              </Link>
            </div>


          </motion.div>
        </motion.div>
      </section>

      <main className="max-w-7xl mx-auto px-8 py-32">
        {/* Live demos row — chat, sim, deck */}
        <section id="demos" className="py-24 border-t border-white/5">
          <SectionHeading
            number="01"
            title="Live. Embodied. Remembers."
            subtitle="Three live surfaces. Talk to EDEN, watch it think and move, or read the whole architecture in the deck."
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { to: '/chat',          label: 'Open Chat', tag: 'Cognitive + memory', desc: 'Mention @eden. Watch Perception→Context→Memory→Cognitive→Planning→Action fire live.', accent: 'cyan' },
              { to: '/sim',           label: 'Open Sim',  tag: 'ROS 2 · Gazebo-style', desc: 'A 3D lab world. Drive EDEN from chat. Real obstacles, LIDAR, odometry, /cmd_vel bus.', accent: 'violet' },
              { href: `${import.meta.env.BASE_URL}dr3.html`, label: 'View Deck', tag: 'Design Review 3', desc: 'Every layer, every decision, every diagram. The full technical detail lives here.', accent: 'rose' },
            ].map((c) => {
              const Inner = (
                <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 p-8 h-full flex flex-col justify-between hover:border-white/30 transition-all group">
                  <div>
                    <div className={`text-[10px] font-mono uppercase tracking-widest mb-4 ${c.accent === 'cyan' ? 'text-cyan-300' : c.accent === 'violet' ? 'text-violet-300' : 'text-rose-300'}`}>{c.tag}</div>
                    <h3 className="text-2xl font-bold mb-3">{c.label}</h3>
                    <p className="text-text-dim text-sm leading-relaxed">{c.desc}</p>
                  </div>
                  <div className="inline-flex items-center gap-2 text-sm font-semibold mt-6 group-hover:translate-x-1 transition-transform">
                    Open <ArrowRight size={14} />
                  </div>
                </div>
              )
              return c.to ? (
                <Link key={c.label} to={c.to}>{Inner}</Link>
              ) : (
                <a key={c.label} href={c.href} target="_blank" rel="noopener noreferrer">{Inner}</a>
              )
            })}
          </div>
        </section>

        {/* Slide deck preview — thumbnail grid (the detail entry point) */}
        <section id="slides" className="py-24">
          <SectionHeading
            number="02"
            title="Design Review 3"
            subtitle="Architecture, cognitive layer, hardware, demos. Click any card to jump into the deck."
          />
          <SlideGrid />
          <div className="mt-8 flex justify-center">
            <a
              href={`${import.meta.env.BASE_URL}dr3.html`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm font-semibold px-5 py-2.5 rounded-lg bg-white text-black hover:bg-white/90 transition-colors"
            >
              Open full deck <ArrowRight size={14} />
            </a>
          </div>
        </section>

        {/* 3. Roadmap */}
        <section id="roadmap" className="py-32">
          <SectionHeading
            number="03"
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
              number="04"
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
              <Link
                to="/docs"
                className="px-8 py-4 bg-transparent border border-white/10 text-white font-semibold rounded-full hover:bg-white/5 transition-all"
              >
                Browse Documentation
              </Link>
            </div>
          </div>
        </section>

        {/* 10. Team Section */}
        <section id="team" className="py-32">
          <SectionHeading
            number="05"
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
