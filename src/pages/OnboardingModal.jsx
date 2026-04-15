import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, ArrowRight, Sparkles, Eye } from 'lucide-react'
import { addProfileFact } from '../lib/supermemory'

const PROMPTS = [
  {
    key: 'role',
    label: 'What do you work on?',
    placeholder: 'I lead hardware integration · I handle the cognitive layer · …',
    hint: 'A sentence about your role or focus.',
  },
  {
    key: 'signal',
    label: 'Tell EDEN one thing about you',
    placeholder: 'I prefer Rust · I live in Pittsburgh · I built the gazebo sim · …',
    hint: 'Anything EDEN should remember long-term.',
  },
]

export default function OnboardingModal({ open, user, onClose }) {
  const [step, setStep] = useState(0)
  const [values, setValues] = useState({})
  const [submitting, setSubmitting] = useState(false)

  if (!user) return null
  const prompt = PROMPTS[step]
  const isLast = step === PROMPTS.length - 1

  async function handleNext() {
    const value = (values[prompt.key] || '').trim()
    if (!value) { handleSkip(); return }

    setSubmitting(true)
    await addProfileFact({
      userId: user.id,
      userName: user.fullName || user.firstName || 'Member',
      fact: value,
    })
    setSubmitting(false)

    if (isLast) { finish(); return }
    setStep(step + 1)
  }

  function handleSkip() {
    if (isLast) { finish(); return }
    setStep(step + 1)
  }

  function finish() {
    try {
      window.localStorage.setItem(`eden:onboarded:${user.id}`, '1')
    } catch {}
    onClose?.()
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ type: 'spring', damping: 24, stiffness: 280 }}
            className="relative w-full max-w-lg bg-gradient-to-br from-bg-secondary to-black border border-cyan-400/20 rounded-2xl shadow-2xl overflow-hidden"
          >
            {/* Header with glow */}
            <div className="absolute -top-32 -right-20 w-80 h-80 bg-cyan-500/20 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute -bottom-32 -left-20 w-80 h-80 bg-violet-500/20 rounded-full blur-3xl pointer-events-none" />

            <div className="relative px-8 pt-8 pb-6">
              <div className="flex items-center gap-2 mb-4">
                <Eye size={14} className="text-amber-300" />
                <span className="text-[10px] font-mono uppercase tracking-widest text-amber-300/80">Perception Layer · first capture</span>
              </div>
              <h2 className="text-2xl font-bold mb-2 text-white">Welcome, {user.firstName || 'friend'}.</h2>
              <p className="text-sm text-white/50 leading-relaxed">
                EDEN is a cognitive architecture with persistent memory. Give it two grounding facts about you — these become permanent identity records in the Perception Layer and EDEN will use them across every conversation.
              </p>
            </div>

            {/* Step progress */}
            <div className="flex gap-1 px-8 mb-6">
              {PROMPTS.map((_, i) => (
                <div key={i} className={`h-0.5 flex-1 rounded-full transition-colors ${i <= step ? 'bg-cyan-400' : 'bg-white/10'}`} />
              ))}
            </div>

            <div className="px-8 pb-8">
              <label className="block text-sm font-semibold text-white mb-1">{prompt.label}</label>
              <p className="text-xs text-white/40 mb-3">{prompt.hint}</p>
              <input
                autoFocus
                type="text"
                value={values[prompt.key] || ''}
                onChange={(e) => setValues({ ...values, [prompt.key]: e.target.value })}
                onKeyDown={(e) => { if (e.key === 'Enter' && !submitting) handleNext() }}
                placeholder={prompt.placeholder}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder:text-white/25 outline-none focus:border-cyan-400/40 transition-colors"
                disabled={submitting}
              />

              <div className="mt-3 flex items-center gap-2 text-[10px] text-white/30 font-mono">
                <Brain size={10} className="text-cyan-400/70" />
                Stored as <code className="text-cyan-300">isStatic: true</code> — EDEN never forgets this.
              </div>

              <div className="flex items-center justify-between mt-6">
                <button
                  onClick={handleSkip}
                  disabled={submitting}
                  className="text-xs text-white/40 hover:text-white/70 transition-colors"
                >
                  {isLast ? 'Skip & enter chat' : 'Skip this one'}
                </button>
                <button
                  onClick={handleNext}
                  disabled={submitting}
                  className="flex items-center gap-2 bg-white text-black text-sm font-semibold px-4 py-2 rounded-lg hover:bg-white/90 transition-colors disabled:opacity-50"
                >
                  {submitting ? 'Saving…' : isLast ? (<>Enter chat <Sparkles size={13} /></>) : (<>Next <ArrowRight size={13} /></>)}
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
