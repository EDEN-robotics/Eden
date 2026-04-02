import React, { useState, useEffect, useRef, Component } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, Send, LogIn, LogOut, Loader2 } from 'lucide-react';
import { useUser, useClerk, SignInButton } from '@clerk/clerk-react';

const STORAGE_KEY = 'eden_chat_history';
const SYSTEM_PROMPT = `You are EDEN (Emotionally-Driven Embodied Navigation), a humanoid robotics cognitive architecture assistant. You help users understand EDEN's layered architecture including the Perception Layer, Context Layer, Cognitive Layer, Supermemory, Planning Layer, and Action Layer. You are friendly, concise, and technically knowledgeable about robotics, AI, ROS 2, and human-robot interaction. Keep responses focused and under 200 words unless the user asks for detail.`;

function loadHistory(userId) {
  try {
    const data = localStorage.getItem(`${STORAGE_KEY}_${userId}`);
    return data ? JSON.parse(data) : [];
  } catch { return []; }
}

function saveHistory(userId, messages) {
  try {
    localStorage.setItem(`${STORAGE_KEY}_${userId}`, JSON.stringify(messages));
  } catch { /* quota exceeded */ }
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

// Letter initial avatar
function InitialAvatar({ letter, variant = 'eden' }) {
  const styles = variant === 'eden'
    ? 'bg-white/10 text-white/80'
    : 'bg-white text-black';
  return (
    <div className={`w-8 h-8 rounded-lg ${styles} flex items-center justify-center shrink-0 text-xs font-bold select-none`}>
      {letter}
    </div>
  );
}

// ── Chat UI core (shared between auth and guest modes) ──

function ChatUI({ userId, userName, userInitial, headerRight, isOpen, setIsOpen }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { setMessages(loadHistory(userId)); }, [userId]);
  useEffect(() => {
    if (messages.length > 0) saveHistory(userId, messages);
  }, [messages, userId]);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 300);
  }, [isOpen]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMsg = { role: 'user', content: text, timestamp: Date.now() };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setIsStreaming(true);

    const apiMessages = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...updated.map(m => ({ role: m.role, content: m.content }))
    ];

    try {
      const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_OPENROUTER_API_KEY}`,
          'HTTP-Referer': window.location.origin,
          'X-Title': 'EDEN Robotics'
        },
        body: JSON.stringify({
          model: 'nvidia/nemotron-3-super-120b-a12b:free',
          messages: apiMessages,
          stream: true
        })
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let content = '';

      setMessages(prev => [...prev, { role: 'assistant', content: '', timestamp: Date.now() }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '));
        for (const line of lines) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const delta = JSON.parse(data).choices?.[0]?.delta?.content;
            if (delta) {
              content += delta;
              const snapshot = content;
              setMessages(prev => {
                const copy = [...prev];
                copy[copy.length - 1] = { ...copy[copy.length - 1], content: snapshot };
                return copy;
              });
            }
          } catch { /* malformed chunk */ }
        }
      }
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant', content: 'Sorry, I encountered an error. Please try again.',
        timestamp: Date.now(), error: true
      }]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem(`${STORAGE_KEY}_${userId}`);
  };

  return (
    <>
      <ChatBubble isOpen={isOpen} setIsOpen={setIsOpen} />
      <AnimatePresence>
        {isOpen && (
          <motion.div
            id="eden-chat-panel"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="fixed bottom-24 right-6 z-[9998] w-[400px] max-h-[560px] flex flex-col rounded-2xl overflow-hidden shadow-2xl"
            style={{
              background: 'rgba(10,10,10,0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255,255,255,0.08)'
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
              <div className="flex items-center gap-2.5">
                <span className="text-[15px] font-bold text-white tracking-tight">EDEN</span>
                <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.4)]" />
              </div>
              <div className="flex items-center gap-3">
                <button onClick={clearHistory} className="text-[11px] text-white/25 hover:text-white/50 transition-colors font-mono cursor-pointer">Clear</button>
                {headerRight}
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-5 py-4 eden-chat-scroll" style={{ maxHeight: '400px', minHeight: '200px' }}>
              {messages.length === 0 && (
                <div className="py-6">
                  <div className="flex items-start gap-3">
                    <InitialAvatar letter="E" variant="eden" />
                    <div>
                      <div className="flex items-baseline gap-2 mb-1">
                        <span className="text-[13px] font-bold text-white">EDEN</span>
                        <span className="text-[11px] text-white/20 font-mono">now</span>
                      </div>
                      <p className="text-[13px] text-white/50 leading-relaxed">
                        Hi {userName}! Ask me anything about EDEN's architecture.
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {messages.map((msg, i) => {
                const isUser = msg.role === 'user';
                const showHeader = i === 0 || messages[i - 1].role !== msg.role;
                return (
                  <div
                    key={i}
                    className={`group ${showHeader ? 'mt-4 first:mt-0' : 'mt-0.5'} ${!isUser && msg.error ? 'text-red-400' : ''}`}
                  >
                    {showHeader ? (
                      <div className="flex items-start gap-3">
                        <InitialAvatar
                          letter={isUser ? userInitial : 'E'}
                          variant={isUser ? 'user' : 'eden'}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-baseline gap-2 mb-0.5">
                            <span className="text-[13px] font-bold text-white">{isUser ? userName : 'EDEN'}</span>
                            <span className="text-[11px] text-white/20 font-mono">{formatTime(msg.timestamp)}</span>
                          </div>
                          <div className="text-[13px] text-white/70 leading-relaxed whitespace-pre-wrap">
                            {msg.content || (
                              <span className="inline-flex items-center gap-1.5 text-white/25">
                                <Loader2 size={12} className="animate-spin" /> Typing...
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="pl-11">
                        <div className="text-[13px] text-white/70 leading-relaxed whitespace-pre-wrap">
                          {msg.content}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="px-4 py-3 border-t border-white/[0.06]">
              <div className="flex items-center gap-2 bg-white/[0.04] rounded-lg px-3 py-2.5 border border-white/[0.06] focus-within:border-white/15 transition-colors">
                <input
                  ref={inputRef} id="eden-chat-input" type="text" value={input}
                  onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
                  placeholder="Message EDEN..." disabled={isStreaming}
                  className="flex-1 bg-transparent text-white text-[13px] outline-none placeholder:text-white/20 disabled:opacity-50"
                />
                <button onClick={sendMessage} disabled={!input.trim() || isStreaming}
                  className="p-1 rounded hover:bg-white/10 transition-colors text-white/30 hover:text-white disabled:opacity-20 disabled:hover:bg-transparent cursor-pointer">
                  {isStreaming ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// ── Chat bubble button ──

function ChatBubble({ isOpen, setIsOpen }) {
  return (
    <motion.button
      id="eden-chat-toggle"
      onClick={() => setIsOpen(!isOpen)}
      className="fixed bottom-6 right-6 z-[9999] w-14 h-14 rounded-full flex items-center justify-center shadow-2xl transition-colors cursor-pointer"
      style={{
        background: isOpen ? 'rgba(255,255,255,0.1)' : 'white',
        color: isOpen ? 'white' : 'black',
        border: '1px solid rgba(255,255,255,0.15)'
      }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
    >
      <AnimatePresence mode="wait">
        {isOpen ? (
          <motion.div key="close" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.15 }}>
            <X size={22} />
          </motion.div>
        ) : (
          <motion.div key="open" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} transition={{ duration: 0.15 }}>
            <MessageCircle size={22} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.button>
  );
}

// ── Authenticated version ──

function AuthenticatedChat() {
  const [isOpen, setIsOpen] = useState(false);
  const { isSignedIn, user, isLoaded } = useUser();
  const { signOut } = useClerk();

  if (!isLoaded) {
    return (
      <>
        <ChatBubble isOpen={isOpen} setIsOpen={setIsOpen} />
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              className="fixed bottom-24 right-6 z-[9998] w-[400px] flex items-center justify-center py-20 rounded-2xl shadow-2xl"
              style={{ background: 'rgba(10,10,10,0.95)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.08)' }}
            >
              <Loader2 size={20} className="text-white/20 animate-spin" />
            </motion.div>
          )}
        </AnimatePresence>
      </>
    );
  }

  if (!isSignedIn) {
    return (
      <>
        <ChatBubble isOpen={isOpen} setIsOpen={setIsOpen} />
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              className="fixed bottom-24 right-6 z-[9998] w-[400px] flex flex-col items-center px-10 py-14 text-center rounded-2xl shadow-2xl"
              style={{ background: 'rgba(10,10,10,0.95)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.08)' }}
            >
              <div className="w-14 h-14 rounded-xl bg-white/[0.04] flex items-center justify-center mb-6 border border-white/[0.06]">
                <span className="text-xl font-bold text-white/50">E</span>
              </div>
              <h4 className="text-base font-semibold text-white mb-1.5">Chat with EDEN</h4>
              <p className="text-[13px] text-white/35 mb-8 leading-relaxed">
                Sign in to ask questions about EDEN's architecture and capabilities.
              </p>
              <SignInButton mode="modal">
                <button className="flex items-center gap-2 px-6 py-2.5 bg-white text-black font-semibold rounded-lg text-sm hover:bg-gray-200 transition-colors cursor-pointer">
                  <LogIn size={15} />
                  Sign In
                </button>
              </SignInButton>
            </motion.div>
          )}
        </AnimatePresence>
      </>
    );
  }

  const firstName = user.firstName || 'You';
  const initial = firstName.charAt(0).toUpperCase();

  return (
    <ChatUI
      userId={user.id}
      userName={firstName}
      userInitial={initial}
      isOpen={isOpen}
      setIsOpen={setIsOpen}
      headerRight={
        <button
          onClick={() => signOut()}
          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors text-white/25 hover:text-white/50 cursor-pointer"
          title="Sign out"
        >
          <LogOut size={13} />
        </button>
      }
    />
  );
}

// ── Guest version ──

function GuestChat() {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <ChatUI
      userId="guest"
      userName="there"
      userInitial="?"
      isOpen={isOpen}
      setIsOpen={setIsOpen}
      headerRight={null}
    />
  );
}

// ── Error boundary for missing ClerkProvider ──

class ClerkErrorBoundary extends Component {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) return <GuestChat />;
    return this.props.children;
  }
}

// ── Main export ──

export default function ChatWidget() {
  return (
    <ClerkErrorBoundary>
      <AuthenticatedChat />
    </ClerkErrorBoundary>
  );
}
