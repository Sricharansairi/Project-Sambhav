import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { MessageCircle, Send, Loader2, Bot, User, X, ShieldAlert } from 'lucide-react';
import { runResultChat } from '../lib/api';

interface Message { role: 'user' | 'assistant'; content: string; }

interface ResultChatbotProps {
  isOpen: boolean;
  onClose: () => void;
  context: any;          // full result object — will be JSON.stringified
  mode: string;
  domain: string;
  title?: string;
}

export function ResultChatbot({ isOpen, onClose, context, mode, domain, title }: ResultChatbotProps) {
  const [messages, setMessages]     = useState<Message[]>([]);
  const [input, setInput]           = useState('');
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const bottomRef                   = useRef<HTMLDivElement>(null);
  const inputRef                    = useRef<HTMLTextAreaElement>(null);

  // Seed a welcome message when opened
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{
        role: 'assistant',
        content: `Hi! I'm Sambhav's analysis assistant for this **${domain}** prediction result. Ask me anything about the outcomes, probabilities, reasoning, or what you can do to improve them. I'll only discuss this specific result.`,
      }]);
    }
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100);
  }, [isOpen]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    setError(null);

    const userMsg: Message = { role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const contextStr = typeof context === 'string' ? context : JSON.stringify(context, null, 2);
      const history = messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content }));

      const res = await runResultChat({
        question: q,
        context: contextStr.slice(0, 3000),
        mode,
        domain,
        history,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: res.reply || 'No response received.' }]);
    } catch (e: any) {
      setError('Failed to get a response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ type: 'spring', damping: 20 }}
        onClick={e => e.stopPropagation()}
        className="relative w-full max-w-lg h-[540px] flex flex-col rounded-2xl overflow-hidden shadow-2xl bg-black/95 border border-white/10 support-backdrop-blur"
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/5 bg-white/5">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-white/10 border border-white/20">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-foreground truncate">{title || 'Sambhav Analysis Assistant'}</p>
            <p className="text-[10px] text-muted-foreground capitalize">{mode} mode · {domain}</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>

        {/* Safety notice */}
        <div className="flex items-center gap-2 px-4 py-1.5 bg-amber-500/5 border-b border-amber-500/10">
          <ShieldAlert className="w-3 h-3 text-amber-400 shrink-0" />
          <p className="text-[9px] text-amber-400/80">Only answers questions about this specific prediction result</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scrollbar-thin">
          {messages.map((msg, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center
                ${msg.role === 'user' ? 'bg-primary/20 border border-primary/30' : 'bg-white/10 border border-white/15'}`}>
                {msg.role === 'user'
                  ? <User className="w-3 h-3 text-primary" />
                  : <Bot className="w-3 h-3 text-muted-foreground" />}
              </div>
              <div className={`max-w-[82%] rounded-xl px-3 py-2 text-[11px] leading-relaxed
                ${msg.role === 'user'
                  ? 'bg-primary/20 border border-primary/20 text-foreground'
                  : 'bg-white/5 border border-white/10 text-muted-foreground'}`}>
                {msg.content.split('**').map((part, j) =>
                  j % 2 === 0 ? part : <strong key={j} className="text-foreground font-semibold">{part}</strong>
                )}
              </div>
            </motion.div>
          ))}
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-2">
              <div className="w-6 h-6 rounded-full bg-white/10 border border-white/15 flex items-center justify-center">
                <Bot className="w-3 h-3 text-muted-foreground" />
              </div>
              <div className="px-3 py-2 bg-white/5 border border-white/10 rounded-xl">
                <div className="flex gap-1 items-center h-4">
                  {[0,0.2,0.4].map((d, i) => (
                    <motion.div key={i} animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2, delay: d }}
                      className="w-1.5 h-1.5 rounded-full bg-primary/60" />
                  ))}
                </div>
              </div>
            </motion.div>
          )}
          {error && (
            <p className="text-[10px] text-destructive/80 text-center italic">{error}</p>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-4 py-3 border-t border-white/10">
          <div className="flex gap-2 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              disabled={loading}
              placeholder="Ask about this prediction result…"
              rows={2}
              className="flex-1 px-3 py-2 text-[11px] bg-white/5 border border-white/10 rounded-xl focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/30 transition-all resize-none placeholder:text-muted-foreground/50 disabled:opacity-50"
            />
            <motion.button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              className="flex-shrink-0 w-9 h-9 rounded-xl bg-primary flex items-center justify-center disabled:opacity-40 transition-all"
            >
              {loading ? <Loader2 className="w-4 h-4 text-black animate-spin" /> : <Send className="w-4 h-4 text-black" />}
            </motion.button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
