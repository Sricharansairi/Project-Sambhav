import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { Send, Loader2, Bot, User } from 'lucide-react';
import { runPragmaChat } from '../lib/api';

interface PragmaChatProps {
  predictionId?: string;
  contextParams: Record<string, any>;
  baselinePrediction: any;
}

export function PragmaChat({ predictionId, contextParams, baselinePrediction }: PragmaChatProps) {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([{
    role: 'assistant',
    content: 'Dr. Elias Vance here. I’ve reviewed the forensic parameters of this subject. What specific psychological or linguistic markers would you like me to elaborate on?'
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const history = messages.filter(m => m.role !== 'system');
      const res = await runPragmaChat({
        prediction_id: predictionId,
        question: userMsg,
        context: JSON.stringify(baselinePrediction),
        history,
        parameters: contextParams
      });
      
      setMessages(prev => [...prev, { role: 'assistant', content: res.reply }]);
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: 'assistant', content: "Error communicating with profiler system. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[400px] bg-background/50 border border-white/10 rounded-lg overflow-hidden relative">
      <div className="bg-white/5 border-b border-white/10 px-3 py-2 flex items-center gap-2">
        <Bot className="w-4 h-4 text-primary" />
        <span className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">Dr. Vance (Expert Profiler)</span>
      </div>
      
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 5 }} 
            animate={{ opacity: 1, y: 0 }} 
            key={i} 
            className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-primary/20 text-primary' : 'bg-white/10 text-muted-foreground'}`}>
              {msg.role === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
            </div>
            <div className={`text-xs leading-relaxed max-w-[85%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-primary/10 border border-primary/20 text-primary-foreground' : 'bg-white/5 border border-white/10 text-muted-foreground'}`}>
              {msg.content}
            </div>
          </motion.div>
        ))}
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-6 h-6 rounded-full bg-white/10 text-muted-foreground flex items-center justify-center shrink-0">
              <Bot className="w-3.5 h-3.5" />
            </div>
            <div className="text-xs leading-relaxed max-w-[85%] p-3 rounded-lg bg-white/5 border border-white/10 text-muted-foreground flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" /> Analyzing parameters...
            </div>
          </div>
        )}
      </div>

      <div className="p-3 border-t border-white/10 bg-background/80">
        <div className="relative">
          <input
            type="text"
            className="w-full h-10 pl-3 pr-10 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-primary/50 text-white placeholder:text-muted-foreground/50 transition-colors"
            placeholder="Ask Dr. Vance about specific cues..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isLoading}
          />
          <button 
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-muted-foreground hover:text-primary transition-colors disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
