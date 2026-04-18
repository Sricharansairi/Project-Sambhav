import { motion } from 'motion/react';
import { useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { sounds } from '../lib/audio';

interface OutcomeRowProps {
  name: string;
  probability: number;
  delay: number;
  onWhyClick: () => void;
  isAnimating: boolean;
}

export function OutcomeRow({ name, probability, delay, onWhyClick, isAnimating }: OutcomeRowProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay + 0.05, duration: 0.15 }}
      className="group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-foreground/90">{name}</span>
          <motion.button
            onClick={() => { sounds.click(); onWhyClick(); }}
            className="px-1.5 py-0.5 text-[10px] rounded bg-primary/10 border border-primary/30 
                     text-primary hover:bg-primary/20 transition-colors flex items-center gap-1"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <HelpCircle className="w-2.5 h-2.5" />
            WHY
          </motion.button>
        </div>
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.1 }}
          className="text-xs font-medium text-primary"
        >
          {probability.toFixed(1)}%
        </motion.span>
      </div>
      
      {/* Progress Bar */}
      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={isAnimating ? { width: `${probability}%` } : { width: 0 }}
          transition={{ 
            delay: delay + 0.05,
            duration: 0.6,
            ease: [0.25, 0.1, 0.25, 1]
          }}
          className="h-full bg-gradient-to-r from-primary/80 to-secondary/60"
          style={{
            boxShadow: isHovered ? '0 0 8px rgba(192, 192, 192, 0.4)' : 'none'
          }}
        />
      </div>
    </motion.div>
  );
}
