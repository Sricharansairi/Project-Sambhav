import { motion } from 'motion/react';
import { useEffect, useState } from 'react';

interface ProbabilityBarProps {
  value: number; // 0-100
  label: string;
  color?: string;
  delay?: number;
}

export function ProbabilityBar({ 
  value, 
  label, 
  color = '#00fff2',
  delay = 0 
}: ProbabilityBarProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDisplayValue(value);
    }, delay * 1000);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm text-foreground/80">{label}</span>
        <motion.span 
          className="text-sm font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.5 }}
          style={{ color }}
        >
          {displayValue.toFixed(1)}%
        </motion.span>
      </div>
      
      <div className="relative h-3 bg-white/5 rounded-full overflow-hidden backdrop-blur-sm border border-white/10">
        {/* Glow background */}
        <motion.div
          className="absolute inset-0 rounded-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: displayValue > 0 ? 0.3 : 0 }}
          style={{
            background: `radial-gradient(ellipse at center, ${color}40 0%, transparent 70%)`,
            filter: 'blur(8px)',
          }}
        />
        
        {/* Glass fill */}
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${displayValue}%` }}
          transition={{ 
            duration: 1.5, 
            delay,
            ease: [0.16, 1, 0.3, 1] // Custom ease-out
          }}
          style={{
            background: `linear-gradient(90deg, ${color}80 0%, ${color}40 100%)`,
            boxShadow: `0 0 20px ${color}60, inset 0 1px 2px rgba(255, 255, 255, 0.2)`,
          }}
        >
          {/* Refractive shimmer */}
          <motion.div
            className="absolute inset-0 rounded-full"
            animate={{
              backgroundPosition: ['0% 0%', '200% 0%'],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'linear',
            }}
            style={{
              background: 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.3) 50%, transparent 100%)',
              backgroundSize: '50% 100%',
            }}
          />
        </motion.div>
        
        {/* Glass highlight */}
        <div className="absolute inset-x-0 top-0 h-[40%] rounded-full bg-gradient-to-b from-white/20 to-transparent pointer-events-none" />
      </div>
    </div>
  );
}
