import { ReactNode } from 'react';
import { motion, HTMLMotionProps } from 'motion/react';

interface GlassCardProps extends Omit<HTMLMotionProps<'div'>, 'children'> {
  children: ReactNode;
  variant?: 'default' | 'elevated' | 'interactive';
  glow?: boolean;
  glowColor?: string;
}

export function GlassCard({ 
  children, 
  variant = 'default', 
  glow = false,
  glowColor = 'rgba(0, 255, 242, 0.2)',
  className = '',
  ...props 
}: GlassCardProps) {
  const baseStyles = `
    relative backdrop-blur-xl rounded-xl border
    transition-all duration-300
  `;
  
  const variants = {
    default: 'bg-white/[0.03] border-white/10',
    elevated: 'bg-white/[0.05] border-white/20 shadow-lg shadow-black/50',
    interactive: 'bg-white/[0.03] border-white/10 hover:bg-white/[0.06] hover:border-white/20 cursor-pointer'
  };
  
  const glowStyles = glow ? {
    boxShadow: `0 0 40px ${glowColor}, inset 0 0 20px rgba(255, 255, 255, 0.03)`,
  } : {
    boxShadow: 'inset 0 0 20px rgba(255, 255, 255, 0.02)',
  };

  return (
    <motion.div
      className={`${baseStyles} ${variants[variant]} ${className}`}
      style={glowStyles}
      {...props}
    >
      {/* Liquid glass refraction effect */}
      <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
        <div 
          className="absolute inset-0 opacity-30"
          style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%, rgba(0, 255, 242, 0.05) 100%)',
          }}
        />
      </div>
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  );
}
