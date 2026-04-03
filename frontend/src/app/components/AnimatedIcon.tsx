import { motion } from 'motion/react';
import { LucideIcon } from 'lucide-react';
import { useState } from 'react';

interface AnimatedIconProps {
  icon: LucideIcon;
  size?: number;
  color?: string;
  hoverEffect?: 'pulse' | 'twitch' | 'rotate' | 'scale' | 'glow';
  className?: string;
}

export function AnimatedIcon({ 
  icon: Icon, 
  size = 24, 
  color = 'currentColor',
  hoverEffect = 'twitch',
  className = ''
}: AnimatedIconProps) {
  const [isHovered, setIsHovered] = useState(false);

  const hoverAnimations = {
    pulse: {
      scale: isHovered ? [1, 1.2, 1] : 1,
      transition: { duration: 0.6, repeat: isHovered ? Infinity : 0 }
    },
    twitch: {
      rotate: isHovered ? [0, -5, 5, -3, 3, 0] : 0,
      x: isHovered ? [0, -2, 2, -1, 1, 0] : 0,
      transition: { duration: 0.5, repeat: isHovered ? Infinity : 0, repeatDelay: 0.3 }
    },
    rotate: {
      rotate: isHovered ? 360 : 0,
      transition: { duration: 0.8, ease: 'easeInOut' }
    },
    scale: {
      scale: isHovered ? 1.3 : 1,
      transition: { duration: 0.3, ease: 'easeOut' }
    },
    glow: {
      filter: isHovered ? 'drop-shadow(0 0 8px currentColor)' : 'drop-shadow(0 0 0px currentColor)',
      scale: isHovered ? 1.1 : 1,
      transition: { duration: 0.3 }
    }
  };

  return (
    <motion.div
      className={`inline-flex items-center justify-center ${className}`}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      animate={hoverAnimations[hoverEffect]}
      style={{ color }}
    >
      <Icon size={size} />
    </motion.div>
  );
}
