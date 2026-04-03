import { motion } from 'motion/react';
import logoImage from '../../assets/066d6bda782cfe271b2a192b0848783b83987f2e.png';

export function BackgroundLogo() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {/* Logo Watermark */}
      <motion.div
        animate={{
          opacity: [0.015, 0.025, 0.015],
          scale: [1, 1.02, 1],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
        style={{
          width: '60vw',
          maxWidth: '800px',
          aspectRatio: '1',
        }}
      >
        <img
          src={logoImage}
          alt=""
          className="w-full h-full object-contain opacity-20"
          style={{ filter: 'blur(1px)' }}
        />
      </motion.div>
      
      {/* Subtle ambient glow */}
      <motion.div
        animate={{
          opacity: [0.01, 0.02, 0.01],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full"
        style={{
          background: 'radial-gradient(circle, rgba(192, 192, 192, 0.03) 0%, transparent 70%)',
          filter: 'blur(80px)',
        }}
      />
    </div>
  );
}