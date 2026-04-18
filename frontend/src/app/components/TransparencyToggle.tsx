import { motion } from 'motion/react';

interface TransparencyToggleProps {
  value: 'simple' | 'detailed' | 'full';
  onChange: (value: 'simple' | 'detailed' | 'full') => void;
}

export function TransparencyToggle({ value, onChange }: TransparencyToggleProps) {
  const options = [
    { id: 'simple' as const, label: 'Simple' },
    { id: 'detailed' as const, label: 'Detailed' },
    { id: 'full' as const, label: 'Full Breakdown' }
  ];

  return (
    <div className="inline-flex bg-white/5 rounded-lg p-0.5 border border-white/10">
      {options.map((option) => (
        <motion.button
          key={option.id}
          onClick={() => onChange(option.id)}
          className={`
            relative px-3 py-1 text-[11px] rounded-md transition-colors
            ${value === option.id 
              ? 'text-black' 
              : 'text-muted-foreground hover:text-foreground'
            }
          `}
          whileHover={{ scale: value === option.id ? 1 : 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {value === option.id && (
            <motion.div
              layoutId="active-toggle"
              className="absolute inset-0 bg-[#00fff2] rounded-md"
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            />
          )}
          <span className="relative z-10">{option.label}</span>
        </motion.button>
      ))}
    </div>
  );
}
