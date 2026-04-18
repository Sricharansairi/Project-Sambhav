import { motion } from 'motion/react';
import { TrendingUp } from 'lucide-react';

interface ReliabilityIndexProps {
  score: number; // 0-100
  suggestions: string[];
  isVisible: boolean;
}

export function ReliabilityIndex({ score, suggestions, isVisible }: ReliabilityIndexProps) {
  const getScoreColor = (score: number) => {
    if (score >= 75) return 'text-[#ffb7c5]'; // Cyan (Clear)
    if (score >= 50) return 'text-[#c0c0c0]'; // Silver (Moderate)
    if (score >= 30) return 'text-[#ffb7c5]'; // Pink/Red (Low)
    return 'text-[#ff6b6b]'; // Critical Red
  };

  const getScoreLabel = (score: number) => {
    if (score >= 75) return 'CLEAR';
    if (score >= 50) return 'MODERATE';
    if (score >= 30) return 'LOW';
    return 'CRITICAL';
  };

  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="mb-4 p-4 rounded-lg bg-white/5 border border-white/10"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-xs font-medium mb-1 text-muted-foreground">Reliability Index</h4>
          <div className="flex items-baseline gap-2">
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className={`text-2xl font-bold ${getScoreColor(score)}`}
            >
              {score}%
            </motion.span>
            <span className="text-[10px] text-muted-foreground">{getScoreLabel(score)}</span>
          </div>
        </div>
        
        {/* Circular Progress */}
        <div className="relative w-12 h-12">
          <svg className="transform -rotate-90 w-12 h-12">
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
              className="text-white/10"
            />
            <motion.circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 20}`}
              initial={{ strokeDashoffset: 2 * Math.PI * 20 }}
              animate={{ strokeDashoffset: 2 * Math.PI * 20 * (1 - score / 100) }}
              transition={{ duration: 1, ease: [0.25, 0.1, 0.25, 1] }}
              className={getScoreColor(score)}
              strokeLinecap="round"
            />
          </svg>
        </div>
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="space-y-1.5"
        >
          <div className="flex items-center gap-1.5">
            <TrendingUp className="w-3 h-3 text-[#ffb7c5]" />
            <span className="text-[10px] font-medium text-muted-foreground">
              Suggestions to Improve Accuracy
            </span>
          </div>
          <div className="space-y-1">
            {suggestions.map((suggestion, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + idx * 0.1 }}
                className="flex gap-2 text-[10px] text-foreground/70"
              >
                <div className="w-1 h-1 rounded-full bg-[#ffb7c5] mt-1 shrink-0" />
                <span>{suggestion}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
