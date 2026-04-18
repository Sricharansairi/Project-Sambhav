import { motion, AnimatePresence } from 'motion/react';
import { useState, useEffect } from 'react';
import { CheckCircle2, Loader2 } from 'lucide-react';

const PIPELINE_STAGES = [
  "Ingesting and sanitizing inputs...",
  "Validating domain registry...",
  "Feature engineering & extrapolation...",
  "Running ML core ensembles...",
  "Querying LLM semantic layer...",
  "Reconciling model divergence...",
  "Calculating Monte Carlo stability..."
];

export function LoadingAnimation({ durationMs = 800 }: { durationMs?: number }) {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    // We want to progress through all stages within the duration.
    const timePerStage = durationMs / PIPELINE_STAGES.length;
    
    if (currentStage < PIPELINE_STAGES.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStage(prev => prev + 1);
      }, timePerStage);
      return () => clearTimeout(timer);
    }
  }, [currentStage, durationMs]);

  return (
    <div className="flex flex-col items-center justify-center p-8 w-full max-w-md mx-auto">
      
      {/* Central Core Spinner */}
      <div className="relative w-24 h-24 mb-10">
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute inset-0 border border-primary/30 rounded-full"
            style={{
              borderTopColor: i === 0 ? 'var(--primary, #c0c0c0)' : 'transparent',
              borderRightColor: i === 1 ? 'var(--primary, #c0c0c0)' : 'transparent',
              borderBottomColor: i === 2 ? 'var(--primary, #c0c0c0)' : 'transparent',
            }}
            animate={{
              rotate: [0, 360],
              scale: [1, 1.05, 1],
            }}
            transition={{
              duration: 2 - (i * 0.4),
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        ))}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          animate={{ scale: [0.9, 1.1, 0.9], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        >
          <div className="w-8 h-8 rounded-full bg-[#ffb7c5]/20 blur-sm" />
          <div className="absolute w-3 h-3 rounded-full bg-[#ffb7c5]" />
        </motion.div>
      </div>

      {/* Stage Listing */}
      <div className="w-full space-y-3">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isCompleted = idx < currentStage;
          const isActive = idx === currentStage;
          const isPending = idx > currentStage;

          return (
            <div key={idx} className="flex items-center gap-3">
              <div className="w-5 flex justify-center flex-shrink-0">
                {isCompleted ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring" }}
                  >
                    <CheckCircle2 className="w-4 h-4 text-[#c0c0c0]" />
                  </motion.div>
                ) : isActive ? (
                  <Loader2 className="w-4 h-4 text-[#ffb7c5] animate-spin" />
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-white/10" />
                )}
              </div>
              
              <div className="flex-1">
                <motion.div 
                  className={`text-xs ${
                    isCompleted ? "text-muted-foreground" : 
                    isActive ? "text-foreground font-medium" : 
                    "text-muted-foreground/40"
                  }`}
                  animate={{ opacity: isPending ? 0.4 : 1 }}
                >
                  {stage}
                </motion.div>
                
                {/* Active progress bar underneath the current stage */}
                <AnimatePresence>
                  {isActive && (
                    <motion.div 
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 2 }}
                      exit={{ opacity: 0, height: 0 }}
                      className="w-full bg-white/10 overflow-hidden mt-1.5 rounded-full"
                    >
                      <motion.div 
                        className="h-full bg-[#ffb7c5] rounded-full"
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: durationMs / PIPELINE_STAGES.length / 1000, ease: "linear" }}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          );
        })}
      </div>
      
    </div>
  );
}
