import { motion } from 'motion/react';
import { AlertTriangle } from 'lucide-react';

export function StickyDisclaimer() {
  return (
    <motion.div
      initial={{ y: 100 }}
      animate={{ y: 0 }}
      transition={{ delay: 1, duration: 0.5 }}
      className="fixed bottom-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-md 
                 border-t border-warning/30 px-4 py-2"
    >
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-2">
        <AlertTriangle className="w-3 h-3 text-[#ffb7c5] shrink-0" />
        <p className="text-[10px] text-[#ffb7c5]/90 text-center">
          <span className="font-medium">Sambhav may be incorrect.</span> Always verify critical predictions with domain experts. This system provides probabilistic estimates, not certainties.
        </p>
      </div>
    </motion.div>
  );
}
