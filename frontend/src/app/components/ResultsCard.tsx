import { motion } from 'motion/react';
import { TrendingUp, AlertCircle, Info, Download, Share2 } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { ProbabilityBar } from './ProbabilityBar';

interface PredictionResult {
  prediction: number;
  confidenceLower: number;
  confidenceUpper: number;
  baseRate: number;
  adjustedRate: number;
  insights: string[];
  warnings?: string[];
  recommendations: string[];
}

interface ResultsCardProps {
  result: PredictionResult;
  title?: string;
}

export function ResultsCard({ result, title = 'Prediction Results' }: ResultsCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-xl font-bold mb-2">{title}</h3>
          <p className="text-sm text-muted-foreground">
            Generated on {new Date().toLocaleDateString()} at {new Date().toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-2">
          <motion.button
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Download className="w-5 h-5" />
          </motion.button>
          <motion.button
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Share2 className="w-5 h-5" />
          </motion.button>
        </div>
      </div>

      {/* Main Prediction */}
      <GlassCard glow glowColor="rgba(0, 255, 242, 0.3)" className="p-8">
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', bounce: 0.4, delay: 0.2 }}
            className="text-6xl font-bold text-[#ffb7c5] mb-2"
          >
            {result.prediction.toFixed(1)}%
          </motion.div>
          <div className="text-muted-foreground">Predicted Probability</div>
        </div>

        <div className="space-y-4">
          <ProbabilityBar
            value={result.baseRate}
            label="Base Rate"
            color="#00fff2"
            delay={0.3}
          />
          <ProbabilityBar
            value={result.adjustedRate}
            label="Adjusted for Context"
            color="#9d4eff"
            delay={0.5}
          />
        </div>
      </GlassCard>

      {/* Confidence Interval */}
      <GlassCard variant="elevated" className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-5 h-5 text-accent" />
          <h4 className="font-medium">90% Confidence Interval</h4>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="text-3xl font-bold text-accent mb-2"
            >
              {result.confidenceLower.toFixed(1)}%
            </motion.div>
            <div className="text-sm text-muted-foreground">Lower Bound</div>
          </div>

          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="text-3xl font-bold text-[#ffb7c5] mb-2"
            >
              {result.confidenceUpper.toFixed(1)}%
            </motion.div>
            <div className="text-sm text-muted-foreground">Upper Bound</div>
          </div>
        </div>

        {/* Visual Range */}
        <div className="mt-6 relative h-3 bg-white/5 rounded-full">
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1, delay: 0.9 }}
            className="absolute h-full rounded-full bg-gradient-to-r from-accent to-warning"
            style={{
              left: `${result.confidenceLower}%`,
              width: `${result.confidenceUpper - result.confidenceLower}%`,
            }}
          />
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 1.2, type: 'spring' }}
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-[#ffb7c5] rounded-full border-2 border-black"
            style={{ left: `${result.prediction}%`, marginLeft: '-8px' }}
          />
        </div>
      </GlassCard>

      {/* Insights */}
      {result.insights && result.insights.length > 0 && (
        <GlassCard variant="elevated" className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-[#c0c0c0]" />
            <h4 className="font-medium">Key Insights</h4>
          </div>
          <div className="space-y-3">
            {result.insights.map((insight, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.3 + i * 0.1 }}
                className="flex gap-3 text-sm"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-success mt-2 shrink-0" />
                <span className="text-foreground/80">{insight}</span>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Warnings */}
      {result.warnings && result.warnings.length > 0 && (
        <GlassCard variant="elevated" className="p-6 border-warning/30">
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-[#ffb7c5]" />
            <h4 className="font-medium text-[#ffb7c5]">Important Considerations</h4>
          </div>
          <div className="space-y-3">
            {result.warnings.map((warning, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.5 + i * 0.1 }}
                className="flex gap-3 text-sm"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-2 shrink-0" />
                <span className="text-foreground/80">{warning}</span>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Recommendations */}
      {result.recommendations && result.recommendations.length > 0 && (
        <GlassCard variant="elevated" className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-5 h-5 text-[#ffb7c5]" />
            <h4 className="font-medium">Recommendations</h4>
          </div>
          <div className="space-y-3">
            {result.recommendations.map((rec, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.7 + i * 0.1 }}
                className="flex gap-3 text-sm"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-[#ffb7c5] mt-2 shrink-0" />
                <span className="text-foreground/80">{rec}</span>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      )}
    </motion.div>
  );
}
