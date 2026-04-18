import { motion } from 'motion/react';
import { Brain, Zap, GitMerge, TrendingUp, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { useState } from 'react';

interface PredictionBreakdownProps {
  mlProbability?: number;       // 0-100
  llmProbability?: number;      // 0-100
  reconciledProbability?: number; // 0-100
  reliabilityIndex?: number;    // 0-100
  gap?: number;
  reconciliationMethod?: string;
  shapValues?: Record<string, number>;
  mode?: string;
  delay?: number;
}

function Bar({ value, color, delay }: { value: number; color: string; delay: number }) {
  return (
    <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        transition={{ delay, duration: 0.8, ease: 'easeOut' }}
        className="h-full rounded-full"
        style={{ background: color }}
      />
    </div>
  );
}

export function PredictionBreakdown({
  mlProbability,
  llmProbability,
  reconciledProbability,
  reliabilityIndex,
  gap,
  reconciliationMethod,
  shapValues,
  mode = 'guided',
  delay = 0,
}: PredictionBreakdownProps) {
  const [expanded, setExpanded] = useState(false);

  // Only render if we have meaningful data
  const hasData = mlProbability !== undefined || llmProbability !== undefined || reliabilityIndex !== undefined;
  if (!hasData) return null;

  const mlPct   = mlProbability   ?? 50;
  const llmPct  = llmProbability  ?? 50;
  const recPct  = reconciledProbability ?? Math.round((mlPct + llmPct) / 2);
  const ri      = reliabilityIndex ?? 0;
  const gapAbs  = gap !== undefined ? Math.abs(gap * 100) : Math.abs(mlPct - llmPct);
  const method  = reconciliationMethod || (gapAbs > 15 ? 'Weighted blend (high gap)' : 'Weighted average');

  const topShap = shapValues
    ? Object.entries(shapValues)
        .sort(([, a], [, b]) => Math.abs(b as number) - Math.abs(a as number))
        .slice(0, 4)
    : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-xl overflow-hidden border border-white/10 bg-white/5 backdrop-blur-md"
    >
      {/* Header row — always visible */}
      <button
        onClick={() => setExpanded(p => !p)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/10 transition-colors text-left"
      >
        <div className="flex items-center justify-center w-6 h-6 rounded-lg bg-white/5 border border-white/10">
          <Info className="w-3.5 h-3.5 text-muted-foreground" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-semibold text-foreground/90">How was this generated?</p>
          <p className="text-[9px] text-muted-foreground">ML {mlPct.toFixed(0)}% · LLM {llmPct.toFixed(0)}% · RI {ri.toFixed(0)}%</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Mini inline bars */}
          <div className="hidden sm:flex items-center gap-1 w-24">
            <span className="text-[8px] text-muted-foreground shrink-0">ML</span>
            <Bar value={mlPct} color="#ffb7c5" delay={delay + 0.2} />
            <span className="text-[8px] text-muted-foreground shrink-0">LLM</span>
            <Bar value={llmPct} color="#00fff2" delay={delay + 0.3} />
          </div>
          {expanded ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </div>
      </button>

      {/* Expanded panel */}
      {expanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.25 }}
          className="px-4 pb-4 space-y-4 border-t border-white/10 pt-3"
        >
          {/* Three-column stat grid */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: 'ML Model', value: `${mlPct.toFixed(1)}%`, icon: Brain, color: '#ffb7c5', bar: mlPct },
              { label: 'LLM Layer', value: `${llmPct.toFixed(1)}%`, icon: Zap, color: '#00fff2', bar: llmPct },
              { label: 'Reconciled', value: `${recPct.toFixed(1)}%`, icon: GitMerge, color: '#c0c0c0', bar: recPct },
            ].map(({ label, value, icon: Icon, color, bar }) => (
              <div key={label} className="rounded-lg bg-black/20 border border-white/5 p-2.5 space-y-2">
                <div className="flex items-center gap-1.5">
                  <Icon className="w-3 h-3" style={{ color }} />
                  <p className="text-[9px] text-muted-foreground uppercase tracking-wide font-medium">{label}</p>
                </div>
                <p className="text-base font-bold text-foreground">{value}</p>
                <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${bar}%` }}
                    transition={{ delay: 0.1, duration: 0.7, ease: 'easeOut' }}
                    className="h-full rounded-full"
                    style={{ background: color }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Reliability + Gap */}
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-black/20 border border-white/5 p-2.5">
              <div className="flex items-center gap-1.5 mb-1.5">
                <TrendingUp className="w-3 h-3 text-[#f0f0f0]" />
                <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Reliability Index</p>
              </div>
              <div className="flex items-center gap-2">
                <p className="text-sm font-bold text-[#f0f0f0]">{ri.toFixed(0)}%</p>
                <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${ri}%` }}
                    transition={{ delay: 0.2, duration: 0.7 }}
                    className="h-full rounded-full bg-[#f0f0f0]"
                  />
                </div>
              </div>
            </div>
            <div className="rounded-lg bg-black/20 border border-white/5 p-2.5">
              <p className="text-[9px] text-muted-foreground uppercase tracking-wide mb-1.5">ML↔LLM Gap</p>
              <p className={`text-sm font-bold ${gapAbs > 20 ? 'text-[#ff6b6b]' : 'text-[#f0f0f0]'}`}>
                {gapAbs.toFixed(1)}%
              </p>
              <p className="text-[9px] text-muted-foreground/70 mt-0.5 leading-tight">{method}</p>
            </div>
          </div>

          {/* SHAP top drivers */}
          {topShap.length > 0 && (
            <div className="rounded-lg bg-black/20 border border-white/5 p-2.5 space-y-1.5">
              <p className="text-[9px] text-muted-foreground uppercase tracking-wide font-medium">Top ML Drivers (SHAP)</p>
              {topShap.map(([k, v]) => {
                const pct = Math.abs((v as number) * 100);
                const positive = (v as number) > 0;
                return (
                  <div key={k} className="flex items-center gap-2">
                    <span className="text-[9px] text-muted-foreground min-w-[90px] truncate">{k.replace(/_/g, ' ')}</span>
                    <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(100, pct * 5)}%` }}
                        transition={{ duration: 0.6 }}
                        className="h-full rounded-full"
                        style={{ background: positive ? '#c0c0c0' : '#ff6b6b' }}
                      />
                    </div>
                    <span className={`text-[9px] font-medium ${positive ? 'text-[#c0c0c0]' : 'text-[#ff6b6b]'}`}>
                      {positive ? '+' : ''}{(v as number).toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          <p className="text-[8px] text-muted-foreground/40 text-center italic">
            ML: XGBoost/LightGBM ensemble · LLM: Multi-provider reasoning · Reconciliation: {method}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}
