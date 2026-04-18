import { motion, AnimatePresence } from 'motion/react';
import { X, ChevronRight, ChevronLeft, SkipForward, Check } from 'lucide-react';
import { useState, useEffect } from 'react';
import { GlassCard } from './GlassCard';
import type { DomainParam } from '../lib/api';
import { sounds } from '../lib/audio';

interface ParamConfig extends DomainParam {
  key: string;
}

/**
 * A chip option as returned by /predict/domains after the BUG 1 fix.
 * Backend now returns [{label: "0 – 1 hr", value: 0.5}, ...]
 *
 * FIX (BUG 2): Previously the modal received raw {label,value} objects,
 * passed them directly into chip buttons as strings, and React rendered
 * them as "[object Object]". Now we read .label for display and .value
 * for the actual submitted parameter value.
 */
interface ChipOption {
  label: string;
  value: string | number;
}

interface ChipParameterModalProps {
  isOpen:           boolean;
  onClose:          () => void;
  parameters:       string[];         // display labels (legacy — still accepted)
  parameterConfigs: ParamConfig[];    // full schema from /predict/domains (used)
  currentStep:      number;           // 1-based, controlled by parent
  totalSteps:       number;
  onNext:           (values: string[]) => void;
  onPrevious:       () => void;
  onComplete:       (answers: Record<string, any>) => void;
  currentAnswers:   Record<string, any>;
}

export function ChipParameterModal({
  isOpen,
  onClose,
  parameterConfigs,
  currentStep,
  totalSteps,
  onNext,
  onPrevious,
  onComplete,
  currentAnswers,
}: ChipParameterModalProps) {
  // Internal accumulated answers — merges with parent's currentAnswers on open
  const [answers, setAnswers] = useState<Record<string, any>>(currentAnswers || {});
  // The label string of the currently highlighted chip (for visual selection)
  const [selectedLabel, setSelectedLabel] = useState<string>('');
  // Free-text field below the chips
  const [freeText, setFreeText] = useState('');

  // Sync with parent's currentAnswers only when modal opens (not on every parent rerender)
  useEffect(() => {
    if (isOpen) {
      setAnswers(currentAnswers || {});
    }
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // When the step changes, pre-fill selectedLabel and freeText from saved answers
  useEffect(() => {
    const cfg = parameterConfigs[currentStep - 1];
    if (!cfg) return;

    const existing = answers[cfg.key];
    if (existing != null) {
      // If the existing value matches a chip option's value, highlight that chip
      const opts = _getChipOptions(cfg);
      const matched = opts.find(o => String(o.value) === String(existing));
      setSelectedLabel(matched ? matched.label : '');
      // If it was a free-text entry (no chip match), show it in the text field
      setFreeText(matched ? '' : String(existing));
    } else {
      setSelectedLabel('');
      setFreeText('');
    }
  }, [currentStep, parameterConfigs]); // eslint-disable-line react-hooks/exhaustive-deps

  const cfg    = parameterConfigs[currentStep - 1];
  const isLast = currentStep === totalSteps;
  // Progress = proportion of steps *completed* (not current), so bar advances as steps finish
  const progress = ((currentStep - 1) / Math.max(totalSteps, 1)) * 100;

  if (!cfg) return null;

  // ── Normalise chip options ─────────────────────────────────
  function _getChipOptions(p: ParamConfig): ChipOption[] {
    if (p.options && p.options.length > 0) {
      return p.options.map((opt: any) => {
        if (typeof opt === 'object' && opt !== null && 'label' in opt) {
          // Correct format from fixed backend: {label, value}
          return { label: String(opt.label), value: opt.value ?? opt.label };
        }
        // Fallback: plain string
        return { label: String(opt), value: opt };
      });
    }
    // Numeric range — generate 5 evenly spaced options
    if (p.type === 'numeric' && Array.isArray(p.range) && p.range.length === 2) {
      const [min, max] = p.range;
      const step = (max - min) / 4;
      return [0, 1, 2, 3, 4].map(i => {
        const v = Math.round((min + i * step) * 10) / 10;
        return { label: String(v), value: v };
      });
    }
    return [];
  }

  const chipOptions = _getChipOptions(cfg);

  // ── Commit current step and advance ───────────────────────
  const commitAndAdvance = (skip = false) => {
    let value: any = null;

    if (!skip) {
      if (freeText.trim()) {
        // Free-text takes priority over chip selection if both filled
        value = freeText.trim();
        // Cast to number if the field type is numeric
        if (cfg.type === 'numeric' || cfg.type === 'number') {
          const n = parseFloat(value);
          if (!isNaN(n)) value = n;
        }
      } else if (selectedLabel) {
        // Find the chip option matching the selected label → return its .value
        const matched = chipOptions.find(o => o.label === selectedLabel);
        value = matched ? matched.value : selectedLabel;
      }
    }

    const updated = { ...answers, [cfg.key]: value };
    setAnswers(updated);

    if (isLast) {
      sounds.success();
      onComplete(updated);
    } else {
      sounds.click();
      // Pass string representation to parent's onNext for legacy compatibility
      onNext(value != null ? [String(value)] : []);
    }
  };

  const handleChipClick = (opt: ChipOption) => {
    sounds.click();
    // Toggle: clicking the same chip deselects it
    setSelectedLabel(prev => prev === opt.label ? '' : opt.label);
    setFreeText(''); // clear free text when a chip is selected
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Modal */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              transition={{ type: 'spring', bounce: 0.3 }}
              className="w-full max-w-lg"
            >
              <GlassCard variant="elevated" className="p-6">

                {/* ── Header ─────────────────────────────────────── */}
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold mb-0.5">Configure Parameters</h3>
                    <p className="text-xs text-muted-foreground">
                      Step {currentStep} of {totalSteps}
                    </p>
                  </div>
                  <button
                    onClick={onClose}
                    className="p-1.5 hover:bg-white/10 rounded-lg transition-colors"
                    aria-label="Close"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* ── Progress bar ────────────────────────────────── */}
                <div className="h-1 bg-white/10 rounded-full mb-5 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>

                {/* ── Parameter label ──────────────────────────────── */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-0.5">
                    {cfg.label || cfg.key.replace(/_/g, ' ')}
                  </h4>
                  <p className="text-[10px] text-muted-foreground">
                    {cfg.description ||
                      (cfg.type === 'numeric' && Array.isArray(cfg.range) && cfg.range.length === 2
                        ? `Range: ${cfg.range[0]} — ${cfg.range[1]}`
                        : chipOptions.length > 0
                        ? 'Select one option below'
                        : 'Enter a value')}
                    {cfg.weight === 'high' && (
                      <span className="ml-2 text-primary">• High impact</span>
                    )}
                    {/* Required indicator */}
                    {(cfg as any).required && (
                      <span className="ml-2 text-destructive/70">• Required</span>
                    )}
                  </p>
                </div>

                {/* ── Chip options ─────────────────────────────────── */}
                {chipOptions.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-4">
                    {chipOptions.map((opt) => {
                      const isSelected = selectedLabel === opt.label;
                      return (
                        <motion.button
                          key={opt.label}
                          onClick={() => handleChipClick(opt)}
                          className={`
                            px-3 py-1.5 rounded-lg text-xs border transition-all
                            flex items-center gap-1.5
                            ${isSelected
                              ? 'bg-primary/20 border-primary text-primary'
                              : 'bg-white/5 border-white/10 text-muted-foreground hover:border-white/30 hover:text-foreground'
                            }
                          `}
                          whileHover={{ scale: 1.04 }}
                          whileTap={{ scale: 0.96 }}
                          type="button"
                        >
                          {isSelected && <Check className="w-2.5 h-2.5 shrink-0" />}
                          {/* FIX: render opt.label (string), not opt itself (would be [object Object]) */}
                          {opt.label}
                        </motion.button>
                      );
                    })}
                  </div>
                )}

                {/* ── Free-text / numeric input ────────────────────── */}
                {cfg.type === 'text' || cfg.type === 'textarea' ? (
                  <textarea
                    placeholder={
                      (cfg as any).placeholder ||
                      `Enter ${cfg.label || cfg.key.replace(/_/g, ' ')}...`
                    }
                    value={freeText}
                    onChange={(e) => { setFreeText(e.target.value); setSelectedLabel(''); }}
                    rows={4}
                    className="w-full px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg
                               focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                               transition-all placeholder:text-muted-foreground/50 mb-5 resize-none"
                  />
                ) : (
                  <input
                    type={cfg.type === 'numeric' || cfg.type === 'number' ? 'number' : 'text'}
                    placeholder={
                      chipOptions.length > 0
                        ? 'Or type a custom value...'
                        : ((cfg as any).placeholder || `Enter ${cfg.label || cfg.key.replace(/_/g, ' ')}...`)
                    }
                    value={freeText}
                    onChange={(e) => { setFreeText(e.target.value); setSelectedLabel(''); }}
                    min={(cfg.range as any)?.[0]}
                    max={(cfg.range as any)?.[1]}
                    onKeyDown={(e) => { if (e.key === 'Enter') commitAndAdvance(); }}
                    className="w-full px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg
                               focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50
                               transition-all placeholder:text-muted-foreground/50 mb-5"
                  />
                )}

                {/* ── Navigation row ───────────────────────────────── */}
                <div className="flex items-center gap-2">
                  {/* Back */}
                  <motion.button
                    onClick={() => { sounds.click(); onPrevious(); }}
                    disabled={currentStep <= 1}
                    className="px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10
                               hover:bg-white/10 transition-colors disabled:opacity-30
                               flex items-center gap-1"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="button"
                  >
                    <ChevronLeft className="w-3 h-3" />
                    Back
                  </motion.button>

                  {/* Skip */}
                  <motion.button
                    onClick={() => commitAndAdvance(true)}
                    className="px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10
                               hover:bg-white/10 transition-colors flex items-center gap-1
                               text-muted-foreground"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="button"
                  >
                    <SkipForward className="w-3 h-3" />
                    Skip
                  </motion.button>

                  {/* Next / Finish */}
                  <motion.button
                    onClick={() => commitAndAdvance(false)}
                    disabled={
                      // Require a value only for required fields
                      !!(cfg as any).required &&
                      !selectedLabel &&
                      !freeText.trim()
                    }
                    className="flex-1 px-3 py-2 text-xs rounded-lg bg-primary text-black font-medium
                               flex items-center justify-center gap-1
                               disabled:opacity-40 disabled:cursor-not-allowed"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="button"
                  >
                    {isLast ? 'Finish' : 'Next'}
                    {!isLast && <ChevronRight className="w-3 h-3" />}
                  </motion.button>
                </div>

                {/* ── Answers summary at bottom of modal ───────────── */}
                {Object.keys(answers).length > 0 && (
                  <div className="mt-4 pt-3 border-t border-white/10">
                    <p className="text-[10px] text-muted-foreground mb-1.5">Configured so far:</p>
                    <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
                      {Object.entries(answers)
                        .filter(([, v]) => v != null)
                        .map(([k, v]) => (
                          <span
                            key={k}
                            className="px-2 py-0.5 rounded text-[10px] bg-primary/10 text-primary border border-primary/20"
                          >
                            {k.replace(/_/g, ' ')}: {String(v)}
                          </span>
                        ))}
                    </div>
                  </div>
                )}

              </GlassCard>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}