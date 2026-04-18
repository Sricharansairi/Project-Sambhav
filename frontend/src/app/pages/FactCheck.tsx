import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';
import { Search, CheckCircle, XCircle, AlertCircle, Sparkles, Loader2, Shield, ChevronRight } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { LoadingAnimation } from '../components/LoadingAnimation';
import { ResultChatbot } from '../components/ResultChatbot';
import { PredictionBreakdown } from '../components/PredictionBreakdown';
import { MessageCircle } from 'lucide-react';
import { CREDIBILITY_DIMENSIONS } from '../lib/constants';
import { factCheck, SambhavAPIError, type FactCheckResult, type FactCheckSource } from '../lib/api';

export function FactCheck() {
  const [statement,   setStatement]   = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [result,      setResult]      = useState<FactCheckResult | null>(null);
  const [apiError,    setApiError]    = useState<string | null>(null);
  const [chatOpen,    setChatOpen]    = useState(false);

  const handleAnalyze = async () => {
    const trimmed = statement.trim();
    if (!trimmed || trimmed.length < 8) {
      setApiError('Please enter a meaningful claim to verify (at least 8 characters).');
      return;
    }

    setIsAnalyzing(true);
    setShowResults(false);
    setResult(null);
    setApiError(null);

    try {
      const res = await factCheck(trimmed);
      setResult(res.result);
      setShowResults(true);
    } catch (err) {
      if (err instanceof SambhavAPIError) {
        setApiError(err.isBlocked
          ? `Blocked: ${err.message}`
          : `Error ${err.status}: ${err.message}`);
      } else {
        setApiError('Fact-check failed. Is the backend running?');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) handleAnalyze();
  };

  const getVerdict = (score: number) => {
    if (score >= 85) return { label: 'TRUE',         color: '#00ff88', icon: CheckCircle };
    if (score >= 70) return { label: 'MOSTLY TRUE',  color: '#00d9ff', icon: CheckCircle };
    if (score >= 50) return { label: 'MIXED',        color: '#ff6b35', icon: AlertCircle };
    if (score >= 30) return { label: 'MOSTLY FALSE', color: '#ff6b35', icon: AlertCircle };
    return               { label: 'FALSE',           color: '#ff4757', icon: XCircle };
  };

  // Build dimensions array from real result or fallback to constants list
  const getDimensions = () => {
    if (!result) return [];
    const dims = result.dimensions || {};
    // Map backend dimension keys → display names from constants
    return CREDIBILITY_DIMENSIONS.map(c => {
      const backendData = dims[c.id] || {};
      return {
        id:        c.id,
        name:      c.name,
        weight:    c.weight,
        score:     typeof backendData === 'object' ? (backendData.score ?? 60) : Number(backendData) || 60,
        reasoning: typeof backendData === 'object' ? backendData.reasoning || '' : '',
      };
    });
  };

  const dimensions = getDimensions();
  const score   = result?.credibility_score ?? 0;
  const verdict = getVerdict(score);
  const VerdictIcon = verdict.icon;

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-24 pb-12 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/30 mb-6"
              style={{ boxShadow: '0 0 20px rgba(208, 208, 208, 0.15)' }}
            >
              <Sparkles className="w-4 h-4 text-accent" />
              <span className="text-sm text-accent font-medium">CLAIM ANALYSIS MODULE</span>
            </motion.div>

            <h1 className="text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">
                Fact-Check Engine
              </span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              8-dimensional credibility analysis with dual-LLM cross-validation and live web evidence
            </p>
          </motion.div>

          {/* Input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            <GlassCard variant="elevated" className="p-6">
              <label className="block text-sm font-medium mb-3">Enter Statement to Verify</label>
              <div className="relative">
                <textarea
                  value={statement}
                  onChange={(e) => setStatement(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter a statement, claim, or fact you want to verify... (Ctrl+Enter to analyse)"
                  className="w-full h-32 px-4 py-3 pr-12 bg-white/5 border border-white/10 rounded-lg
                           focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent/50
                           transition-all resize-none placeholder:text-muted-foreground/50"
                />
                <Search className="absolute right-4 top-4 w-6 h-6 text-muted-foreground" />
              </div>

              {/* Error */}
              <AnimatePresence>
                {apiError && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mt-2 flex items-center gap-2 text-xs text-destructive"
                  >
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    {apiError}
                  </motion.div>
                )}
              </AnimatePresence>

              <motion.button
                onClick={handleAnalyze}
                disabled={!statement.trim() || isAnalyzing}
                className="w-full mt-4 px-6 py-3 rounded-lg bg-accent text-black font-medium
                         flex items-center justify-center gap-2 disabled:opacity-50"
                whileHover={!isAnalyzing && statement.trim() ? { scale: 1.02 } : {}}
                whileTap={!isAnalyzing && statement.trim() ? { scale: 0.98 } : {}}
              >
                {isAnalyzing ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /><span>Analysing...</span></>
                ) : (
                  <><Search className="w-4 h-4" /><span>Analyse Claim</span></>
                )}
              </motion.button>
            </GlassCard>
          </motion.div>

          {/* Loading */}
          {isAnalyzing && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-8">
              <GlassCard variant="elevated" className="p-8 flex items-center justify-center">
                <LoadingAnimation />
              </GlassCard>
            </motion.div>
          )}

          {/* Results */}
          <AnimatePresence>
            {showResults && result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {/* Verdict Card */}
                <GlassCard variant="elevated" className="p-8">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <VerdictIcon className="w-8 h-8" style={{ color: verdict.color }} />
                        <span className="text-4xl font-bold" style={{ color: verdict.color }}>
                          {result.verdict || verdict.label}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground max-w-xl">{result.explanation || result.summary || ''}</p>
                      {/* Sources Section Removed as requested */}
                    </div>
                    <div className="text-right">
                      <div className="text-5xl font-bold" style={{ color: verdict.color }}>
                        {score.toFixed(1)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Credibility Score</div>
                    </div>
                  </div>
                </GlassCard>

                {/* 8-Dimension Analysis */}
                <div>
                  <h2 className="text-xl font-bold mb-4">8-Dimension Analysis</h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
                    {dimensions.map((dim, index) => (
                      <motion.div
                        key={dim.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.08 }}
                      >
                        <GlassCard variant="elevated" className="p-5 h-full">
                          <div className="flex items-center justify-between mb-3">
                            <div>
                              <h3 className="text-sm font-medium">{dim.name}</h3>
                              <p className="text-[10px] text-muted-foreground">
                                Weight: {Math.round(dim.weight * 100)}%
                              </p>
                            </div>
                            <span className="text-xl font-bold text-primary">
                              {dim.score.toFixed(1)}%
                            </span>
                          </div>
                          <ProbabilityBar 
                            value={dim.score} 
                            label={`Confidence: ${dim.name}`}
                            delay={index * 0.08 + 0.3} 
                          />
                          {dim.reasoning && (
                            <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed line-clamp-2 hover:line-clamp-none transition-all">
                              {dim.reasoning}
                            </p>
                          )}
                        </GlassCard>
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Prediction Breakdown & Chatbot */}
                <PredictionBreakdown 
                  mode="factcheck"
                  mlProbability={result.credibility_score}
                  llmProbability={result.credibility_score}
                  reliabilityIndex={Math.round((result.credibility_score + 100) / 2)}
                  delay={0.6}
                />

                <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 mb-6 px-3 py-2 text-xs rounded-xl bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-all flex items-center justify-center gap-1.5 font-medium">
                  <MessageCircle className="w-4 h-4" /> Ask the Fact Checker
                </motion.button>

                <ResultChatbot 
                  isOpen={chatOpen} 
                  onClose={() => setChatOpen(false)} 
                  context={result}
                  mode="Fact Check"
                  domain="General Fact Verification"
                  title="Fact Verification Assistant"
                />

                {/* Weighted score note */}
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8 }}
                  className="text-[10px] text-muted-foreground text-center italic"
                >
                  * Credibility score is a weighted average across all 8 dimensions using dual-LLM cross-validation.
                  Sambhav may be incorrect — always verify independently.
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

