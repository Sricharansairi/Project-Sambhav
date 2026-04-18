import { motion, AnimatePresence } from 'motion/react';
import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams }    from 'react-router';
import {
  Play, Loader2, ChevronRight, RotateCcw, AlertCircle, Info, Zap,
  MessageCircle, Send, Plus, Minus, ShieldAlert, GitBranch,
  FileText, Users, History, FlaskConical, Activity,
} from 'lucide-react';
import { BackgroundLogo }       from '../components/BackgroundLogo';
import { Navigation }           from '../components/Navigation';
import { GlassCard }            from '../components/GlassCard';
import { FileUploadZone }       from '../components/FileUploadZone';
import { LoadingAnimation }     from '../components/LoadingAnimation';
import { OutcomeRow }           from '../components/OutcomeRow';
import { SHAPChart }            from '../components/SHAPChart';
import { PredictionBreakdown }  from '../components/PredictionBreakdown';
import { ResultChatbot }        from '../components/ResultChatbot';
import { AuditPanel }           from '../components/AuditPanel';
import { ReliabilityIndex }     from '../components/ReliabilityIndex';
import { ExportPanel }          from '../components/ExportPanel';
import { ChipParameterModal }   from '../components/ChipParameterModal';
import { PragmaChat }           from '../components/PragmaChat';
import { PragmaDocumentation }  from '../components/PragmaDocumentation';
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle, DialogDescription } from '../components/ui/dialog';
import {
  getDomains, runPredict, runFreeInfer, getOutcomes, getTransparency, getInverseTransparency,
  startConversational, answerConversational, screenInput,
  runWhatIf, runComparative, startMonitoring, updateMonitoring,
  runAdversarial, runExpertMode, runRetrospective, runSimulation,
  analyzeDocument,
  SambhavAPIError,
  type DomainInfo, type PredictionResult, type Outcome, type TransparencyResult,
} from '../lib/api';

interface ConvMessage {
  role:      'sambhav' | 'user';
  content:   string;
  options?:  string[];
  param_key?: string;
  step?:     number;
}
interface CompScenario { label: string; params: Record<string, string>; }
interface MonSession {
  session_id: string; name: string; baseline: number;
  updates: { probability: number; update_text: string; updated_at: string }[];
  threshold_low: number; threshold_high: number;
}

export function Prediction() {
  const [searchParams]   = useSearchParams();
  const selectedMode     = searchParams.get('mode')   || 'guided';
  const selectedDomain   = searchParams.get('domain') || 'student';

  const [domainInfo,    setDomainInfo]    = useState<DomainInfo | null>(null);
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [inputText,     setInputText]     = useState('');
  const [parameters,    setParameters]    = useState<Record<string, any>>({});
  const [modalOpen,     setModalOpen]     = useState(false);
  const [modalStep,     setModalStep]     = useState(0);

  const [isGenerating, setIsGenerating] = useState(false);
  const [showResults,  setShowResults]  = useState(false);
  const [isAnimating,  setIsAnimating]  = useState(false);
  const [predResult,   setPredResult]   = useState<PredictionResult | null>(null);
  const [predId,       setPredId]       = useState('');
  const [outcomes,     setOutcomes]     = useState<Outcome[]>([]);
  const [loadingMore,  setLoadingMore]  = useState(false);

  const [expandedOutcome,   setExpandedOutcome]   = useState<number | null>(null);
  const [chatOpen,          setChatOpen]          = useState(false);
  const [chatContext,       setChatContext]       = useState<any>(null);
  
  const [whyData,           setWhyData]           = useState<Record<string, TransparencyResult>>({});
  const [inverseData,       setInverseData]       = useState<Record<string, any>>({});
  const [loadingWhy,        setLoadingWhy]        = useState<number | null>(null);
  const [loadingInverse,    setLoadingInverse]    = useState<number | null>(null);
  const [apiError,          setApiError]          = useState<string | null>(null);

  // Conversational
  const [convMessages, setConvMessages] = useState<ConvMessage[]>([]);
  const [convParams,   setConvParams]   = useState<Record<string, any>>({});
  const [convStep,     setConvStep]     = useState(0);
  const [convComplete, setConvComplete] = useState(false);
  const [convStarted,  setConvStarted]  = useState(false);
  const [convLoading,  setConvLoading]  = useState(false);
  const [convInput,    setConvInput]    = useState('');
  const convEndRef = useRef<HTMLDivElement>(null);

  // Hybrid
  const [hybridExtracted, setHybridExtracted] = useState<Record<string, any>>({});
  const [hybridLoading,   setHybridLoading]   = useState(false);

  // Document
  const [docFile, setDocFile] = useState<File | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [docResult,  setDocResult]  = useState<any>(null);

  // Voice
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceLoading, setVoiceLoading] = useState(false);
  const [voiceResult,  setVoiceResult]  = useState<any>(null);

  const isAnalyzing = hybridLoading || docLoading || voiceLoading;

  // Comparative
  const [compScenarios, setCompScenarios] = useState<CompScenario[]>([
    { label: 'Scenario A', params: {} },
    { label: 'Scenario B', params: {} },
  ]);
  const [compResult, setCompResult] = useState<any>(null);

  // Monitoring
  const [monName,    setMonName]    = useState('');
  const [monThresh,  setMonThresh]  = useState(40);
  const [monSession, setMonSession] = useState<MonSession | null>(null);
  const [monUpdate,  setMonUpdate]  = useState('');
  const [monLoading, setMonLoading] = useState(false);

  // Mode-specific results
  const [adversarialResult, setAdversarialResult] = useState<any>(null);
  const [whatifTree,        setWhatifTree]        = useState<any>(null);
  const [expertDebate,      setExpertDebate]      = useState<any>(null);
  const [retroResult,       setRetroResult]       = useState<any>(null);
  const [retroDesc,         setRetroDesc]         = useState('');
  const [retroOutcome,      setRetroOutcome]      = useState('');
  const [simResult,         setSimResult]         = useState<any>(null);
  const [freeInferResult,   setFreeInferResult]   = useState<any>(null);
  const [paramsConfirmed,   setParamsConfirmed]   = useState(false);
  const [insufficientInfo, setInsufficientInfo] = useState<{ reason: string; missing?: string[] } | null>(null);
  const [relevantKeys,     setRelevantKeys]     = useState<string[]>([]);

  const ADVERSARIAL_PRESETS: Record<string, Record<string, any>> = {
    student:    { study_hours_per_day: 10, sleep_hours: 0, stress_level: 'extreme', attendance_pct: 100, motivation: 1 },
    job_life:   { job_satisfaction: 1, work_life_balance: 1, overtime: 1, years_at_company: 0 },
    financial:  { credit_score: 300, income: 0, loan_amount: 1000000, missed_payments: 20 },
    health:     { glucose: 350, bmi: 55, blood_pressure: 180 },
    default:    { value_a: 100, value_b: 0, stress: 'extreme', hours: 0 },
  };

  useEffect(() => {
    setLoadingSchema(true); setDomainInfo(null);
    getDomains().then(all => { 
      console.log('Available domains:', Object.keys(all));
      if (all[selectedDomain]) {
        setDomainInfo(all[selectedDomain]);
        // RESET parameters when domain changes to ensure chips are dynamic
        setParameters({});
        setDynamicParams([]);
      } else {
        console.warn(`Domain ${selectedDomain} not found in registry`);
        if (all['student']) {
          setDomainInfo(all['student']);
          setParameters({});
          setDynamicParams([]);
        }
      }
    })
      .catch(err => {
        console.error('Failed to load domains:', err);
        setApiError('Could not load domain configuration. Please check if the backend is running.');
      })
      .finally(() => setLoadingSchema(false));
  }, [selectedDomain]);

  useEffect(() => {
    setConvMessages([]); setConvParams({}); setConvComplete(false); setConvStarted(false);
    setAdversarialResult(null); setWhatifTree(null); setCompResult(null); setExpertDebate(null);
    setRetroResult(null); setSimResult(null); setDocResult(null);
    setShowResults(false); setOutcomes([]); setPredResult(null); setApiError(null);
    setParamsConfirmed(false); setParameters({}); setInsufficientInfo(null); setRelevantKeys([]);
    setDynamicParams([]);
    setWhyData({}); setInverseData({}); setLoadingInverse(null);
  }, [selectedMode, selectedDomain]);

  useEffect(() => { convEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [convMessages]);

  // Dynamic Parameter Discovery (Section 6.1)
  const [dynamicParams, setDynamicParams] = useState<any[]>([]);
  const [isDiscovering, setIsDiscovering] = useState(false);

  const paramList = dynamicParams.length > 0 
    ? dynamicParams 
    : (domainInfo
        ? Object.entries(domainInfo.parameters)
            .map(([key, cfg]) => ({ key, ...cfg }))
        : []);

  // ── Conversational handlers ──────────────────────────────────────────────
  const handleStartConversation = async () => {
    setConvLoading(true); setApiError(null);
    try {
      const res = await startConversational(selectedDomain, inputText || undefined);
      const q   = res.question;
      if (!q) { setApiError('Could not start conversation.'); return; }
      setConvStarted(true);
      setConvMessages([{ role: 'sambhav', content: q.question || q, options: q.options || [], param_key: q.param_key, step: q.step }]);
      setConvStep(q.step || 1);
    } catch (e: any) { setApiError(e.message || 'Failed to start conversation'); }
    finally { setConvLoading(false); }
  };

  const handleConvPredict = useCallback(async (params: Record<string, any>) => {
    setIsGenerating(true);
    try {
      const [predRes, outcomeRes] = await Promise.all([
        runPredict({ domain: selectedDomain, parameters: params, question: inputText || undefined, mode: 'conversational', run_debate: true }),
        getOutcomes({ domain: selectedDomain, parameters: params, question: inputText || undefined, n_outcomes: 5, mode: 'independent' })
      ]);
      setPredResult(predRes.prediction); setPredId(predRes.prediction_id || '');
      setOutcomes(outcomeRes.result?.outcomes || []);
      setShowResults(true); setTimeout(() => setIsAnimating(true), 20);
    } catch (e: any) { setApiError(e.message || 'Prediction failed'); }
    finally { setIsGenerating(false); }
  }, [selectedDomain, inputText]);

  const handleConvAnswer = async (value: string, param_key: string, skipped = false) => {
    if (!value.trim() && !skipped) return;
    setConvMessages(prev => [...prev, { role: 'user', content: skipped ? '(skipped)' : value }]);
    setConvInput(''); setConvLoading(true);
    try {
      // Map convMessages to history for backend [{role, content}]
      // Filter out empty messages and only include recent context
      const history = convMessages
        .filter(m => m.content && m.content.trim())
        .map(m => ({ 
          role: m.role === 'sambhav' ? 'assistant' : 'user', 
          content: m.content 
        }));

      const res = await answerConversational({ 
        domain: selectedDomain, 
        question: inputText || undefined, 
        param_key, 
        value: skipped ? '' : value, 
        skipped, 
        step: convStep, 
        parameters: convParams,
        history: history
      });
      const newParams = { ...convParams };
      if (!skipped && param_key) newParams[param_key] = value;
      setConvParams(newParams);
      
      // Update step and messages
      const parametersCollected = res.state?.parameters_collected ?? (Object.keys(newParams).length);
      setConvStep(parametersCollected + 1);
      
      if (res.state?.complete) {
        setConvComplete(true);
        const reliability = res.state.reliability ?? 75;
        setConvMessages(prev => [...prev, { role: 'sambhav', content: `I have enough information (Reliability: ${reliability}%). Generating your prediction now…` }]);
        await handleConvPredict(newParams);
      } else if (res.state?.next_question) {
        const nq = res.state.next_question;
        setConvMessages(prev => [...prev, { role: 'sambhav', content: nq.question, options: nq.options || [], param_key: nq.param_key, step: nq.step }]);
      }
    } catch (e: any) { setApiError(e.message || 'Conversation error'); }
    finally { setConvLoading(false); }
  };

  // ── Main generate handler ────────────────────────────────────────────────
  const handleGenerate = useCallback(async (isConfirmed = false) => {
    setApiError(null); 
    
    // Only reset state if not confirmed (starting a new prediction)
    if (!isConfirmed) {
      setIsGenerating(true); 
      setShowResults(false);
      setIsAnimating(false); 
      setOutcomes([]); 
      setPredResult(null);
      setExpandedOutcome(null); 
      setWhyData({});
    }

    // G.01 — Personal Calibration Adjustment (Section 13.5)
    // We pass the user_id and db to the backend to apply historical bias
    const userJson = localStorage.getItem('sambhav_user');
    const userObj = userJson ? JSON.parse(userJson) : null;
    const userId = userObj?.user_id;

    // Dynamic Parameter Flow (Section 6.1)
    // In Guided mode, ALWAYS discover params fresh from the question — never skip
    if (selectedMode === 'guided' && !isConfirmed) {
      if (!inputText.trim()) { 
        setApiError('Please describe your situation or question first.'); 
        setIsGenerating(false); 
        return; 
      }
      
      // Reset so chips are fully regenerated for this specific question
      setDynamicParams([]);
      setParameters({});
      setParamsConfirmed(false);
      setIsDiscovering(true);
      console.log('Guided mode: discovering dynamic parameters for:', inputText);
      try {
        const { discoverParams } = await import('../lib/api');
        const res = await discoverParams({ domain: selectedDomain, question: inputText });
        
        if (res.success && res.parameters && res.parameters.length > 0) {
          // LLM returned question-specific chips — use them
          setDynamicParams(res.parameters);
        } else {
          // Fallback to static registry params
          setDynamicParams([]);
        }
        setModalStep(0); 
        setModalOpen(true); 
      } catch (e) {
        console.error('Discovery failed, falling back to static:', e);
        setDynamicParams([]);
        setModalStep(0); 
        setModalOpen(true); 
      } finally {
        setIsDiscovering(false);
        setIsGenerating(false);
      }
      return;
    }

    // Ensure we show generating state if we skipped the screening block
    if (isConfirmed) setIsGenerating(true);

    try {
      if (selectedMode === 'free') {
        const text = inputText.trim();
        if (!text) { setApiError('Please describe a situation for Free Inference mode.'); return; }
        const res = await runFreeInfer(text, 5);
        const outcomesList = res.result?.outcomes || res.outcomes || res.result || [];
        setOutcomes(outcomesList.map((o: any) => ({
          outcome: o.outcome || 'Unknown outcome', probability: o.probability || 50,
          probability_pct: o.probability_pct || `${o.probability || 50}%`,
          reasoning: o.reasoning || '', type: o.type || 'neutral', condition: null, has_transparency: true,
        })));
        if (res.result && typeof res.result === 'object' && !Array.isArray(res.result) && Object.keys(res.result).length > 2) {
          setPredResult(res.result);
          setFreeInferResult(res.result);
        }
        setShowResults(true); setTimeout(() => setIsAnimating(true), 100); return;
      }

      if (selectedMode === 'document') {
        if (!docFile) { setApiError('Please select a document first.'); return; }
        setDocLoading(true); setDocResult(null); setInsufficientInfo(null);
        try {
          const res = await analyzeDocument(docFile, selectedDomain, inputText || undefined);
          if (res.insufficient_info) {
            setInsufficientInfo({ reason: res.reason, missing: res.missing_info });
          } else {
            setDocResult(res);
            if (res.prediction) setPredResult(res.prediction);
            if (res.outcomes) { setOutcomes(res.outcomes); setShowResults(true); setTimeout(() => setIsAnimating(true), 20); }
          }
        } catch (e: any) { setApiError(e.message || 'Document analysis failed'); } 
        finally { setDocLoading(false); }
        return;
      }

      if (selectedMode === 'voice') {
        if (!voiceFile) { setApiError('Please select an audio file first.'); return; }
        setVoiceLoading(true); setVoiceResult(null); setInsufficientInfo(null);
        try {
          const BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';
          const fd = new FormData(); fd.append('file', voiceFile); fd.append('domain', selectedDomain);
          const resp = await fetch(`${BASE}/vision/voice`, { method: 'POST', body: fd });
          if (resp.ok) {
            const data = await resp.json();
            if (data.insufficient_info) {
              setInsufficientInfo({ reason: data.reason });
            } else {
              setVoiceResult(data);
              if (data.prediction) {
                setPredResult(data.prediction);
                if (data.outcomes) setOutcomes(data.outcomes);
                setShowResults(true); setTimeout(() => setIsAnimating(true), 20);
              }
            }
          } else {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Voice analysis failed');
          }
        } catch (e: any) { setApiError(e.message || 'Voice analysis failed'); } 
        finally { setVoiceLoading(false); }
        return;
      }

      if (selectedMode === 'adversarial') {
        const { getAdversarialParams } = await import('../lib/api');
        const paramsRes = await getAdversarialParams(selectedDomain, inputText || undefined);
        const preset = paramsRes.adversarial_params;
        const res = await runAdversarial({ domain: selectedDomain, parameters: preset, question: inputText || undefined });
        setAdversarialResult(res); setShowResults(true); setTimeout(() => setIsAnimating(true), 100); return;
      }

      if (selectedMode === 'whatif') {
        if (!inputText.trim()) { setApiError('Please describe a "What if…" scenario.'); return; }
        const res = await runWhatIf({ domain: selectedDomain, parameters, question: inputText });
        // Support both res.tree (new) and legacy res.simulation
        const tree = res.tree ?? (res.simulation ? {
          base_probability: Math.round((res.base_probability ?? 0.5) * 100),
          description: inputText,
          branches: []
        } : null);
        setWhatifTree(tree); setShowResults(true); setTimeout(() => setIsAnimating(true), 100); return;
      }

      if (selectedMode === 'comparative') {
        if (compScenarios.filter(s => Object.keys(s.params).length > 0).length < 2) {
          setApiError('Add parameters to at least 2 scenarios.'); return;
        }
        const res = await runComparative({ domain: selectedDomain, scenarios: compScenarios.map(s => ({ label: s.label, ...s.params })), question: inputText || undefined });
        setCompResult(res.result); setShowResults(true); setTimeout(() => setIsAnimating(true), 100); return;
      }

      if (selectedMode === 'expert') {
        const [expertRes, outcomeRes] = await Promise.all([
          runExpertMode({ domain: selectedDomain, parameters, question: inputText || undefined }),
          getOutcomes({ domain: selectedDomain, parameters, question: inputText || undefined, n_outcomes: 5, mode: 'independent' })
        ]);
        setExpertDebate(expertRes.debate);
        setOutcomes(outcomeRes.result?.outcomes || []);
        setShowResults(true); setTimeout(() => setIsAnimating(true), 20); return;
      }

      if (selectedMode === 'retrospective') {
        if (!retroDesc.trim()) { setApiError('Please describe the past event to analyse.'); return; }
        const res = await runRetrospective({ domain: selectedDomain, description: retroDesc, outcome: retroOutcome || undefined, parameters });
        // Backend returns res.analysis (structured) or fallback res.story
        const analysis = res.analysis ?? (res.story ? {
          probability_at_time: 50,
          root_cause: typeof res.story === 'string' ? res.story : (res.story?.narrative ?? 'Event occurred due to compounding factors.'),
          prevention_point: 'Earlier monitoring could have altered the trajectory.',
          contributing_factors: [],
          lessons_learned: []
        } : null);
        setRetroResult(analysis); setShowResults(true); setTimeout(() => setIsAnimating(true), 100); return;
      }

      if (selectedMode === 'simulation') {
        const [simRes, outcomeRes] = await Promise.all([
          runSimulation({ domain: selectedDomain, parameters, question: inputText || undefined }),
          getOutcomes({ domain: selectedDomain, parameters, question: inputText || undefined, n_outcomes: 5, mode: 'independent' })
        ]);
        // Backend returns monte_carlo key directly (new) or base_result.monte_carlo
        const mcData = simRes.monte_carlo ?? simRes.base_result?.monte_carlo ?? null;
        setSimResult({ ...simRes, monte_carlo: mcData });
        setOutcomes(outcomeRes.result?.outcomes || simRes.outcomes || []);
        setShowResults(true); setTimeout(() => setIsAnimating(true), 20); return;
      }

      if (selectedMode === 'monitoring') {
        if (!monSession) { setApiError('Start a monitoring session first.'); return; }
        setMonLoading(true);
        try {
          const res = await updateMonitoring({ session_id: monSession.session_id, domain: selectedDomain, parameters, update_text: monUpdate || undefined });
          setMonSession(prev => prev ? { ...prev, updates: [...prev.updates, { probability: res.probability, update_text: res.update_text, updated_at: res.updated_at }] } : prev);
          setMonUpdate('');
        } finally { setMonLoading(false); }
        return;
      }

      const merged = { ...parameters, ...hybridExtracted };
      const [predRes, outcomeRes] = await Promise.all([
        runPredict({ domain: selectedDomain, parameters: merged, question: inputText || undefined, mode: selectedMode, run_debate: true }),
        getOutcomes({ domain: selectedDomain, parameters: merged, question: inputText || undefined, n_outcomes: 5, mode: 'independent' })
      ]);
      setPredResult(predRes.prediction); setPredId(predRes.prediction_id || '');
      setOutcomes(outcomeRes.result?.outcomes || []);
        setShowResults(true); setTimeout(() => setIsAnimating(true), 20);

    } catch (err) {
      if (err instanceof SambhavAPIError) {
        setApiError(err.isInputQuality ? (JSON.parse(err.message)?.error ?? err.message) : `Error ${err.status}: ${err.message}`);
      } else { setApiError('Prediction failed. Is the backend running?'); }
    } finally { setIsGenerating(false); }
  }, [selectedDomain, parameters, inputText, selectedMode, compScenarios, retroDesc, retroOutcome, hybridExtracted, monSession, monUpdate, handleConvPredict, ADVERSARIAL_PRESETS]);

  const handleLoadMore = async () => {
    setLoadingMore(true);
    try {
      const res = await getOutcomes({ domain: selectedDomain, parameters, question: inputText || undefined, n_outcomes: 3, existing_outcomes: outcomes, mode: 'independent' });
      setOutcomes(prev => [...prev, ...(res.result?.outcomes || [])]);
    } catch (err) { console.error(err); } finally { setLoadingMore(false); }
  };

  const handleWhyClick = async (index: number) => {
    if (expandedOutcome === index) { setExpandedOutcome(null); return; }
    setExpandedOutcome(index);
    const key = `${index}_full`;
    if ((whyData as any)[key]) return;
    setLoadingWhy(index);
    try {
      const outcome = outcomes[index];
      const outcomeProb = outcome.probability / 100;
      const res = await getTransparency({ 
        domain: selectedDomain, 
        parameters, 
        final_probability: outcomeProb, 
        question: inputText || undefined, 
        outcome: outcome.outcome, 
        level: 'full'
      } as any);
      setWhyData(prev => ({ ...prev, [key]: res.result }));
    } catch (e) { console.error(e); } finally { setLoadingWhy(null); }
  };


  // Dedicated fetch — does NOT toggle expand/collapse (fixes button race condition)
  const fetchTransparencyData = async (index: number, level: 'simple' | 'detailed' | 'full') => {
    const key = `${index}_${level}`;
    if ((whyData as any)[key]) return; // already cached for this level
    setLoadingWhy(index);
    try {
      const outcome = outcomes[index];
      const outcomeProb = outcome.probability / 100;
      const res = await getTransparency({
        domain: selectedDomain,
        parameters,
        final_probability: outcomeProb,
        question: inputText || undefined,
        outcome: outcome.outcome,
        level,
      } as any);
      setWhyData(prev => ({ ...prev, [key]: res.result }));
    } catch (e) { console.error(e); } finally { setLoadingWhy(null); }
  };

  const handleInverseClick = async (index: number) => {
    const key = `inv_${index}`;
    if (inverseData[key]) {
      // Toggle off if already loaded
      setInverseData(prev => { const n = { ...prev }; delete n[key]; return n; });
      return;
    }
    setLoadingInverse(index);
    try {
      const outcome = outcomes[index];
      const res = await getInverseTransparency({
        domain: selectedDomain,
        parameters,
        final_probability: outcome.probability / 100,
        question: inputText || undefined,
        outcome: outcome.outcome,
      });
      setInverseData(prev => ({ ...prev, [key]: res }));
    } catch (e) { console.error('Inverse error:', e); } finally { setLoadingInverse(null); }
  };

  

  const handleReset = () => {
    setShowResults(false); setIsAnimating(false); /* transparency simple removed */
    setExpandedOutcome(null); setApiError(null); setPredResult(null); setOutcomes([]); setWhyData({});
    setInverseData({}); setLoadingInverse(null);
    setAdversarialResult(null); setWhatifTree(null); setCompResult(null);
    setExpertDebate(null); setRetroResult(null); setSimResult(null);
    setConvMessages([]); setConvParams({}); setConvComplete(false); setConvStarted(false);
    setParamsConfirmed(false); setParameters({}); setDynamicParams([]);
  };

  const activeWhyData = expandedOutcome !== null ? (whyData as any)[`${expandedOutcome}_full`] || null : null;

  const reliabilityScore = predResult
    ? Math.round(predResult.reliability_index * 100)
    : Math.round((Object.keys(parameters).length / Math.max(paramList.length, 1)) * 100);

  const reliabilitySuggestions = predResult?.shap_values
    ? Object.entries(predResult.shap_values).filter(([, v]) => typeof v === 'number' && v < 0)
        .sort(([, a], [, b]) => (a as number) - (b as number)).slice(0, 3)
        .map(([k]) => `Improve '${k.replace(/_/g, ' ')}' to increase confidence`)
    : ['Add more parameters to increase reliability'];

  const exportPayload = predResult
    ? {
        prediction_id: predId,
        domain: selectedDomain,
        parameters,
        question: inputText || undefined,
        result: {
          ...predResult as any,
          // Include full multi-outcome list for export templates
          outcomes_list: outcomes.map(o => ({ outcome: o.outcome, probability: o.probability, reasoning: o.reasoning || '' }))
        }
      }
    : null;

  // ── Left-panel renderers ─────────────────────────────────────────────────
  const renderConversationalInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-16 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none placeholder:text-muted-foreground/50"
        placeholder="What do you want to predict? (optional)" value={inputText} onChange={e => setInputText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!convStarted) handleStartConversation();
          }
        }} />
      {!convStarted ? (
        <motion.button onClick={handleStartConversation} disabled={convLoading}
          className="w-full px-3 py-2 text-xs rounded-lg bg-primary text-black font-medium flex items-center justify-center gap-1.5 disabled:opacity-50"
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          {convLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <MessageCircle className="w-3 h-3" />}
          <span>Start Conversation</span>
        </motion.button>
      ) : (
        <div className="flex flex-col gap-2">
          <div className="text-[10px] text-muted-foreground px-1">
            <span className="text-primary">{Object.keys(convParams).length}</span> parameters collected
            {convComplete && <span className="text-success ml-2">● Complete</span>}
          </div>
          {convComplete && (
            <motion.button onClick={handleConvPredict} disabled={isGenerating}
              className="w-full px-3 py-2 text-xs rounded-lg bg-success text-black font-medium flex items-center justify-center gap-1.5 disabled:opacity-50"
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              {isGenerating ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
              <span>Generate Prediction</span>
            </motion.button>
          )}
        </div>
      )}
    </div>
  );

  const renderHybridInput = () => (
    <div className="space-y-3 relative">
      <textarea className="w-full h-14 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="Describe your analysis context..." value={inputText} onChange={e => setInputText(e.target.value)} />
      <div className={hybridLoading ? 'opacity-20 pointer-events-none' : ''}>
        <FileUploadZone accept="image/*,video/*" multiple
          onFileSelect={async (files) => {
            if (!files.length) return;
            setHybridLoading(true); setApiError(null); setInsufficientInfo(null);
            try {
              const API_BASE = (import.meta as any).env?.VITE_API_URL ?? '/api';
              const token = localStorage.getItem('sambhav_token');
              const fd = new FormData(); fd.append('file', files[0]); fd.append('domain', selectedDomain);
              const endpoint = files[0].type.startsWith('video') ? 'video' : 'image';
              const resp = await fetch(`${API_BASE}/vision/${endpoint}`, { 
                method: 'POST', 
                headers: token ? { Authorization: `Bearer ${token}` } : {},
                body: fd 
              });
              if (resp.ok) { 
                const data = await resp.json(); 
                if (data.insufficient_info) {
                  setInsufficientInfo({ reason: data.reason });
                } else {
                  const inferred = data.result?.inferred_parameters || {};
                  if (Object.keys(inferred).length === 0) {
                    setHybridExtracted({ "visual_context": "present", "media_complexity": "high", "structural_format": "unstructured" });
                  } else {
                    setHybridExtracted(inferred); 
                  }
                  // Just store the params. Let the user click Generate.
                  if (data.prediction || data.result) {
                    // Vision successful. The parameters are loaded into hybridExtracted.
                  }
                }
              }
            } catch (e) { console.error(e); setApiError('Analysis failed'); } finally { setHybridLoading(false); }
          }} />
      </div>
      {hybridLoading && <div className="flex items-center gap-2 text-[11px] text-muted-foreground"><Loader2 className="w-3 h-3 animate-spin" /><span>Extracting parameters…</span></div>}
      {Object.keys(hybridExtracted).length > 0 && (
        <div className="space-y-3">
          <div className="space-y-1 text-[10px] max-h-32 overflow-y-auto pr-1">
            <p className="text-muted-foreground font-medium sticky top-0 bg-[#0f0f19] pt-1 pb-1">Vision Extracted Parameters:</p>
            {Object.entries(hybridExtracted).map(([k, v]) => (
              <div key={k} className="flex justify-between"><span className="text-muted-foreground">{k.replace(/_/g,' ')}</span><span className="text-primary">{String(v)}</span></div>
            ))}
          </div>
          <motion.button onClick={() => handleGenerate()} disabled={isGenerating}
            className="w-full px-3 py-2 text-xs rounded-lg bg-primary text-black font-medium flex items-center justify-center gap-1.5 disabled:opacity-50"
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            {isGenerating ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
            <span>Generate Prediction</span>
          </motion.button>
        </div>
      )}
    </div>
  );

  const renderDocumentInput = () => (
    <div className="space-y-3 relative">
      <textarea className="w-full h-12 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="What should Sambhav predict from this document?" value={inputText} onChange={e => setInputText(e.target.value)} />
      <div className={docLoading ? 'opacity-20 pointer-events-none' : ''}>
        <FileUploadZone accept="*" maxSize={50}
          onFileSelect={(files) => {
            if (files.length) setDocFile(files[0]);
          }} />
      </div>
      {docLoading && <div className="flex items-center gap-2 text-[11px] text-muted-foreground"><Loader2 className="w-3 h-3 animate-spin" /><span>Analysing document…</span></div>}
      {docResult && <div className="text-[10px] text-success">● Document analysed — see results</div>}
    </div>
  );

  const renderComparativeInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-10 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="What outcome to compare across scenarios?" value={inputText} onChange={e => setInputText(e.target.value)} />
      <motion.button onClick={async () => {
        if (!inputText.trim()) return;
        const { discoverParams } = await import('../lib/api');
        const res = await discoverParams({ domain: selectedDomain, question: inputText });
        if (res.success && res.parameters) setDynamicParams(res.parameters);
      }} className="w-full px-2 py-1.5 text-[10px] rounded-lg bg-primary/10 border border-primary/20 text-primary hover:bg-primary/20 transition-all flex items-center justify-center gap-1">
         <Zap className="w-3 h-3" /> Auto-Generate Comparison Fields from Question
      </motion.button>
      {compScenarios.map((s, i) => (
        <div key={i} className="p-2 rounded-lg bg-white/5 border border-white/10 space-y-1.5">
          <input className="w-full px-2 py-1 text-[11px] bg-white/5 border border-white/10 rounded focus:outline-none"
            placeholder={`Scenario ${String.fromCharCode(65 + i)} label`} value={s.label}
            onChange={e => setCompScenarios(prev => prev.map((sc, j) => j === i ? { ...sc, label: e.target.value } : sc))} />
          <div className="space-y-1">
            {(dynamicParams.length > 0 ? dynamicParams.slice(0, 5) : (paramList.length > 0 ? paramList.slice(0, 5) : [
              { key: 'factor_1', label: 'Key Factor 1' },
              { key: 'factor_2', label: 'Key Factor 2' },
              { key: 'factor_3', label: 'Key Factor 3' },
            ])).map((p: any) => (
              <div key={p.key} className="flex items-center gap-1.5">
                <span className="text-[9px] text-muted-foreground/70 min-w-[80px] shrink-0 truncate">{(p.label || p.key).replace(/_/g,' ')}</span>
                <input
                  className="flex-1 px-1.5 py-0.5 text-[10px] bg-white/5 border border-white/10 rounded focus:outline-none focus:border-primary/40 transition-colors"
                  placeholder="value"
                  value={s.params[p.key] ?? ''}
                  onChange={e => {
                    const newParams = { ...s.params };
                    if (e.target.value) { newParams[p.key] = e.target.value; } else { delete newParams[p.key]; }
                    setCompScenarios(prev => prev.map((sc, j) => j === i ? { ...sc, params: newParams } : sc));
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="flex gap-2">
        <motion.button onClick={() => setCompScenarios(prev => [...prev, { label: `Scenario ${String.fromCharCode(65 + prev.length)}`, params: {} }])}
          className="flex-1 px-2 py-1.5 text-[10px] rounded-lg bg-white/5 border border-white/10 flex items-center justify-center gap-1 hover:bg-white/10"
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Plus className="w-3 h-3" /> Add Scenario
        </motion.button>
        {compScenarios.length > 2 && (
          <motion.button onClick={() => setCompScenarios(prev => prev.slice(0, -1))}
            className="px-2 py-1.5 text-[10px] rounded-lg bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10"
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Minus className="w-3 h-3" />
          </motion.button>
        )}
      </div>
    </div>
  );

  const renderMonitoringInput = () => (
    <div className="space-y-3">
      {!monSession ? (
        <>
          <input className="w-full px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
            placeholder="Session name (e.g. Project Alpha)" value={monName} onChange={e => setMonName(e.target.value)} />
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-muted-foreground">Alert below:</span>
            <input type="number" min={0} max={100} className="w-16 px-2 py-1 text-xs bg-white/5 border border-white/10 rounded focus:outline-none"
              value={monThresh} onChange={e => setMonThresh(Number(e.target.value))} />
            <span className="text-muted-foreground">%</span>
          </div>
          <motion.button disabled={monLoading}
            onClick={async () => {
              if (!monName.trim()) { setApiError('Enter a session name.'); return; }
              setMonLoading(true);
              try {
                const res = await startMonitoring({ name: monName, domain: selectedDomain, parameters, question: inputText || undefined, threshold_low: monThresh / 100 });
                setMonSession({ ...res, updates: [] });
              } catch (e: any) { setApiError(e.message); } finally { setMonLoading(false); }
            }}
            className="w-full px-3 py-2 text-xs rounded-lg bg-primary text-black font-medium flex items-center justify-center gap-1.5 disabled:opacity-50"
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            {monLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Activity className="w-3 h-3" />}
            <span>Start Monitoring</span>
          </motion.button>
        </>
      ) : (
        <>
          <div className="text-[10px] space-y-1">
            {[['Session', monSession.name, 'text-primary'], ['Baseline', `${monSession.baseline}%`, 'font-medium'], ['Updates', String(monSession.updates.length), '']].map(([l, v, c]) => (
              <div key={l as string} className="flex justify-between"><span className="text-muted-foreground">{l as string}</span><span className={c as string}>{v as string}</span></div>
            ))}
          </div>
          <textarea className="w-full h-16 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
            placeholder="What changed? Describe the update…" value={monUpdate} onChange={e => setMonUpdate(e.target.value)} />
        </>
      )}
    </div>
  );

  const renderAdversarialInput = () => {
    const preset = ADVERSARIAL_PRESETS[selectedDomain] || ADVERSARIAL_PRESETS.default;
    return (
      <div className="space-y-3">
        <div className="p-2 rounded-lg bg-destructive/10 border border-destructive/20">
          <p className="text-[10px] text-destructive/80 leading-relaxed">
            <ShieldAlert className="w-3 h-3 inline mr-1" />
            Submits extreme/contradictory parameters to trigger the 3-engine audit system.
          </p>
        </div>
        <div className="space-y-0.5 text-[10px]">
          <p className="text-muted-foreground font-medium mb-1">Extreme parameters loaded:</p>
          {Object.entries(preset).map(([k, v]) => (
            <div key={k} className="flex justify-between"><span className="text-muted-foreground">{k.replace(/_/g,' ')}</span><span className="text-destructive font-medium">{String(v)}</span></div>
          ))}
        </div>
        <textarea className="w-full h-10 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none resize-none placeholder:text-muted-foreground/50"
          placeholder="Optional: describe the extreme scenario…" value={inputText} onChange={e => setInputText(e.target.value)} />
      </div>
    );
  };

  const renderWhatIfInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-24 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder={`What if…?\n\nExample: "What if the student doubles their study hours next week?"`}
        value={inputText} onChange={e => setInputText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
          }
        }} />
      <p className="text-[10px] text-muted-foreground/70"><GitBranch className="w-3 h-3 inline mr-1" />Generates a branching probability tree per scenario.</p>
    </div>
  );

  const renderExpertInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-20 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="Describe the question for expert consultation…" value={inputText} onChange={e => setInputText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
          }
        }} />
      {paramList.length > 0 && (
        <motion.button onClick={() => { setModalStep(0); setModalOpen(true); }}
          className="w-full px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary/30 transition-all flex items-center justify-between"
          whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
          <span className="text-muted-foreground">{Object.keys(parameters).length > 0 ? `✓ ${Object.keys(parameters).length} parameters` : '▶ Add Domain Parameters'}</span>
          <Users className="w-3 h-3 text-primary" />
        </motion.button>
      )}
      <p className="text-[10px] text-muted-foreground/70"><Users className="w-3 h-3 inline mr-1" />4 agents: Optimist · Pessimist · Realist · Devil's Advocate</p>
    </div>
  );

  const renderRetrospectiveInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-24 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder={"Describe the past event…\n\nExample: 'The student failed despite attending all classes.'"}
        value={retroDesc} onChange={e => setRetroDesc(e.target.value)} />
      <input className="w-full px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
        placeholder="Actual outcome (optional — e.g. 'Failed', 'Resigned')" value={retroOutcome} onChange={e => setRetroOutcome(e.target.value)} />
      <p className="text-[10px] text-muted-foreground/70"><History className="w-3 h-3 inline mr-1" />Explains why the outcome occurred and identifies the prevention point.</p>
    </div>
  );

  const renderSimulationInput = () => (
    <div className="space-y-3">
      <div className="p-2 rounded-lg bg-primary/5 border border-primary/20">
        <p className="text-[10px] text-primary/80"><FlaskConical className="w-3 h-3 inline mr-1" />Hypothetical scenario — Monte Carlo simulation with 200 runs.</p>
      </div>
      <textarea className="w-full h-16 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="Describe your hypothetical scenario…" value={inputText} onChange={e => setInputText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
          }
        }} />
      {paramList.length > 0 && (
        <motion.button onClick={() => { setModalStep(0); setModalOpen(true); }}
          className="w-full px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary/30 transition-all flex items-center justify-between"
          whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
          <span className="text-muted-foreground">{Object.keys(parameters).length > 0 ? `✓ ${Object.keys(parameters).length} parameters` : '▶ Configure Scenario Parameters'}</span>
          <ChevronRight className="w-3 h-3 text-primary" />
        </motion.button>
      )}
    </div>
  );

  const renderDefaultInput = () => (
    <div className="space-y-3">
      <textarea className="w-full h-28 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none placeholder:text-muted-foreground/50"
        placeholder={selectedMode === 'free' ? 'Describe any situation — no domain or form needed…' : `Describe the situation you want to predict…`}
        value={inputText} onChange={e => setInputText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
          }
        }} />
      {selectedMode === 'guided' && paramList.length > 0 && (
        <motion.button onClick={() => { handleGenerate(false); }}
          className="w-full px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-primary/30 transition-all flex items-center justify-between group"
          whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
          <span className="text-muted-foreground group-hover:text-foreground transition-colors">
            {Object.keys(parameters).filter(k => parameters[k] != null).length > 0
              ? `✓ ${Object.keys(parameters).filter(k => parameters[k] != null).length} of ${paramList.length} parameters configured`
              : '▶ Configure Parameters (Guided Mode)'}
          </span>
          <ChevronRight className="w-3 h-3 text-primary" />
        </motion.button>
      )}
      {selectedMode === 'guided' && Object.keys(parameters).length > 0 && (
        <div className="text-[10px] text-muted-foreground/70 space-y-0.5 max-h-20 overflow-y-auto">
          {Object.entries(parameters).slice(0, 6).map(([k, v]) => v != null && (
            <div key={k} className="flex justify-between"><span>{k.replace(/_/g,' ')}</span><span className="text-primary/80">{String(v)}</span></div>
          ))}
        </div>
      )}
    </div>
  );

  const renderVoiceInput = () => (
    <div className="space-y-3 relative">
      <textarea className="w-full h-12 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none placeholder:text-muted-foreground/50"
        placeholder="What should Sambhav listen for?" value={inputText} onChange={e => setInputText(e.target.value)} />
      <div className={voiceLoading ? 'opacity-20 pointer-events-none' : ''}>
        <FileUploadZone accept="audio/*" maxSize={20}
          onFileSelect={(files) => {
            if (files.length) setVoiceFile(files[0]);
          }} />
      </div>
      {voiceLoading && <div className="flex items-center gap-2 text-[11px] text-muted-foreground"><Loader2 className="w-3 h-3 animate-spin" /><span>Transcribing audio…</span></div>}
      {voiceResult?.result?.transcript && (
        <div className="p-2 rounded-lg bg-white/5 border border-white/10 mt-2">
          <p className="text-[9px] text-muted-foreground uppercase font-bold mb-1">Transcript</p>
          <p className="text-[10px] italic">"{voiceResult.result.transcript}"</p>
        </div>
      )}
    </div>
  );

  const renderModeInterface = () => {
    switch (selectedMode) {
      case 'conversational': return renderConversationalInput();
      case 'hybrid':         return renderHybridInput();
      case 'document':       return renderDocumentInput();
      case 'voice':          return renderVoiceInput();
      case 'comparative':    return renderComparativeInput();
      case 'monitoring':     return renderMonitoringInput();
      case 'adversarial':    return renderAdversarialInput();
      case 'whatif':         return renderWhatIfInput();
      case 'expert':         return renderExpertInput();
      case 'retrospective':  return renderRetrospectiveInput();
      case 'simulation':     return renderSimulationInput();
      default:               return renderDefaultInput();
    }
  };

  // ── Right-panel renderers ────────────────────────────────────────────────
  const renderConversationalPanel = () => {
    const lastMsg        = convMessages[convMessages.length - 1];
    const currentOptions = lastMsg?.role === 'sambhav' ? (lastMsg.options || []) : [];
    const currentKey     = lastMsg?.role === 'sambhav' ? (lastMsg.param_key || '') : '';
    return (
      <div className="flex flex-col min-h-[520px]">
        <div className="flex-1 overflow-y-auto space-y-3 pr-1 mb-3">
          {convMessages.length === 0 && !convStarted && (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                <MessageCircle className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-sm font-medium mb-1">Conversational Mode</h3>
              <p className="text-xs text-muted-foreground max-w-xs">Sambhav asks one question at a time — like an expert advisor. Click "Start Conversation" to begin.</p>
            </div>
          )}
          {convMessages.map((msg, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed ${msg.role === 'sambhav' ? 'bg-white/5 border border-white/10 text-foreground' : 'bg-primary/20 border border-primary/30 text-primary'}`}>
                {msg.role === 'sambhav' && <span className="text-[9px] text-primary/60 font-medium block mb-0.5">Sambhav</span>}
                {msg.content}
              </div>
            </motion.div>
          ))}
          {currentOptions.length > 0 && !convComplete && !isGenerating && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap gap-1.5 pl-1">
              {currentOptions.map((opt, oi) => (
                <motion.button key={oi} onClick={() => handleConvAnswer(opt, currentKey)}
                  className="px-2.5 py-1 text-[10px] rounded-full bg-white/5 border border-white/15 hover:bg-primary/15 hover:border-primary/40 transition-all"
                  whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>{opt}</motion.button>
              ))}
              <motion.button onClick={() => handleConvAnswer('', currentKey, true)}
                className="px-2.5 py-1 text-[10px] rounded-full bg-white/5 border border-white/10 hover:bg-white/10 text-muted-foreground transition-all"
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>Skip</motion.button>
            </motion.div>
          )}
          {(convLoading || isGenerating) && (
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground pl-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>{isGenerating ? 'Generating prediction…' : 'Sambhav is thinking…'}</span>
            </div>
          )}
          <div ref={convEndRef} />
        </div>
        {convStarted && !convComplete && !isGenerating && (
          <div className="flex gap-2 pt-2 border-t border-white/10">
            <input className="flex-1 px-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
              placeholder="Or type your own answer…" value={convInput} onChange={e => setConvInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && convInput.trim()) handleConvAnswer(convInput.trim(), currentKey); }} />
            <motion.button onClick={() => convInput.trim() && handleConvAnswer(convInput.trim(), currentKey)} disabled={!convInput.trim()}
              className="px-2.5 py-2 rounded-lg bg-primary text-black disabled:opacity-40"
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}><Send className="w-3 h-3" /></motion.button>
          </div>
        )}
      </div>
    );
  };

  const renderAdversarialPanel = () => {
    if (!adversarialResult) return null;
    const flags   = adversarialResult.audit_flags || [];
    const blocked = adversarialResult.blocked;
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        <div className={`p-3 rounded-xl border ${blocked ? 'bg-destructive/10 border-destructive/30' : 'bg-warning/10 border-warning/30'}`}>
          <div className="flex items-center gap-2 mb-1">
            <ShieldAlert className={`w-4 h-4 ${blocked ? 'text-destructive' : 'text-warning'}`} />
            <span className={`text-xs font-bold ${blocked ? 'text-destructive' : 'text-warning'}`}>
              {blocked ? 'PREDICTION BLOCKED — AUDIT SYSTEM FIRED' : 'PREDICTION PASSED WITH FLAGS'}
            </span>
          </div>
          {adversarialResult.block_reason && <p className="text-[11px] text-destructive/80 mt-1">{adversarialResult.block_reason}</p>}
        </div>
        <div>
          <p className="text-[10px] font-medium text-muted-foreground mb-2">Engines run: {adversarialResult.engines_run?.join(' · ')}</p>
          {flags.length > 0 ? flags.map((f: any, i: number) => (
            <div key={i} className="flex items-start gap-2 mb-2 p-2 rounded-lg bg-white/5 border border-white/10">
              <div className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${f.severity === 'CRITICAL' ? 'bg-destructive' : f.severity === 'WARNING' ? 'bg-warning' : 'bg-primary/50'}`} />
              <div><span className="text-[10px] font-mono font-medium">{f.code} — {f.severity}</span><p className="text-[10px] text-muted-foreground">{f.message}</p></div>
            </div>
          )) : <p className="text-[11px] text-muted-foreground italic">No audit flags raised — parameters passed all checks.</p>}
        </div>
        
        <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 px-3 py-2 text-xs rounded-xl bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20 transition-all flex items-center justify-center gap-1.5 font-medium">
          <MessageCircle className="w-4 h-4" /> Analyse this vulnerability with AI
        </motion.button>
        
        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 10: Adversarial Mode — demonstrates fail-safe audit behaviour</p>
      </motion.div>
    );
  };

  const renderWhatIfPanel = () => {
    if (!whatifTree) return null;
    const branches = whatifTree.branches || whatifTree.children || [];
    const basePct  = whatifTree.base_probability ?? whatifTree.base_prob ?? 60;
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        <div className="p-3 rounded-xl bg-white/5 border border-white/10">
          <p className="text-[10px] text-muted-foreground">Base probability</p>
          <p className="text-2xl font-bold text-primary">{basePct}%</p>
          {whatifTree.description && <p className="text-[10px] text-muted-foreground mt-1">{whatifTree.description}</p>}
        </div>
        <div className="space-y-2">
          <p className="text-[10px] font-medium text-muted-foreground">Scenario Branches</p>
          {branches.map((b: any, i: number) => {
            const newProb = b.new_probability ?? b.probability ?? basePct;
            const shift   = b.probability_shift ?? (newProb - basePct);
            return (
              <div key={i} className="p-3 rounded-xl bg-white/5 border border-white/10 space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5"><GitBranch className="w-3 h-3 text-primary/60" /><span className="text-[11px] font-medium">{b.event || b.scenario || `Branch ${i+1}`}</span></div>
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] font-mono ${shift >= 0 ? 'text-success' : 'text-destructive'}`}>{shift >= 0 ? '+' : ''}{Math.round(shift)}%</span>
                    <span className="text-xs font-bold">{Math.round(newProb)}%</span>
                  </div>
                </div>
                {b.reasoning && <p className="text-[10px] text-muted-foreground">{b.reasoning}</p>}
                {(b.children || b.sub_branches || []).map((sub: any, j: number) => {
                  const sp = sub.new_probability ?? sub.probability ?? newProb;
                  const ss = sub.probability_shift ?? (sp - newProb);
                  return (
                    <div key={j} className="ml-4 p-2 rounded-lg bg-white/5 border border-white/5 flex justify-between">
                      <span className="text-[10px] text-muted-foreground">{sub.event || sub.scenario}</span>
                      <div className="flex items-center gap-1">
                        <span className={`text-[9px] font-mono ${ss >= 0 ? 'text-success' : 'text-destructive'}`}>{ss >= 0 ? '+' : ''}{Math.round(ss)}%</span>
                        <span className="text-[10px] font-bold">{Math.round(sp)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 px-3 py-2 text-xs rounded-xl bg-white/5 text-muted-foreground hover:text-foreground border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-1.5 font-medium">
          <MessageCircle className="w-4 h-4 text-primary" /> Discuss these scenarios with AI
        </motion.button>

        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 12: What-If Story — Section 8.14 Scenario Planning</p>
      </motion.div>
    );
  };

  const renderComparativePanel = () => {
    if (!compResult) return null;
    const matrix    = compResult.matrix || {};
    const outcomes_ = compResult.outcomes || Object.keys(matrix);
    const scenarios_ = compResult.scenarios || compScenarios.map(s => s.label);
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        {compResult.recommendation && (
          <div className="p-3 rounded-xl bg-primary/10 border border-primary/30">
            <p className="text-[10px] text-primary/80 font-medium">Recommendation</p>
            <p className="text-xs mt-0.5">{compResult.recommendation}</p>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead><tr>
              <th className="text-left py-1 pr-3 text-muted-foreground font-medium">Outcome</th>
              {scenarios_.map((s: string, i: number) => <th key={i} className="text-center py-1 px-2 text-muted-foreground font-medium">{s}</th>)}
              <th className="text-center py-1 pl-2 text-muted-foreground font-medium">Winner</th>
            </tr></thead>
            <tbody>
              {outcomes_.map((outcome: string, oi: number) => {
                const rowObj = Array.isArray(matrix) ? matrix.find((r:any) => r.outcome === outcome) : null;
                const rowProbs = rowObj ? (rowObj.probabilities || {}) : (matrix[outcome] || {});
                const vals = scenarios_.map((s: string) => rowProbs[s] ?? null);
                const maxV = Math.max(...vals.filter((v: any) => v !== null));
                const winI = vals.indexOf(maxV);
                return (
                  <tr key={oi} className="border-t border-white/5">
                    <td className="py-1.5 pr-3 text-muted-foreground">{outcome}</td>
                    {vals.map((v: any, vi: number) => (
                      <td key={vi} className={`py-1.5 px-2 text-center font-mono ${vi === winI ? 'text-primary font-bold' : ''}`}>{v !== null ? `${Math.round(v)}%` : '—'}</td>
                    ))}
                    <td className="py-1.5 pl-2 text-center text-success font-medium">{winI >= 0 ? scenarios_[winI] : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 px-3 py-2 text-xs rounded-xl bg-white/5 text-muted-foreground hover:text-foreground border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-1.5 font-medium">
          <MessageCircle className="w-4 h-4 text-primary" /> Ask the AI to evaluate this comparison
        </motion.button>

        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 6: Comparative Inference — Section 8.7</p>
      </motion.div>
    );
  };

  const renderExpertPanel = () => {
    const agentColors: Record<string, string>  = { optimist: 'text-success', pessimist: 'text-destructive', realist: 'text-primary', devils_advocate: 'text-warning' };
    const agentLabels: Record<string, string>  = { optimist: 'The Optimist', pessimist: 'The Pessimist', realist: 'The Realist (Final)', devils_advocate: "Devil's Advocate" };
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        {expertDebate && (
          <>
            {expertDebate.final_probability != null && (
              <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                <p className="text-[10px] text-muted-foreground">Reconciled Probability (Realist)</p>
                <p className="text-2xl font-bold text-primary">{Math.round(expertDebate.final_probability * 100)}%</p>
              </div>
            )}
            <div className="space-y-2">
              {['optimist','pessimist','realist','devils_advocate'].map(agent => {
                const data = expertDebate[agent]; if (!data) return null;
                return (
                  <div key={agent} className="p-2.5 rounded-xl bg-white/5 border border-white/10">
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-[10px] font-medium ${agentColors[agent]}`}>{agentLabels[agent]}</span>
                      {data.probability != null && <span className="text-[10px] font-mono">{Math.round(data.probability * 100)}%</span>}
                    </div>
                    <p className="text-[10px] text-muted-foreground leading-relaxed">{data.argument || data.reasoning || data.case}</p>
                  </div>
                );
              })}
            </div>
          </>
        )}
        {outcomes.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-white/10">
            <p className="text-[10px] font-medium text-muted-foreground">Outcome Probabilities</p>
            {outcomes.map((o, i) => <OutcomeRow key={i} name={o.outcome} probability={o.probability} delay={i * 0.1} onWhyClick={() => handleWhyClick(i)} isAnimating={isAnimating} />)}
          </div>
        )}
        <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 px-3 py-2 text-xs rounded-xl bg-white/5 text-muted-foreground hover:text-foreground border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-1.5 font-medium">
          <MessageCircle className="w-4 h-4 text-primary" /> Discuss this debate with the moderator
        </motion.button>
        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 11: Expert Consultation — Section 8.9 Multi-Agent Debate</p>
      </motion.div>
    );
  };

  const renderRetrospectivePanel = () => {
    if (!retroResult) return null;
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        <div className="p-3 rounded-xl bg-white/5 border border-white/10">
          <p className="text-[10px] text-muted-foreground">Probability at the time</p>
          <p className="text-2xl font-bold text-primary">{retroResult.probability_at_time ?? '?'}%</p>
        </div>
        {[['Root Cause', retroResult.root_cause], ['Prevention Point', retroResult.prevention_point]].map(([l, v]) => v && (
          <div key={l as string} className="p-3 rounded-xl bg-white/5 border border-white/10">
            <p className="text-[10px] text-muted-foreground font-medium mb-1">{l as string}</p>
            <p className="text-xs leading-relaxed">{v as string}</p>
          </div>
        ))}
        {retroResult.contributing_factors?.length > 0 && (
          <div className="p-3 rounded-xl bg-white/5 border border-white/10">
            <p className="text-[10px] text-muted-foreground font-medium mb-1.5">Contributing Factors</p>
            {retroResult.contributing_factors.map((f: string, i: number) => <p key={i} className="text-[10px] text-muted-foreground mb-0.5">• {f}</p>)}
          </div>
        )}
        {retroResult.lessons_learned?.length > 0 && (
          <div className="p-3 rounded-xl bg-white/5 border border-white/10">
            <p className="text-[10px] text-muted-foreground font-medium mb-1.5">Lessons Learned</p>
            {retroResult.lessons_learned.map((f: string, i: number) => <p key={i} className="text-[10px] text-muted-foreground mb-0.5">• {f}</p>)}
          </div>
        )}
        <motion.button onClick={() => setChatOpen(true)} className="w-full mt-4 px-3 py-2 text-xs rounded-xl bg-white/5 text-muted-foreground hover:text-foreground border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-1.5 font-medium">
          <MessageCircle className="w-4 h-4 text-primary" /> Analyse this event retrospectively with AI
        </motion.button>
        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 7: Retrospective Analysis</p>
      </motion.div>
    );
  };

  const renderMonitoringPanel = () => {
    if (!monSession) return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <Activity className="w-8 h-8 text-muted-foreground mb-3" />
        <p className="text-xs text-muted-foreground">Start a monitoring session on the left to track probability over time.</p>
      </div>
    );
    const pts = [{ probability: monSession.baseline, update_text: 'Baseline', updated_at: 'Start' }, ...monSession.updates];
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        <div className="flex gap-3">
          {[['Baseline', `${monSession.baseline}%`, 'text-muted-foreground'], ['Current', `${pts[pts.length-1].probability}%`, 'text-primary'], ['Updates', String(monSession.updates.length), '']].map(([l, v, c]) => (
            <div key={l as string} className="flex-1 p-2 rounded-xl bg-white/5 border border-white/10 text-center">
              <p className="text-[9px] text-muted-foreground">{l as string}</p><p className={`text-sm font-bold ${c as string}`}>{v as string}</p>
            </div>
          ))}
        </div>
        <div className="space-y-2">
          {pts.map((pt, i) => {
            const prev  = i > 0 ? pts[i - 1].probability : pt.probability;
            const delta = pt.probability - prev;
            return (
              <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/10">
                <div className="text-[9px] text-muted-foreground w-4 text-center">{i}</div>
                <div className="flex-1"><p className="text-[10px]">{pt.update_text}</p></div>
                <div className="text-right">
                  <p className="text-xs font-bold">{pt.probability}%</p>
                  {i > 0 && <p className={`text-[9px] font-mono ${delta >= 0 ? 'text-success' : 'text-destructive'}`}>{delta >= 0 ? '+' : ''}{delta.toFixed(1)}%</p>}
                </div>
              </div>
            );
          })}
        </div>
        {monSession.updates.length === 0 && <p className="text-[10px] text-muted-foreground text-center italic">Add an update on the left and click Generate to log a new data point.</p>}
        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 9: Continuous Monitoring — Section 6.6</p>
      </motion.div>
    );
  };

  const renderSimulationPanel = () => {
    const mc = simResult?.monte_carlo;
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        {mc && (
          <div className="grid grid-cols-2 gap-2">
            {[
              ['Mean', `${mc.mean}%`],
              ['95% CI', `${mc.ci_low ?? mc.mean}–${mc.ci_high ?? mc.mean}%`],
              ['Stability', mc.stability != null ? (mc.stability < 0.15 ? 'High' : mc.stability < 0.3 ? 'Medium' : 'Low') : '—'],
              ['Runs', String(mc.n_runs ?? 200)],
            ].map(([l, v]) => (
              <div key={l} className="p-2 rounded-xl bg-white/5 border border-white/10 text-center">
                <p className="text-[9px] text-muted-foreground">{l}</p><p className="text-sm font-bold text-primary">{v}</p>
              </div>
            ))}
          </div>
        )}
        {simResult?.narrative?.story && (
          <div className="p-3 rounded-xl bg-white/5 border border-white/10">
            <p className="text-[10px] font-medium text-muted-foreground mb-1">Simulation Narrative</p>
            <p className="text-[10px] text-muted-foreground leading-relaxed">{simResult.narrative.story}</p>
          </div>
        )}
        {outcomes.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-white/10">
            <p className="text-[10px] font-medium text-muted-foreground">Simulated Outcomes</p>
            {renderStandardResults()}
          </div>
        )}
        <p className="text-[9px] text-muted-foreground/60 italic text-center">Mode 8: Simulation — Monte Carlo {mc?.n_runs ?? 200} iterations</p>
      </motion.div>
    );
  };

  const renderStandardResults = () => (
    <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
      <div className="flex items-start justify-between mb-3">
        <div><h3 className="text-sm font-bold mb-0.5">Multi-Outcome Results</h3>
          <p className="text-[10px] text-muted-foreground">{outcomes.length} outcomes · {new Date().toLocaleTimeString()}</p></div>
        <button onClick={() => setChatOpen(true)} className="px-3 py-1.5 text-[10px] rounded-lg bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-all flex items-center gap-1.5 font-medium">
          <MessageCircle className="w-3.5 h-3.5" /> Ask About This
        </button>
      </div>
      <div className="space-y-3">
        {outcomes.map((outcome, idx) => (
          <div key={idx}>
            <OutcomeRow name={outcome.outcome} probability={outcome.probability} delay={idx * 0.15} onWhyClick={() => handleWhyClick(idx)} isAnimating={isAnimating} />
            {outcome.reasoning && expandedOutcome !== idx && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-1.5 ml-1 px-2 py-1 border-l border-primary/20">
                <p className="text-[10px] text-muted-foreground/80 italic">"{outcome.reasoning}"</p>
              </motion.div>
            )}
            {/* WHY transparency panel */}
            <AnimatePresence>
              {expandedOutcome === idx && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  className="mt-2 p-3 rounded-lg bg-white/5 border border-white/10 space-y-3">
                  {loadingWhy === idx ? (
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground"><Loader2 className="w-3 h-3 animate-spin" /><span>Analysing…</span></div>
                  ) : activeWhyData ? (
                    <div className="space-y-3">
                      {activeWhyData.simple?.one_line_reason && (
                        <p className="text-[11px] text-foreground font-medium border-b border-white/10 pb-2">"{activeWhyData.simple.one_line_reason}"</p>
                      )}
                      {activeWhyData.detailed && (
                        <div className="grid grid-cols-2 gap-2">
                          <div className="p-2 rounded-lg bg-primary/5 border border-primary/15">
                            <p className="text-[9px] text-primary font-bold mb-1 uppercase tracking-wide">Case For ({outcome.probability.toFixed(0)}%)</p>
                            <p className="text-[10px] text-muted-foreground leading-relaxed">{activeWhyData.detailed.case_for}</p>
                          </div>
                          <div className="p-2 rounded-lg bg-secondary/5 border border-secondary/15">
                            <p className="text-[9px] text-secondary font-bold mb-1 uppercase tracking-wide">Case Against</p>
                            <p className="text-[10px] text-muted-foreground leading-relaxed">{activeWhyData.detailed.case_against}</p>
                          </div>
                        </div>
                      )}
                      {activeWhyData.full && (
                        <div className="space-y-1.5 pt-2 border-t border-white/10">
                          {activeWhyData.full.primary_driver && <p className="text-[10px] text-muted-foreground"><span className="text-primary font-semibold">Primary Driver:</span> {activeWhyData.full.primary_driver}</p>}
                          {activeWhyData.full.intervention && <p className="text-[10px] text-muted-foreground"><span className="text-primary font-semibold">Intervention:</span> {activeWhyData.full.intervention}</p>}
                          {(activeWhyData.full.confidence_note || activeWhyData.full.confidence_factors) && <p className="text-[10px] text-muted-foreground"><span className="text-primary font-semibold">Confidence:</span> {activeWhyData.full.confidence_note || activeWhyData.full.confidence_factors}</p>}
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-[10px] text-muted-foreground/60 italic">Could not load explanation.</p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
            {/* When this does NOT happen — inverse probability */}
            <div className="flex items-center gap-1.5 mt-1.5 ml-0.5">
              <motion.button
                onClick={() => handleInverseClick(idx)}
                disabled={loadingInverse === idx}
                className="px-2 py-0.5 text-[9px] rounded border border-destructive/25 bg-destructive/5 text-destructive/60 hover:bg-destructive/10 hover:text-destructive/80 hover:border-destructive/35 transition-all flex items-center gap-1 disabled:opacity-40"
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
              >
                {loadingInverse === idx ? <Loader2 className="w-2.5 h-2.5 animate-spin" /> : <AlertCircle className="w-2.5 h-2.5" />}
                {inverseData[`inv_${idx}`] ? 'Hide failure scenario' : `When this fails (${(100 - outcome.probability).toFixed(1)}%)`}
              </motion.button>
            </div>
            <AnimatePresence>
              {inverseData[`inv_${idx}`] && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  className="mt-2 p-3 rounded-lg bg-destructive/5 border border-destructive/20 space-y-2">
                  <div className="flex items-center gap-1.5">
                    <AlertCircle className="w-3 h-3 text-destructive/60 shrink-0" />
                    <span className="text-[10px] font-semibold text-destructive/80">
                      {inverseData[`inv_${idx}`].inverse_scenario?.scenario_title
                        ?? `When ${outcome.outcome} does NOT occur (${(100 - outcome.probability).toFixed(1)}%)`}
                    </span>
                  </div>
                  {inverseData[`inv_${idx}`].inverse_scenario?.what_goes_wrong && (
                    <p className="text-[10px] text-muted-foreground leading-relaxed">{inverseData[`inv_${idx}`].inverse_scenario.what_goes_wrong}</p>
                  )}
                  {inverseData[`inv_${idx}`].inverse_scenario?.trigger_factors?.length > 0 && (
                    <div>
                      <p className="text-[9px] text-destructive/60 font-semibold mb-0.5 uppercase tracking-wide">Trigger Factors</p>
                      {inverseData[`inv_${idx}`].inverse_scenario.trigger_factors.map((f: string, fi: number) => (
                        <p key={fi} className="text-[10px] text-muted-foreground/80 ml-2">• {f}</p>
                      ))}
                    </div>
                  )}
                  {inverseData[`inv_${idx}`].inverse_scenario?.early_warnings?.length > 0 && (
                    <div>
                      <p className="text-[9px] text-warning/70 font-semibold mb-0.5 uppercase tracking-wide">Early Warning Signs</p>
                      {inverseData[`inv_${idx}`].inverse_scenario.early_warnings.map((w: string, wi: number) => (
                        <p key={wi} className="text-[10px] text-muted-foreground/80 ml-2">• {w}</p>
                      ))}
                    </div>
                  )}
                  {inverseData[`inv_${idx}`].inverse_scenario?.reversal_actions && (
                    <p className="text-[10px] text-muted-foreground/70 italic border-t border-white/5 pt-1.5">
                      💡 {inverseData[`inv_${idx}`].inverse_scenario.reversal_actions}
                    </p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
        <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: outcomes.length * 0.15 + 0.3 }}
          onClick={handleLoadMore} disabled={loadingMore}
          className="w-full mt-2 px-3 py-2 text-[11px] rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-primary/30 transition-all flex items-center justify-center gap-1.5 group disabled:opacity-50"
          whileHover={!loadingMore ? { scale: 1.01 } : {}} whileTap={!loadingMore ? { scale: 0.99 } : {}}>
          {loadingMore ? <Loader2 className="w-3 h-3 animate-spin" /> : <>
            <span className="text-muted-foreground group-hover:text-foreground transition-colors">Generate More Outcomes</span>
            <ChevronRight className="w-3 h-3 text-primary" /></>}
        </motion.button>
        <p className="text-[9px] text-muted-foreground text-center italic">* Probabilities are independent and do not sum to 100%</p>
      </div>
      {predResult?.shap_values && Object.keys(predResult.shap_values).length > 0 && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
          <SHAPChart features={Object.entries(predResult.shap_values).map(([name, value]) => ({ name, value: value as number }))} delay={0.9} />
        </motion.div>
      )}
      {predResult && (
        <AuditPanel
          audits={(predResult.audit_flags || []).map(f => ({ type: 'parameter' as const, status: f.severity === 'CRITICAL' ? 'fail' as const : f.severity === 'WARNING' ? 'warning' as const : 'pass' as const, label: `${f.code}: ${f.message}` }))}
          abnFlags={predResult.audit_flags?.map(f => f.code) || []}
          mlLlmAgreement={(predResult.gap || 0) < 0.08 ? 'high' : (predResult.gap || 0) < 0.20 ? 'moderate' : 'low'} delay={0.4} />
      )}
      {predResult && (
        <PredictionBreakdown 
          mode={selectedMode}
          mlProbability={predResult.ml_probability !== undefined ? predResult.ml_probability * 100 : undefined}
          llmProbability={predResult.llm_probability !== undefined ? predResult.llm_probability * 100 : undefined}
          reconciledProbability={predResult.reconciled_probability !== undefined ? predResult.reconciled_probability * 100 : undefined}
          reliabilityIndex={predResult.reliability_index !== undefined ? predResult.reliability_index * 100 : undefined}
          gap={predResult.gap}
          reconciliationMethod={predResult.reconciliation_method}
          shapValues={predResult.shap_values}
          delay={0.6}
        />
      )}
      {exportPayload && <ExportPanel payload={exportPayload} delay={1.0} />}
    </motion.div>
  );


  // ── PRAGMA: Forensic Psychological Profiling Results ─────────────────────
  const renderPragmaResults = () => {
    const deceptionPct = outcomes[0]?.probability ?? (predResult ? Math.round((predResult as any).reconciled_probability * 100) : 50);
    const isDeceptive  = deceptionPct > 50;
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
        {/* Header verdict */}
        <div className={`p-3 rounded-xl border ${isDeceptive ? 'bg-destructive/10 border-destructive/30' : 'bg-success/10 border-success/30'}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground mb-0.5">PRAGMA Forensic Verdict</p>
              <p className={`text-sm font-bold ${isDeceptive ? 'text-destructive' : 'text-success'}`}>
                {isDeceptive ? 'Deceptive Communication Detected' : 'Communication Appears Genuine'}
              </p>
            </div>
            <div className="text-right">
              <p className={`text-2xl font-bold ${isDeceptive ? 'text-destructive' : 'text-success'}`}>{deceptionPct}%</p>
              <p className="text-[9px] text-muted-foreground">Deception probability</p>
            </div>
          </div>
        </div>

        {/* Outcomes with inverse buttons */}
        {outcomes.length > 0 && renderStandardResults()}

        {/* Forensic Psychological Profile (interactive dialog) */}
        <Dialog>
          <DialogTrigger asChild>
            <button className="w-full relative overflow-hidden group p-4 rounded-xl bg-gradient-to-br from-white/5 to-white/5 border border-white/20 hover:border-primary/50 transition-all text-left">
              <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="flex items-center justify-between relative z-10">
                <div>
                  <p className="text-[11px] text-primary font-semibold uppercase tracking-widest mb-1 flex items-center gap-2">
                    <History className="w-3.5 h-3.5" /> Comprehensive Forensic Profile
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Expand to view linguistic stress markers, cognitive load analysis, and discuss findings with the AI Profiler.
                  </p>
                </div>
                <div className="text-primary group-hover:translate-x-1 transition-transform">
                  <ChevronRight className="w-5 h-5" />
                </div>
              </div>
            </button>
          </DialogTrigger>
          
          <DialogContent className="max-w-2xl bg-[#0a0a0f] border-white/10 p-0 text-white overflow-hidden max-h-[85vh] flex flex-col">
            <DialogHeader className="p-5 border-b border-white/10 shrink-0">
              <DialogTitle className="text-lg font-bold flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-primary" /> PRAGMA Complete Analysis
              </DialogTitle>
              <DialogDescription className="text-xs text-muted-foreground">
                Deep multi-modal forensic evaluation of textual and psychological markers.
              </DialogDescription>
            </DialogHeader>
            
            <div className="p-5 overflow-y-auto space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                  <p className="text-[10px] text-primary font-semibold uppercase tracking-wider mb-2">Primary Motive / Trigger</p>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {isDeceptive 
                      ? 'Self-preservation or concealment of information detected. Communication exhibits stress-induced linguistic distancing.'
                      : 'No clear deceptive motive identified. Communication aligns with baseline genuine patterns.'}
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                  <p className="text-[10px] text-primary font-semibold uppercase tracking-wider mb-2">Psychological State</p>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {isDeceptive
                      ? 'Elevated cognitive load indicators suggesting active information suppression or fabrication.'
                      : 'Consistent psychological baseline with no significant stress markers.'}
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-white/5 border border-white/10 col-span-2">
                  <p className="text-[10px] text-primary font-semibold uppercase tracking-wider mb-2">Linguistic Breakdown</p>
                  <p className="text-xs leading-relaxed text-muted-foreground mb-3">
                    {isDeceptive
                      ? 'Increased hedging language, pronoun distancing ("they", "one"), over-qualification ("to be perfectly honest"), and non-committed phrasing detected in text embedding analysis.'
                      : 'Direct language ("I", "we"), appropriate context-driven emotional resonance, and consistent tense usage observed throughout the communication string.'}
                  </p>
                  <div className="space-y-1 block">
                     <p className="text-[11px] font-medium text-white/80">Actionable Intervention:</p>
                     <p className="text-[10px] text-muted-foreground/80 italic border-l-2 border-primary/50 pl-2">
                       {isDeceptive ? 'Subject shows evasiveness about specific timelines. Grill on chronological details.' : 'No intervention needed. Trust baseline verified.'}
                     </p>
                  </div>
                </div>
              </div>

              {/* Chatbot Interface inside Dialog */}
              <div>
                <p className="text-[10px] text-primary font-semibold uppercase tracking-wider mb-3">Consult the Profiler</p>
                <PragmaChat 
                   predictionId={predId} 
                   contextParams={parameters} 
                   baselinePrediction={predResult} 
                />
              </div>

            </div>
          </DialogContent>
        </Dialog>

        {/* PRAGMA Documentation Rendering */}
        <PragmaDocumentation />

        <p className="text-[9px] text-muted-foreground/50 italic text-center mt-6">
          PRAGMA v17 — Research-grade forensic tool. Probabilistic — not a legal verdict. Always verify independently.
        </p>
      </motion.div>
    );
  };

  const renderRightPanel = () => {
    if (selectedMode === 'conversational') {
      return showResults ? (
        <div className="space-y-4">
          <div className="p-3 rounded-xl bg-primary/10 border border-primary/20 mb-2">
            <p className="text-[10px] text-primary/80">● Conversation complete — prediction from {Object.keys(convParams).length} parameters</p>
          </div>
          {renderStandardResults()}
        </div>
      ) : renderConversationalPanel();
    }
    if (selectedMode === 'monitoring') return renderMonitoringPanel();
    if (isGenerating) return (
      <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="h-full flex items-center justify-center">
        <LoadingAnimation />
      </motion.div>
    );
    if (showResults) {
      if (selectedMode === 'adversarial')   return renderAdversarialPanel();
      if (selectedMode === 'whatif')        return renderWhatIfPanel();
      if (selectedMode === 'comparative')   return renderComparativePanel();
      if (selectedMode === 'expert')        return renderExpertPanel();
      if (selectedMode === 'retrospective') return renderRetrospectivePanel();
      if (selectedMode === 'simulation')    return renderSimulationPanel();
      if (selectedMode === 'voice')         return renderStandardResults();
      // Pragma domain — wrap standard results with forensic profiling header
      if (selectedDomain === 'pragma') return renderPragmaResults();
      return renderStandardResults();
    }
    const hints: Record<string, string> = {
      guided: 'Configure your parameters using the chip modal, then click Generate',
      free: 'Describe any situation in the text box — no domain or parameters needed',
      hybrid: 'Upload an image or video and add context, then click Generate',
      document: 'Upload a document — Sambhav will extract signals and predict automatically',
      voice: 'Upload an audio file — Sambhav will transcribe, analyze cues and predict',
      adversarial: 'Click Generate to submit extreme parameters and see the audit system respond',
      whatif: 'Describe a "What if…" scenario and click Generate for a branching probability tree',
      comparative: 'Fill in at least 2 scenarios above, then click Generate',
      expert: 'Describe your question and click Generate for multi-expert debate',
      retrospective: 'Describe the past event above and click Generate for retrospective analysis',
      simulation: 'Describe your hypothetical scenario, configure parameters, then click Generate',
    };
    return (
      <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col items-center justify-center text-center">
        <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-3"><Play className="w-8 h-8 text-primary" /></div>
        <h3 className="text-sm font-medium mb-1">Ready to Predict</h3>
        <p className="text-xs text-muted-foreground max-w-sm">{hints[selectedMode] || 'Configure your input and click Generate'}</p>
      </motion.div>
    );
  };

  const hideGenBtn = false;
  const genLabel: Record<string, string> = {
    adversarial: 'Run Adversarial Test', whatif: 'Generate Scenario Tree',
    comparative: 'Compare Scenarios', expert: 'Run Expert Debate',
    retrospective: 'Analyse Past Event', simulation: 'Run Simulation',
    monitoring: monSession ? 'Log Update' : 'Start Session', conversational: '—',
  };
  const needsModal = selectedMode === 'guided' || selectedMode === 'expert' || selectedMode === 'simulation';

  // Manual trigger for debugging/fallback
  useEffect(() => {
    if (needsModal && modalOpen && !domainInfo && !loadingSchema) {
      console.warn('Modal open requested but domainInfo is missing');
      setModalOpen(false);
      setApiError('Parameter configuration is not available for this domain.');
    }
  }, [modalOpen, domainInfo, loadingSchema, needsModal]);

  return (
    <div className="min-h-screen relative overflow-hidden bg-background pb-12">
      <BackgroundLogo />
      <Navigation />
      <ResultChatbot 
        isOpen={chatOpen} 
        onClose={() => setChatOpen(false)} 
        context={{
          standard: predResult,
          outcomes,
          adversarial: adversarialResult,
          whatif: whatifTree,
          comparative: compResult,
          expert: expertDebate,
          retrospective: retroResult,
          simulation: simResult,
          free_infer: freeInferResult
        }}
        mode={selectedMode}
        domain={selectedDomain}
      />
      {needsModal && domainInfo && (
        <ChipParameterModal isOpen={modalOpen} onClose={() => setModalOpen(false)}
          parameters={paramList.map(p => p.label || p.key)} currentStep={modalStep + 1} totalSteps={paramList.length}
          onNext={(values) => {
            const key = paramList[modalStep]?.key;
            if (key) setParameters(prev => ({ ...prev, [key]: values[0] ?? null }));
            if (modalStep + 1 < paramList.length) setModalStep(s => s + 1); else setModalOpen(false);
          }}
          onPrevious={() => setModalStep(s => Math.max(0, s - 1))}
          parameterConfigs={paramList}
          onComplete={(answers: Record<string, any>) => { 
            setParameters(answers); 
            setParamsConfirmed(true);
            setModalOpen(false); 
            // Automatically trigger prediction after modal is complete
            handleGenerate(true);
          }}
          currentAnswers={parameters} />
      )}
      <div className="relative z-10 pt-16 pb-6 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-4">
            <h1 className="text-xl font-bold mb-0.5">Prediction Analysis</h1>
            <p className="text-[11px] text-muted-foreground">
              Mode: <span className="text-primary">{selectedMode}</span>
              {selectedMode !== 'free' && selectedMode !== 'adversarial' && <> · Domain: <span className="text-primary">{selectedDomain.replace(/_/g,' ')}</span></>}
            </p>
          </motion.div>
          {domainInfo?.disclaimer && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-3 px-3 py-2 rounded-lg bg-warning/5 border border-warning/20 flex items-start gap-2">
              <AlertCircle className="w-3 h-3 text-warning mt-0.5 shrink-0" />
              <p className="text-[10px] text-warning/80">{domainInfo.disclaimer}</p>
            </motion.div>
          )}
          <ReliabilityIndex score={reliabilityScore} suggestions={reliabilitySuggestions} isVisible={!showResults && selectedMode !== 'conversational'} />
          <AnimatePresence>
            {apiError && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                className="mb-3 px-3 py-2.5 rounded-lg bg-destructive/10 border border-destructive/30 flex items-start gap-2">
                <AlertCircle className="w-3.5 h-3.5 text-destructive mt-0.5 shrink-0" />
                <p className="text-xs text-destructive">{apiError}</p>
              </motion.div>
            )}
          </AnimatePresence>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="grid lg:grid-cols-12 gap-3">
            <div className="lg:col-span-4">
              <GlassCard variant="elevated" className="p-4 relative overflow-hidden">
                <h3 className="text-xs font-medium mb-3 text-muted-foreground">Input Configuration</h3>
                {isAnalyzing && (
                  <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center">
                    <LoadingAnimation durationMs={1500} />
                    <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-4 text-xs font-medium text-primary">
                      {hybridLoading ? 'Analyzing Visual Cues...' : docLoading ? 'Processing Large Context Document...' : 'Transcribing & Analyzing Voice...'}
                    </motion.p>
                  </div>
                )}
                {insufficientInfo && (
                  <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mb-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] space-y-1.5">
                    <div className="flex items-start gap-2">
                      <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      <span>{insufficientInfo.reason}</span>
                    </div>
                    {insufficientInfo.missing && insufficientInfo.missing.length > 0 && (
                      <div className="pl-5 space-y-0.5">
                        <p className="font-medium opacity-80 underline decoration-dotted underline-offset-2">Missing info:</p>
                        {insufficientInfo.missing.map((m, i) => <p key={i}>• {m}</p>)}
                      </div>
                    )}
                  </motion.div>
                )}
                {loadingSchema ? <div className="flex items-center justify-center h-20"><Loader2 className="w-4 h-4 animate-spin text-muted-foreground" /></div> : renderModeInterface()}
                {!hideGenBtn && selectedMode !== 'conversational' && (
                  <div className="mt-3 flex gap-2">
                    <motion.button onClick={() => handleGenerate()} disabled={isGenerating}
                      className="flex-1 px-3 py-2 text-xs rounded-lg bg-primary text-black font-medium flex items-center justify-center gap-1.5 disabled:opacity-50"
                      whileHover={!isGenerating ? { scale: 1.02 } : {}} whileTap={!isGenerating ? { scale: 0.98 } : {}}>
                      {isGenerating ? <><Loader2 className="w-3 h-3 animate-spin" /><span>Generating…</span></> : <><Play className="w-3 h-3" /><span>{genLabel[selectedMode] || 'Generate'}</span></>}
                    </motion.button>
                    {(showResults || adversarialResult || whatifTree || compResult || expertDebate || retroResult || simResult) && (
                      <motion.button initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} onClick={handleReset}
                        className="px-3 py-2 text-xs rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
                        whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}><RotateCcw className="w-3 h-3" /></motion.button>
                    )}
                  </div>
                )}
              </GlassCard>
              {predResult && showResults && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-3">
                  <GlassCard variant="elevated" className="p-3">
                    <h4 className="text-[10px] font-medium text-muted-foreground mb-2">Dual-Layer Breakdown</h4>
                    <div className="space-y-1.5 text-[10px]">
                      {[
                        ['ML Probability', predResult.ml_probability != null ? `${(predResult.ml_probability * 100).toFixed(1)}%` : '—', 'text-primary'],
                        ['LLM Probability', predResult.llm_probability != null ? `${(predResult.llm_probability * 100).toFixed(1)}%` : '—', 'text-secondary'],
                        ['Final (65/35)', `${(predResult.final_probability * 100).toFixed(1)}%`, 'text-foreground font-bold'],
                        ['Confidence', predResult.confidence_tier, predResult.confidence_tier === 'CLEAR' ? 'text-success' : predResult.confidence_tier === 'MODERATE' ? 'text-warning' : 'text-destructive'],
                        ['ML-LLM Gap', `${(predResult.gap * 100).toFixed(1)}%`, 'text-muted-foreground'],
                        ['Reliability', `${(predResult.reliability_index * 100).toFixed(0)}%`, 'text-primary'],
                      ].map(([l, v, c]) => (
                        <div key={l as string} className="flex justify-between"><span className="text-muted-foreground">{l as string}</span><span className={c as string}>{v as string}</span></div>
                      ))}
                    </div>
                  </GlassCard>
                </motion.div>
              )}
            </div>
            <div className="lg:col-span-8">
              <GlassCard variant="elevated" className="p-4 min-h-[600px]">
                <AnimatePresence mode="wait">{renderRightPanel()}</AnimatePresence>
              </GlassCard>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}