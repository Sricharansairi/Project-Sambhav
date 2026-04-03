// ── Operating Modes — all 12 ────────────────────────────────
export const OPERATING_MODES = [
  { id: 'guided',         name: 'Guided Mode',          description: 'Step-by-step parameter collection via chip modal', icon: 'navigation' },
  { id: 'free',           name: 'Free Inference',        description: 'Unstructured text input — no domain needed',       icon: 'text' },
  { id: 'hybrid',         name: 'Hybrid Mode',           description: 'Text + media upload',                              icon: 'layers' },
  { id: 'conversational', name: 'Conversational',        description: 'Multi-turn dialogue — one question at a time',     icon: 'message-circle' },
  { id: 'document',       name: 'Document Analysis',     description: 'PDF/Word file analysis',                           icon: 'file-text' },
  { id: 'comparative',    name: 'Comparative Mode',      description: 'Side-by-side scenario comparison',                 icon: 'git-compare' },
  { id: 'retrospective',  name: 'Retrospective',         description: 'Past event analysis',                              icon: 'history' },
  { id: 'simulation',     name: 'Simulation Mode',       description: 'Hypothetical scenarios',                           icon: 'flask' },
  { id: 'monitoring',     name: 'Continuous Monitoring', description: 'Real-time probability tracking',                   icon: 'activity' },
  { id: 'adversarial',    name: 'Adversarial Mode',      description: 'Stress-test predictions with extreme inputs',      icon: 'shield-alert' },
  { id: 'expert',         name: 'Expert Consultation',   description: 'Simulated multi-expert perspectives',              icon: 'users' },
  { id: 'whatif',         name: 'What-If Story',         description: 'Branching scenario probability trees',             icon: 'git-branch' },
];

// ── Domains — 11 only, IDs match domain_registry.yaml keys exactly ──
export const DOMAINS = [
  { id: 'pragma',           name: 'Pragma',           subtitle: 'Forensic Psychology',    color: '#c0c0c0' },
  { id: 'sarvagna',         name: 'Sarvagna',         subtitle: 'Oracle Multi-Domain',    color: '#c0c0c0' },
  { id: 'claim',            name: 'Claim',            subtitle: 'Fact-Checking',          color: '#c0c0c0' },
  { id: 'behavioral',       name: 'Behavioral',       subtitle: 'Behavioral Analysis',    color: '#c0c0c0' },
  { id: 'mental_health',    name: 'Mental Health',    subtitle: 'Psychological Insights', color: '#c0c0c0' },
  { id: 'financial',        name: 'Financial',        subtitle: 'Credit Assessment',      color: '#c0c0c0' },
  { id: 'fitness',          name: 'Fitness',          subtitle: 'Health & Wellness',      color: '#c0c0c0' },
  { id: 'job_life',         name: 'Job Life',         subtitle: 'Human Resources',        color: '#c0c0c0' },
  { id: 'high_school',      name: 'High School',      subtitle: 'Academic Outcomes',      color: '#c0c0c0' },
  { id: 'health',           name: 'Health',           subtitle: 'Medical Diagnosis',      color: '#c0c0c0' },
  { id: 'student',          name: 'Student',          subtitle: 'Student Performance',    color: '#c0c0c0' },
];

// ── Credibility Dimensions for Fact-Check (8 dimensions matching backend) ──
export const CREDIBILITY_DIMENSIONS = [
  { id: 'factual_accuracy',      name: 'Factual Accuracy',      weight: 0.18 },
  { id: 'temporal_accuracy',     name: 'Temporal Accuracy',     weight: 0.12 },
  { id: 'geographic_accuracy',   name: 'Geographic Accuracy',   weight: 0.10 },
  { id: 'source_reliability',    name: 'Source Reliability',    weight: 0.15 },
  { id: 'linguistic_precision',  name: 'Linguistic Precision',  weight: 0.12 },
  { id: 'context_completeness',  name: 'Context Completeness',  weight: 0.13 },
  { id: 'intent_analysis',       name: 'Intent Analysis',       weight: 0.10 },
  { id: 'viral_risk',            name: 'Viral Risk',            weight: 0.10 },
];

// ── Domain ID → display label map (for UI) ──────────────────
export const DOMAIN_LABELS: Record<string, string> = {
  pragma:           'Pragma',
  sarvagna:         'Sarvagna',
  claim:            'Claim',
  behavioral:       'Behavioral',
  mental_health:    'Mental Health',
  financial:        'Financial',
  fitness:          'Fitness',
  job_life:         'Job Life',
  high_school:      'High School',
  health:           'Health',
  student:          'Student',
};

/**
 * MODES_SKIP_DOMAIN — modes that bypass the domain-selection screen entirely.
 *
 * FIX (BUG 3): Previously only ['free', 'document'].
 * Per spec: domain selection is ONLY for Guided Mode.
 * All other 11 modes skip straight to /prediction with a sensible default domain.
 *
 * - free          → sarvagna (LLM free inference, no ML model needed)
 * - document      → sarvagna (document analysis, LLM-primary)
 * - conversational→ student  (LLM drives questions, uses most common domain)
 * - hybrid        → student  (vision extracts params, can be any domain)
 * - comparative   → student  (compare scenarios, domain applied per-scenario)
 * - retrospective → student  (past event analysis)
 * - simulation    → student  (hypothetical)
 * - monitoring    → student  (ongoing project monitoring)
 * - adversarial   → student  (stress-testing, uses student as demo domain)
 * - expert        → student  (multi-expert, common demo domain)
 * - whatif        → student  (branching scenarios)
 *
 * Guided Mode is the ONLY mode that shows domain selection.
 */
export const MODES_SKIP_DOMAIN: string[] = [
  'free',
  'document',
  'conversational',
  'hybrid',
  'comparative',
  'retrospective',
  'simulation',
  'monitoring',
  'adversarial',
  'whatif',
];

/**
 * Default domain to use when a mode skips domain selection.
 * Text-heavy modes use 'sarvagna' (LLM-primary, no ML model).
 * All others use 'student' as the most common demonstration domain.
 */
export const MODE_DEFAULT_DOMAIN: Record<string, string> = {
  free:           'sarvagna',
  document:       'sarvagna',
  conversational: 'student',
  hybrid:         'student',
  comparative:    'student',
  retrospective:  'student',
  simulation:     'student',
  monitoring:     'student',
  adversarial:    'student',
  whatif:         'student',
};