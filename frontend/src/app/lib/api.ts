/**
 * api.ts — Centralised API client for Project Sambhav frontend.
 * All backend calls go through here. Never hardcode URLs elsewhere.
 */

const API_BASE = (import.meta as any).env.VITE_API_URL ?? '/api';

// ── Auth token management ─────────────────────────────────────
let _token: string | null = localStorage.getItem('sambhav_token');

export function setToken(token: string) {
  _token = token;
  localStorage.setItem('sambhav_token', token);
}
export function clearToken() {
  _token = null;
  localStorage.removeItem('sambhav_token');
}
export function getToken() { return _token; }

function authHeaders(): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' };
  if (_token) h['Authorization'] = `Bearer ${_token}`;
  return h;
}

async function _post(path: string, body: object) {
  const res = await fetch(`${API_BASE}${path}`, {
    method:  'POST',
    headers: authHeaders(),
    body:    JSON.stringify(body),
  });

  if (res.status === 401) {
    const bypassAuth = (import.meta as any).env.VITE_BYPASS_AUTH === 'true';
    if (!bypassAuth) {
      clearToken();
      window.location.href = '/auth';
    }
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new SambhavAPIError(res.status, err.detail || err.error || JSON.stringify(err));
  }
  return res.json();
}

async function _get(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  
  // If 401 Unauthorized, only redirect to auth if we are NOT in bypass mode
  if (res.status === 401) {
    const bypassAuth = (import.meta as any).env.VITE_BYPASS_AUTH === 'true';
    if (!bypassAuth) {
      clearToken();
      window.location.href = '/auth';
    }
  }
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new SambhavAPIError(res.status, err.detail || JSON.stringify(err));
  }
  return res.json();
}

async function _postBlob(path: string, body: object): Promise<Blob> {
  const res = await fetch(`${API_BASE}${path}`, {
    method:  'POST',
    headers: authHeaders(),
    body:    JSON.stringify(body),
  });

  if (res.status === 401) {
    const bypassAuth = (import.meta as any).env.VITE_BYPASS_AUTH === 'true';
    if (!bypassAuth) {
      clearToken();
      window.location.href = '/auth';
    }
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new SambhavAPIError(res.status, err.detail || JSON.stringify(err));
  }
  return res.blob();
}

async function _delete(path: string) {
  const res = await fetch(`${API_BASE}${path}`, {
    method:  'DELETE',
    headers: authHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new SambhavAPIError(res.status, err.detail || JSON.stringify(err));
  }
  return res.json();
}

export class SambhavAPIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'SambhavAPIError';
  }
  get isInputQuality() { return this.status === 422; }
  get isBlocked()      { return this.status === 400; }
  get isServerError()  { return this.status >= 500; }
}

// ── Types ─────────────────────────────────────────────────────
export interface DomainParam {
  type:    string;
  label:   string;
  options: string[];
  range:   number[];
  weight:  string;
}

export interface DomainInfo {
  name:             string;
  description:      string;
  prediction_label: string;
  disclaimer:       string | null;
  brier_score:      number | null;
  auc:              number | null;
  status:           string;
  parameters:       Record<string, DomainParam>;
}

export interface PredictionResult {
  domain:            string;
  question:          string;
  ml_probability:    number | null;
  llm_probability:   number | null;
  final_probability: number;
  confidence_tier:   string;
  gap:               number;
  shap_values:       Record<string, number>;
  audit_flags:       Array<{ code: string; severity: string; message: string }>;
  debate:            Record<string, any>;
  reliability_index: number;
  reasoning:         string;
  key_factors:       string[];
  mode:              string;
  raw_parameters:    Record<string, any>;
}

export interface Outcome {
  outcome:          string;
  probability:      number;
  probability_pct:  string;
  reasoning:        string;
  type:             'positive' | 'negative' | 'neutral';
  condition:        string | null;
  has_transparency: boolean;
}

export interface OutcomesResult {
  outcomes:            Outcome[];
  mode:                string;
  interpretation_note: string;
  total_shown:         number;
  can_generate_more:   boolean;
}

export interface TransparencyResult {
  simple:   { dominant_probability: number; minority_probability: number; one_line_reason: string };
  detailed: Record<string, any>;
  full:     Record<string, any>;
}

export interface FactCheckSource {
  title:  string;
  snippet:string;
  link:   string;
  source: string;
}

export interface FactCheckResult {
  verdict:           string;
  credibility_score: number;
  dimensions:        Record<string, { score: number; reasoning: string }>;
  sources:           FactCheckSource[];
  summary:           string;
}

// ── Auth ──────────────────────────────────────────────────────
export const auth = {
  async guest() {
    const data = await _post('/auth/guest', {});
    if (data.token) setToken(data.token);
    return data;
  },
  async login(email: string, password: string) {
    const data = await _post('/auth/login', { email, password });
    if (data.token) setToken(data.token);
    return data;
  },
  async register(email: string, password: string) {
    const data = await _post('/auth/register', { email, password });
    if (data.token) setToken(data.token);
    return data;
  },
  async getMe() {
    return _get('/auth/me');
  },
  async deleteAccount() {
    return _delete('/auth/me');
  },
  async resetPassword(email: string, newPassword: string) {
    return _post('/auth/reset-password', { email, new_password: newPassword });
  },
  logout: clearToken,
};

// ── Domains ───────────────────────────────────────────────────
export async function getDomains(): Promise<Record<string, DomainInfo>> {
  return _get('/predict/domains');
}

// ── Prediction ────────────────────────────────────────────────
export async function runPredict(payload: {
  domain:      string;
  parameters:  Record<string, any>;
  question?:   string;
  skipped?:    string[];
  run_debate?: boolean;
  mode?:       string;
}): Promise<{ success: boolean; prediction_id: string; prediction: PredictionResult }> {
  // Ensure guest token exists
  if (!_token) await auth.guest();
  return _post('/predict', payload);
}

export async function runFreeInfer(text: string, n_outcomes: number = 5) {
  if (!_token) await auth.guest();
  return _post('/predict/free', { text, n_outcomes });
}

export async function getOutcomes(payload: {
  domain:             string;
  parameters:         Record<string, any>;
  question?:          string;
  n_outcomes?:        number;
  existing_outcomes?: Outcome[];
  mode?:              string;
}): Promise<{ success: boolean; result: OutcomesResult }> {
  if (!_token) await auth.guest();
  return _post('/predict/outcomes', payload);
}

export async function getTransparency(payload: {
  domain:             string;
  parameters:         Record<string, any>;
  final_probability?: number;
  question?:          string;
  outcome?:           string;
}): Promise<{ success: boolean; result: TransparencyResult }> {
  if (!_token) await auth.guest();
  return _post('/predict/transparency', payload);
}

export async function getRichPrediction(payload: {
  domain:     string;
  parameters: Record<string, any>;
  question?:  string;
  mode?:      string;
}) {
  if (!_token) await auth.guest();
  return _post('/predict/rich', payload);
}

export async function startConversational(domain: string, question?: string) {
  if (!_token) await auth.guest();
  return _post('/predict/conversational/start', { domain, question });
}

export async function answerConversational(payload: {
  domain: string;
  question?: string;
  param_key: string;
  value: string;
  skipped?: boolean;
  step: number;
  parameters?: Record<string, any>;
  history?: any[];
}) {
  if (!_token) await auth.guest();
  return _post('/predict/conversational/answer', payload);
}

// ── Operating Modes ───────────────────────────────────────────
export async function runWhatIf(payload: {
  domain: string;
  parameters?: Record<string, any>;
  question?: string;
  base_probability?: number;
  depth?: number;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/whatif', payload);
}

export async function runComparative(payload: {
  domain: string;
  scenarios: { label: string; [key: string]: any }[];
  outcomes?: string[];
  question?: string;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/comparative', payload);
}

export async function startMonitoring(payload: {
  name: string;
  domain: string;
  parameters?: Record<string, any>;
  question?: string;
  threshold_low?: number;
  threshold_high?: number;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/monitoring/start', payload);
}

export async function updateMonitoring(payload: {
  session_id: string;
  domain: string;
  parameters: Record<string, any>;
  update_text?: string;
  question?: string;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/monitoring/update', payload);
}

export async function runAdversarial(payload: {
  domain: string;
  parameters?: Record<string, any>;
  question?: string;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/adversarial', payload);
}

export async function runExpertMode(payload: {
  domain: string;
  parameters?: Record<string, any>;
  question?: string;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/expert', payload);
}

export async function runRetrospective(payload: {
  domain: string;
  description: string;
  outcome?: string;
  parameters?: Record<string, any>;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/retrospective', payload);
}

export async function runSimulation(payload: {
  domain: string;
  parameters?: Record<string, any>;
  question?: string;
  n_runs?: number;
}) {
  if (!_token) await auth.guest();
  return _post('/modes/simulation', payload);
}

export async function discoverParams(payload: {
  domain:   string;
  question: string;
}): Promise<{ success: boolean; parameters: any[] }> {
  if (!_token) await auth.guest();
  return _post('/predict/discover-params', payload);
}

export async function screenInput(payload: {
  domain: string;
  text:   string;
}) {
  if (!_token) await auth.guest();
  // Falling back to discoverParams if screen endpoint is replaced or similar
  return discoverParams({ domain: payload.domain, question: payload.text });
}

export async function analyzeDocument(file: File, domain: string, question?: string) {
  if (!_token) await auth.guest();
  const fd = new FormData();
  fd.append('file', file);
  fd.append('domain', domain);
  if (question) fd.append('question', question);
  const BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';
  const resp = await fetch(`${BASE}/modes/document`, {
    method: 'POST',
    headers: _token ? { Authorization: `Bearer ${_token}` } : {},
    body: fd,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new SambhavAPIError(resp.status, err.detail || 'Document analysis failed');
  }
  return resp.json();
}

// ── Fact Check ────────────────────────────────────────────────
export async function factCheck(claim: string): Promise<{ success: boolean; result: FactCheckResult }> {
  if (!_token) await auth.guest();
  return _post('/fact-check', { claim });
}

export async function factCheckBatch(text: string, max_claims: number = 10) {
  if (!_token) await auth.guest();
  return _post('/fact-check/batch', { text, max_claims });
}

// ── History ───────────────────────────────────────────────────
export async function getHistory(): Promise<{ success: boolean; predictions: any[] }> {
  if (!_token) await auth.guest();
  return _get('/history');
}

export async function deleteHistory(predictionId: string): Promise<{ success: boolean }> {
  if (!_token) await auth.guest();
  return _delete(`/history/${predictionId}`);
}

export async function clearHistory(): Promise<{ success: boolean }> {
  if (!_token) await auth.guest();
  return _delete('/history');
}

export async function saveHistory(data: object) {
  if (!_token) await auth.guest();
  return _post('/history/save', data);
}

// ── Export ────────────────────────────────────────────────────
export interface ExportPayload {
  prediction_id?: string;
  domain?:        string;
  parameters?:    Record<string, any>;
  result?:        Record<string, any>;
  question?:      string;
}

async function _exportBlob(format: string, payload: ExportPayload): Promise<Blob> {
  return _postBlob(`/export/${format}`, payload);
}

function _downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href    = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const exports = {
  async json(payload: ExportPayload) {
    const blob = await _exportBlob('json', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.json`);
  },
  async csv(payload: ExportPayload) {
    const blob = await _exportBlob('csv', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.csv`);
  },
  async xml(payload: ExportPayload) {
    const blob = await _exportBlob('xml', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.xml`);
  },
  async pdf(payload: ExportPayload) {
    const blob = await _exportBlob('pdf', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.pdf`);
  },
  async word(payload: ExportPayload) {
    const blob = await _exportBlob('word', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.docx`);
  },
  async excel(payload: ExportPayload) {
    const blob = await _exportBlob('excel', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.xlsx`);
  },
  async png(payload: ExportPayload) {
    const blob = await _exportBlob('png', payload);
    _downloadBlob(blob, `sambhav_${payload.prediction_id || 'export'}.png`);
  },
  async apiLink(payload: ExportPayload) {
    return _post('/export/api-link', payload);
  },
};