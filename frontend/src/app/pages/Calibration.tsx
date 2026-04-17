import { motion, AnimatePresence } from 'motion/react';
import { useState, useEffect } from 'react';
import { Target, TrendingUp, TrendingDown, CheckCircle, XCircle, Clock, BarChart3, Loader2 } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import { getHistory, SambhavAPIError } from '../lib/api';

interface CalibrationRecord {
  id:          string;
  date:        string;
  domain:      string;
  predicted:   number;   // 0-100
  actual?:     number;   // 0-100, null if not yet resolved
  resolved:    boolean;
  error?:      number;   // |predicted - actual|
}

interface BrierBucket {
  label:     string;
  range:     [number, number];
  predicted: number;
  actual:    number;
  count:     number;
}

export function Calibration() {
  const [records,     setRecords]     = useState<CalibrationRecord[]>([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState<string | null>(null);
  const [resolving,   setResolving]   = useState<string | null>(null);
  const [newActual,   setNewActual]   = useState<Record<string, string>>({});

  // Load history from backend
  useEffect(() => {
    setLoading(true);
    getHistory()
      .then((data: any) => {
        const raw = Array.isArray(data) ? data : (data?.predictions || data?.history || []);
        const mapped: CalibrationRecord[] = raw.map((r: any) => ({
          id:        r.prediction_id || r.id || String(Math.random()),
          date:      r.created_at ? new Date(r.created_at).toLocaleDateString() : r.date || '—',
          domain:    r.domain || '—',
          predicted: typeof r.final_probability === 'number' ? Math.round(r.final_probability * 100) : 0,
          actual:    r.actual_outcome != null ? Math.round(r.actual_outcome * 100) : undefined,
          resolved:  r.actual_outcome != null,
          error:     r.actual_outcome != null
                       ? Math.abs(Math.round(r.final_probability * 100) - Math.round(r.actual_outcome * 100))
                       : undefined,
        }));
        setRecords(mapped);
      })
      .catch((err) => {
        const msg = err instanceof SambhavAPIError ? err.message : 'Failed to load history';
        setError(msg);
        // Use demo data so the page is still useful
        setRecords(DEMO_RECORDS);
      })
      .finally(() => setLoading(false));
  }, []);

  // Calibration metrics
  const resolved  = records.filter(r => r.resolved);
  const pending   = records.filter(r => !r.resolved);
  const brierScore = resolved.length > 0
    ? resolved.reduce((acc, r) => acc + Math.pow((r.predicted / 100) - (r.actual! / 100), 2), 0) / resolved.length
    : null;
  const mae = resolved.length > 0
    ? resolved.reduce((acc, r) => acc + (r.error || 0), 0) / resolved.length
    : null;
  const overallCalibration = resolved.length > 0
    ? 100 - (mae || 0)
    : null;

  // Calibration buckets (0-20, 20-40, …, 80-100)
  const buckets: BrierBucket[] = [
    { label: '0-20%',   range: [0, 20],   predicted: 0, actual: 0, count: 0 },
    { label: '20-40%',  range: [20, 40],  predicted: 0, actual: 0, count: 0 },
    { label: '40-60%',  range: [40, 60],  predicted: 0, actual: 0, count: 0 },
    { label: '60-80%',  range: [60, 80],  predicted: 0, actual: 0, count: 0 },
    { label: '80-100%', range: [80, 100], predicted: 0, actual: 0, count: 0 },
  ];
  resolved.forEach(r => {
    const b = buckets.find(bk => r.predicted >= bk.range[0] && r.predicted < bk.range[1])
            || buckets[buckets.length - 1];
    b.count++;
    b.predicted += r.predicted;
    b.actual    += r.actual || 0;
  });
  buckets.forEach(b => {
    if (b.count > 0) {
      b.predicted /= b.count;
      b.actual    /= b.count;
    }
  });

  // Submit actual outcome — save to backend + update UI
  const handleResolve = async (id: string) => {
    const val = parseFloat(newActual[id] || '');
    if (isNaN(val) || val < 0 || val > 100) return;
    setResolving(id);
    const record = records.find(r => r.id === id);
    if (record) {
      try {
        const BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';
        const userJson = localStorage.getItem('sambhav_user');
        const token = userJson ? JSON.parse(userJson)?.token || '' : '';
        await fetch(`${BASE}/evaluate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
          body: JSON.stringify({
            prediction_id:  record.id,
            predicted_prob: record.predicted / 100,
            actual_outcome: val >= 50, // treat ≥50% as "happened"
            domain:         record.domain,
          }),
        });
      } catch (e) { console.warn('Evaluate POST failed (likely auth):', e); }
    }
    // Optimistic UI update
    setRecords(prev => prev.map(r => r.id === id
      ? { ...r, resolved: true, actual: val, error: Math.abs(r.predicted - val) }
      : r
    ));
    setNewActual(prev => { const n = {...prev}; delete n[id]; return n; });
    setResolving(null);
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-20 pb-12 px-4">
        <div className="max-w-6xl mx-auto">

          {/* Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <Target className="w-6 h-6 text-primary" />
              <h1 className="text-2xl font-bold">Personal Calibration</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Track how well Sambhav's predictions match real outcomes. Enter actual results to measure calibration accuracy.
            </p>
          </motion.div>

          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-4 px-3 py-2 rounded-lg bg-warning/10 border border-warning/20 text-xs text-warning/80"
            >
              {error} — showing demo data
            </motion.div>
          )}

          {/* Metrics Row */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8"
          >
            <GlassCard variant="elevated" className="p-4">
              <div className="flex items-start justify-between mb-1">
                <span className="text-xs text-muted-foreground">Calibration Score</span>
                <TrendingUp className="w-4 h-4 text-primary" />
              </div>
              <div className="text-2xl font-bold text-primary">
                {overallCalibration != null ? `${overallCalibration.toFixed(1)}%` : '—'}
              </div>
              <div className="text-[10px] text-muted-foreground">{resolved.length} resolved</div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-4">
              <div className="flex items-start justify-between mb-1">
                <span className="text-xs text-muted-foreground">Brier Score</span>
                <BarChart3 className="w-4 h-4 text-secondary" />
              </div>
              <div className="text-2xl font-bold text-secondary">
                {brierScore != null ? brierScore.toFixed(3) : '—'}
              </div>
              <div className="text-[10px] text-muted-foreground">lower is better</div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-4">
              <div className="flex items-start justify-between mb-1">
                <span className="text-xs text-muted-foreground">Mean Abs. Error</span>
                <Target className="w-4 h-4 text-accent" />
              </div>
              <div className="text-2xl font-bold text-accent">
                {mae != null ? `${mae.toFixed(1)}%` : '—'}
              </div>
              <div className="text-[10px] text-muted-foreground">avg. prediction error</div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-4">
              <div className="flex items-start justify-between mb-1">
                <span className="text-xs text-muted-foreground">Pending</span>
                <Clock className="w-4 h-4 text-warning" />
              </div>
              <div className="text-2xl font-bold text-warning">{pending.length}</div>
              <div className="text-[10px] text-muted-foreground">awaiting resolution</div>
            </GlassCard>
          </motion.div>

          {/* Calibration Curve */}
          {resolved.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8"
            >
              <GlassCard variant="elevated" className="p-5">
                <h2 className="text-sm font-medium mb-4 text-muted-foreground">Calibration Curve (Predicted vs Actual)</h2>
                <div className="flex items-end gap-3 h-32">
                  {buckets.map((b, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                      <div className="w-full relative flex gap-0.5 items-end" style={{ height: '96px' }}>
                        {/* Predicted bar */}
                        <motion.div
                          initial={{ height: 0 }}
                          animate={{ height: `${b.predicted}%` }}
                          transition={{ delay: 0.3 + i * 0.1, duration: 0.6, ease: 'easeOut' }}
                          className="flex-1 rounded-t bg-primary/40 border border-primary/30"
                        />
                        {/* Actual bar */}
                        {b.count > 0 && (
                          <motion.div
                            initial={{ height: 0 }}
                            animate={{ height: `${b.actual}%` }}
                            transition={{ delay: 0.4 + i * 0.1, duration: 0.6, ease: 'easeOut' }}
                            className="flex-1 rounded-t bg-secondary/40 border border-secondary/30"
                          />
                        )}
                      </div>
                      <span className="text-[9px] text-muted-foreground">{b.label}</span>
                      {b.count > 0 && (
                        <span className="text-[9px] text-muted-foreground/60">n={b.count}</span>
                      )}
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-4 mt-3 text-[10px] text-muted-foreground">
                  <div className="flex items-center gap-1"><div className="w-3 h-2 rounded bg-primary/40 border border-primary/30" /> Predicted</div>
                  <div className="flex items-center gap-1"><div className="w-3 h-2 rounded bg-secondary/40 border border-secondary/30" /> Actual</div>
                </div>
              </GlassCard>
            </motion.div>
          )}

          {/* Records Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-sm font-medium text-muted-foreground mb-3">Prediction Records</h2>

            {loading ? (
              <GlassCard variant="elevated" className="p-8 flex items-center justify-center">
                <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
              </GlassCard>
            ) : records.length === 0 ? (
              <GlassCard variant="elevated" className="p-8 text-center">
                <p className="text-sm text-muted-foreground">No prediction records yet. Run predictions to build your calibration profile.</p>
              </GlassCard>
            ) : (
              <div className="space-y-2">
                {records.map((record, idx) => (
                  <motion.div
                    key={record.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + idx * 0.04 }}
                  >
                    <GlassCard variant="interactive" className="p-4">
                      <div className="grid grid-cols-2 md:grid-cols-6 gap-3 items-center">
                        <div>
                          <div className="text-[10px] text-muted-foreground mb-0.5">Date</div>
                          <div className="text-xs">{record.date}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground mb-0.5">Domain</div>
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-primary/10 text-primary border border-primary/20">
                            {record.domain}
                          </span>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground mb-0.5">Predicted</div>
                          <div className="text-sm font-bold text-primary">{record.predicted}%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground mb-0.5">Actual</div>
                          {record.resolved ? (
                            <div className="text-sm font-bold text-secondary">{record.actual}%</div>
                          ) : (
                            <span className="text-[10px] text-warning">Pending</span>
                          )}
                        </div>
                        <div>
                          <div className="text-[10px] text-muted-foreground mb-0.5">Error</div>
                          {record.resolved ? (
                            <div className={`text-sm font-bold ${(record.error || 0) <= 10 ? 'text-success' : (record.error || 0) <= 20 ? 'text-warning' : 'text-destructive'}`}>
                              ±{record.error}%
                            </div>
                          ) : (
                            <span className="text-[10px] text-muted-foreground">—</span>
                          )}
                        </div>
                        <div>
                          {record.resolved ? (
                            <div className="flex items-center gap-1">
                              {(record.error || 0) <= 15
                                ? <CheckCircle className="w-4 h-4 text-success" />
                                : <XCircle    className="w-4 h-4 text-destructive" />}
                              <span className="text-[10px] text-muted-foreground">
                                {(record.error || 0) <= 15 ? 'Well calibrated' : 'Overconfident'}
                              </span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1">
                              <input
                                type="number"
                                min={0} max={100}
                                placeholder="Actual %"
                                value={newActual[record.id] || ''}
                                onChange={e => setNewActual(prev => ({ ...prev, [record.id]: e.target.value }))}
                                className="w-20 px-2 py-1 text-[10px] bg-white/5 border border-white/10 rounded
                                         focus:outline-none focus:border-primary/50"
                              />
                              <motion.button
                                onClick={() => handleResolve(record.id)}
                                disabled={resolving === record.id || !newActual[record.id]}
                                className="px-2 py-1 text-[10px] rounded bg-primary/20 text-primary
                                         border border-primary/30 hover:bg-primary/30 disabled:opacity-40
                                         transition-colors"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                              >
                                {resolving === record.id ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Resolve'}
                              </motion.button>
                            </div>
                          )}
                        </div>
                      </div>
                    </GlassCard>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>

          {/* Calibration tips */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-8"
          >
            <GlassCard variant="elevated" className="p-5">
              <h3 className="text-sm font-medium mb-3">What is Calibration?</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                A perfectly calibrated system means that when it predicts 70%, the outcome actually occurs 70% of the time.
                A <strong>Brier Score</strong> below 0.10 is excellent. <strong>Mean Absolute Error</strong> below 10% indicates
                high-reliability predictions. Resolve pending predictions as outcomes become known to build your personal calibration profile.
              </p>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

// Demo records shown when backend is unavailable
const DEMO_RECORDS: CalibrationRecord[] = [
  { id: '1', date: '2026-03-28', domain: 'health',      predicted: 78, actual: 82, resolved: true,  error: 4  },
  { id: '2', date: '2026-03-27', domain: 'claim',       predicted: 65, actual: 58, resolved: true,  error: 7  },
  { id: '3', date: '2026-03-26', domain: 'fitness',     predicted: 82, resolved: false },
  { id: '4', date: '2026-03-25', domain: 'job_life',    predicted: 71, actual: 75, resolved: true,  error: 4  },
  { id: '5', date: '2026-03-24', domain: 'financial',   predicted: 89, actual: 91, resolved: true,  error: 2  },
  { id: '6', date: '2026-03-23', domain: 'student',     predicted: 54, resolved: false },
];
