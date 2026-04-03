import { motion, AnimatePresence } from 'motion/react';
import { useState, useEffect } from 'react';
import { Calendar, Filter, TrendingUp, TrendingDown, Target, Download, Trash2, Loader2 } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';

const API_BASE = 'http://localhost:8000';

interface PredictionRecord {
  id: string;
  prediction_id?: string;
  date: string;
  domain: string;
  mode: string;
  prediction: number;
  actual?: number;
  status: 'pending' | 'resolved';
  calibration?: number;
  question?: string;
}

export function History() {
  const [filter, setFilter] = useState('all');
  const [records, setRecords] = useState<PredictionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Fetch history from backend
  useEffect(() => {
    const fetchHistory = async () => {
      setIsLoading(true);
      try {
        const token = localStorage.getItem('sambhav_token') || '';
        const res = await fetch(`${API_BASE}/history`, {
          headers: { ...(token ? { 'Authorization': `Bearer ${token}` } : {}) },
        });
        const data = await res.json();
        if (data.predictions && Array.isArray(data.predictions)) {
          setRecords(data.predictions.map((p: any, i: number) => ({
            id: p.prediction_id || p.id || `${i}`,
            prediction_id: p.prediction_id || p.id,
            date: p.created_at ? new Date(p.created_at).toLocaleDateString() : new Date().toLocaleDateString(),
            domain: p.domain || 'Unknown',
            mode: p.mode || 'Guided',
            prediction: typeof p.final_probability === 'number' 
              ? (p.final_probability > 1 ? p.final_probability : Math.round(p.final_probability * 100))
              : (p.prediction || 50),
            actual: p.actual_outcome,
            status: p.actual_outcome !== undefined ? 'resolved' : 'pending',
            calibration: p.calibration_score,
            question: p.question,
          })));
        } else {
          setRecords([]);
        }
      } catch (e) {
        console.error('History fetch error:', e);
        setRecords([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // Delete a prediction
  const handleDelete = async (predId: string) => {
    setDeletingId(predId);
    try {
      const token = localStorage.getItem('sambhav_token') || '';
      await fetch(`${API_BASE}/history/${predId}`, {
        method: 'DELETE',
        headers: { ...(token ? { 'Authorization': `Bearer ${token}` } : {}) },
      });
      setRecords(prev => prev.filter(r => r.id !== predId));
    } catch (e) {
      console.error('Delete error:', e);
    } finally {
      setDeletingId(null);
    }
  };

  // Export single prediction
  const handleExportSingle = async (record: PredictionRecord) => {
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction_${record.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export full history
  const handleExportAll = () => {
    const blob = new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sambhav_history_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredRecords = records.filter(r => filter === 'all' || r.status === filter);

  const overallCalibration = records.filter(r => r.calibration).length > 0
    ? records.filter(r => r.calibration).reduce((acc, r) => acc + (r.calibration || 0), 0) / records.filter(r => r.calibration).length
    : 0;

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <h1 className="text-4xl font-bold mb-2">Prediction History</h1>
            <p className="text-muted-foreground">
              Track your predictions and calibration performance over time
            </p>
          </motion.div>

          {/* Calibration Scorecard */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8"
          >
            <GlassCard variant="elevated" glow glowColor="rgba(0, 255, 242, 0.2)" className="p-6">
              <div className="flex items-start justify-between mb-2">
                <div className="text-sm text-muted-foreground">Overall Calibration</div>
                <TrendingUp className="w-5 h-5 text-primary" />
              </div>
              <div className="text-3xl font-bold text-primary mb-1">
                {overallCalibration > 0 ? `${overallCalibration.toFixed(1)}%` : '—'}
              </div>
              <div className="text-xs text-muted-foreground">
                {records.filter(r => r.calibration).length} resolved predictions
              </div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-6">
              <div className="flex items-start justify-between mb-2">
                <div className="text-sm text-muted-foreground">Total Predictions</div>
                <Target className="w-5 h-5 text-secondary" />
              </div>
              <div className="text-3xl font-bold text-secondary mb-1">
                {records.length}
              </div>
              <div className="text-xs text-muted-foreground">
                {new Set(records.map(r => r.domain)).size} domains
              </div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-6">
              <div className="flex items-start justify-between mb-2">
                <div className="text-sm text-muted-foreground">Resolved</div>
                <Calendar className="w-5 h-5 text-accent" />
              </div>
              <div className="text-3xl font-bold text-accent mb-1">
                {records.filter(r => r.status === 'resolved').length}
              </div>
              <div className="text-xs text-muted-foreground">
                {records.filter(r => r.status === 'pending').length} pending
              </div>
            </GlassCard>

            <GlassCard variant="elevated" className="p-6">
              <div className="flex items-start justify-between mb-2">
                <div className="text-sm text-muted-foreground">Best Domain</div>
                <TrendingUp className="w-5 h-5 text-success" />
              </div>
              <div className="text-3xl font-bold text-success mb-1">
                {records.filter(r => r.calibration).length > 0 
                  ? `${Math.max(...records.filter(r => r.calibration).map(r => r.calibration || 0)).toFixed(1)}%`
                  : '—'}
              </div>
              <div className="text-xs text-muted-foreground">Top calibration score</div>
            </GlassCard>
          </motion.div>

          {/* Filters */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-wrap items-center gap-4 mb-6"
          >
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Filter by:</span>
            </div>
            {['all', 'resolved', 'pending'].map((f) => (
              <motion.button
                key={f}
                onClick={() => setFilter(f)}
                className={`
                  px-4 py-2 rounded-lg border transition-all capitalize
                  ${filter === f
                    ? 'bg-primary/20 border-primary text-primary'
                    : 'bg-white/5 border-white/10 text-muted-foreground hover:border-white/30'
                  }
                `}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {f}
              </motion.button>
            ))}
          </motion.div>

          {/* Records List */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-4"
          >
            {isLoading ? (
              <GlassCard variant="elevated" className="p-12 text-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">Loading history...</p>
              </GlassCard>
            ) : filteredRecords.length === 0 ? (
              <GlassCard variant="elevated" className="p-12 text-center">
                <Target className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
                <h3 className="text-lg font-medium mb-1">No predictions yet</h3>
                <p className="text-sm text-muted-foreground">
                  Start predicting to build your history and track calibration.
                </p>
              </GlassCard>
            ) : (
              filteredRecords.map((record, index) => (
                <motion.div
                  key={record.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + index * 0.05 }}
                >
                  <GlassCard variant="interactive" className="p-6" whileHover={{ y: -2 }}>
                    <div className="grid grid-cols-1 md:grid-cols-7 gap-4 items-center">
                      {/* Date */}
                      <div>
                        <div className="text-xs text-muted-foreground mb-1">Date</div>
                        <div className="text-sm font-medium">{record.date}</div>
                      </div>

                      {/* Domain & Mode */}
                      <div className="md:col-span-2">
                        <div className="text-xs text-muted-foreground mb-1">Analysis</div>
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-1 rounded text-xs bg-primary/20 text-primary border border-primary/30">
                            {record.domain}
                          </span>
                          <span className="text-xs text-muted-foreground">•</span>
                          <span className="text-sm">{record.mode}</span>
                        </div>
                        {record.question && (
                          <p className="text-[10px] text-muted-foreground mt-1 truncate max-w-xs">{record.question}</p>
                        )}
                      </div>

                      {/* Prediction vs Actual */}
                      <div>
                        <div className="text-xs text-muted-foreground mb-1">Predicted</div>
                        <div className="text-lg font-bold text-primary">
                          {typeof record.prediction === 'number' ? `${record.prediction.toFixed(1)}%` : '—'}
                        </div>
                      </div>

                      {record.status === 'resolved' ? (
                        <>
                          <div>
                            <div className="text-xs text-muted-foreground mb-1">Actual</div>
                            <div className="text-lg font-bold text-accent">
                              {record.actual?.toFixed(1)}%
                            </div>
                          </div>

                          <div>
                            <div className="text-xs text-muted-foreground mb-1">Calibration</div>
                            <div className="flex items-center gap-2">
                              <div className="text-lg font-bold text-success">
                                {record.calibration?.toFixed(1)}%
                              </div>
                              {record.calibration && record.calibration > 90 ? (
                                <TrendingUp className="w-4 h-4 text-success" />
                              ) : (
                                <TrendingDown className="w-4 h-4 text-warning" />
                              )}
                            </div>
                          </div>
                        </>
                      ) : (
                        <div className="md:col-span-2">
                          <span className="px-3 py-1 rounded-full text-xs bg-warning/20 text-warning border border-warning/30">
                            Awaiting Resolution
                          </span>
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex items-center gap-2 justify-end">
                        <motion.button
                          onClick={() => handleExportSingle(record)}
                          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          title="Export prediction"
                        >
                          <Download className="w-4 h-4 text-primary" />
                        </motion.button>
                        <motion.button
                          onClick={() => handleDelete(record.id)}
                          disabled={deletingId === record.id}
                          className="p-2 rounded-lg bg-white/5 hover:bg-destructive/10 border border-white/10 
                                   hover:border-destructive/30 transition-colors disabled:opacity-50"
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          title="Delete prediction"
                        >
                          {deletingId === record.id ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Trash2 className="w-4 h-4 text-destructive" />
                          )}
                        </motion.button>
                      </div>
                    </div>
                  </GlassCard>
                </motion.div>
              ))
            )}
          </motion.div>

          {/* Export Full History Button */}
          {records.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="mt-8 text-center"
            >
              <motion.button
                onClick={handleExportAll}
                className="px-6 py-3 rounded-lg bg-white/5 border border-white/10 
                         text-foreground flex items-center gap-2 mx-auto"
                whileHover={{ scale: 1.05, backgroundColor: 'rgba(255, 255, 255, 0.08)' }}
                whileTap={{ scale: 0.95 }}
              >
                <Download className="w-5 h-5" />
                <span>Export Full History</span>
              </motion.button>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}