import { motion } from 'motion/react';
import { User, Lock, Download, Trash2, LogOut, Shield, ChevronRight, History, BarChart3 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import { LoadingAnimation } from '../components/LoadingAnimation';
import { auth, getHistory } from '../lib/api';

export function Profile() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const data = await auth.getMe();
        setUser(data.user);
        
        // Fetch history immediately for profile
        setLoadingHistory(true);
        const histData = await getHistory();
        if (histData.success) {
          setHistory(histData.predictions || []);
        }
      } catch (err) {
        console.error('Failed to fetch profile', err);
        auth.logout();
        navigate('/auth');
      } finally {
        setLoading(false);
        setLoadingHistory(false);
      }
    };
    fetchProfile();
  }, [navigate]);

  const filteredHistory = history.filter(item => {
    if (filter === 'all') return true;
    if (filter === 'resolved') return item.actual_outcome !== undefined;
    if (filter === 'pending') return item.actual_outcome === undefined;
    return true;
  });

  const handleDeletePrediction = async (id: string) => {
    if (!confirm('Delete this prediction from your history?')) return;
    try {
      // Import and call delete history from api
      const { deleteHistory } = await import('../lib/api');
      await deleteHistory(id);
      setHistory(prev => prev.filter(p => p.prediction_id !== id));
    } catch (err) {
      console.error(err);
      alert('Failed to delete prediction');
    }
  };

  const handleLogout = () => {
    auth.logout();
    localStorage.removeItem('sambhav_user');
    navigate('/auth');
  };

  const handleExportData = async () => {
    setIsExporting(true);
    try {
      const histData = await getHistory();
      if (!histData.success) throw new Error('Failed to fetch history');
      
      const blob = new Blob([JSON.stringify(histData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sambhav_data_export_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed', err);
      alert('Failed to export data');
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (!confirm('Are you sure you want to delete your account? All your prediction history will be permanently removed. This action cannot be undone.')) return;
    
    try {
      await auth.deleteAccount();
      auth.logout();
      localStorage.removeItem('sambhav_user');
      navigate('/');
    } catch (err) {
      console.error(err);
      alert('Failed to delete account');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background relative flex items-center justify-center">
        <BackgroundLogo />
        <Navigation />
        <div className="flex flex-col items-center gap-4">
          <LoadingAnimation />
          <p className="text-muted-foreground animate-pulse">Loading profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-20 pb-12 px-4">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between mb-6"
          >
            <div>
              <h1 className="text-2xl font-bold mb-1">Profile & Settings</h1>
              <p className="text-sm text-muted-foreground">
                Manage your account and preferences
              </p>
            </div>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-sm rounded-lg border border-white/10 hover:bg-white/5 flex items-center gap-2 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-4">
            {/* Left Column */}
            <div className="space-y-4">
              {/* Profile Card */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <GlassCard variant="elevated" className="p-5">
                  <div className="text-center">
                    <div className="w-20 h-20 mx-auto mb-3 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                      <User className="w-10 h-10 text-black" />
                    </div>
                    <h3 className="text-lg font-bold mb-1">{user?.email?.split('@')[0] || 'User'}</h3>
                    <p className="text-xs text-muted-foreground mb-3">{user?.email || 'user@example.com'}</p>
                    <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-primary/20 text-primary text-xs">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full" />
                      Pro Plan
                    </div>
                  </div>

                    <div className="mt-5 pt-5 border-t border-white/10 space-y-4">
                      <div className="flex justify-between items-center text-xs">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <BarChart3 className="w-3.5 h-3.5" />
                          <span>Predictions Made</span>
                        </div>
                        <span className="font-bold text-primary">{history.length}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">User ID</span>
                        <span className="font-mono text-[10px]">{user?.id?.substring(0, 8)}...</span>
                      </div>
                    </div>
                </GlassCard>
              </motion.div>

              {/* Quick Actions */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <GlassCard variant="elevated" className="p-5">
                  <h3 className="text-base font-medium mb-3">Data Management</h3>
                  <div className="space-y-2">
                    <motion.button
                      className="w-full px-3 py-2.5 text-sm rounded-lg bg-white/5 border border-white/10
                               hover:bg-white/10 transition-colors text-left flex items-center justify-between"
                      onClick={handleExportData}
                      disabled={isExporting}
                      whileHover={{ x: 4 }}
                    >
                      <div className="flex items-center gap-2">
                        <Download className="w-4 h-4 text-primary" />
                        <span>Export My Data</span>
                      </div>
                      <ChevronRight className="w-3 h-3 text-muted-foreground" />
                    </motion.button>

                    <motion.button
                      className="w-full px-3 py-2.5 text-sm rounded-lg bg-white/5 border border-white/10
                               hover:bg-destructive/10 hover:border-destructive/30 transition-colors 
                               text-left flex items-center gap-2 text-destructive mt-4"
                      onClick={handleDeleteAccount}
                      whileHover={{ x: 4 }}
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>Delete Account</span>
                    </motion.button>
                  </div>
                </GlassCard>
              </motion.div>
            </div>

            {/* Right Column */}
            <div className="lg:col-span-2 space-y-4">
              <GlassCard variant="elevated" className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-primary/20">
                        <History className="w-4 h-4 text-primary" />
                      </div>
                      <h3 className="text-base font-medium">Prediction History</h3>
                    </div>
                    <div className="flex items-center gap-1">
                      {['all', 'resolved', 'pending'].map(f => (
                        <button
                          key={f}
                          onClick={() => setFilter(f)}
                          className={`px-2 py-1 text-[10px] rounded-md transition-all capitalize ${
                            filter === f ? 'bg-primary/20 text-primary border border-primary/30' : 'text-muted-foreground hover:bg-white/5'
                          }`}
                        >
                          {f}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                    {loadingHistory ? (
                      <div className="py-8 text-center text-xs text-muted-foreground animate-pulse">
                        Fetching predictions...
                      </div>
                    ) : filteredHistory.length > 0 ? (
                      filteredHistory.map((item, i) => (
                        <div key={i} className="p-3 bg-white/5 border border-white/10 rounded-lg flex items-center justify-between group hover:bg-white/10 transition-all">
                          <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                              item.actual_outcome !== undefined ? 'bg-success/20 text-success' : 'bg-primary/20 text-primary'
                            }`}>
                              {Math.round((item.final_probability || 0.5) * 100)}%
                            </div>
                            <div className="flex flex-col gap-0.5">
                              <span className="text-xs font-medium truncate max-w-[200px]">{item.question || 'Untitled Prediction'}</span>
                              <span className="text-[10px] text-muted-foreground">{item.domain} • {new Date(item.created_at).toLocaleDateString()}</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={() => navigate(`/prediction?id=${item.prediction_id}`)}
                              className="p-1.5 hover:bg-white/10 rounded-md text-muted-foreground hover:text-primary transition-colors"
                            >
                              <ChevronRight className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleDeletePrediction(item.prediction_id)}
                              className="p-1.5 hover:bg-destructive/10 rounded-md text-muted-foreground hover:text-destructive transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="py-12 text-center text-xs text-muted-foreground">
                        No {filter !== 'all' ? filter : ''} predictions found.
                      </div>
                    )}
                  </div>
                </GlassCard>

                {/* Privacy Controls */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <GlassCard variant="elevated" className="p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="p-1.5 rounded-lg bg-secondary/20">
                      <Lock className="w-4 h-4 text-secondary" />
                    </div>
                    <div>
                      <h3 className="text-base font-medium">Privacy Controls</h3>
                      <p className="text-xs text-muted-foreground">
                        Manage data collection and sharing preferences
                      </p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {[
                      { label: 'Store prediction history', description: 'Save all predictions for calibration tracking' },
                      { label: 'Anonymous analytics', description: 'Help improve Sambhav with usage data' },
                    ].map((control, i) => (
                      <div key={i} className="flex items-start justify-between p-3 bg-white/5 border border-white/10 rounded-lg">
                        <div className="flex-1">
                          <div className="text-sm font-medium mb-0.5">{control.label}</div>
                          <div className="text-xs text-muted-foreground">{control.description}</div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer ml-3">
                          <input type="checkbox" defaultChecked={i < 2} className="sr-only peer" />
                          <div className="w-9 h-5 bg-white/20 peer-focus:outline-none rounded-full peer 
                                        peer-checked:after:translate-x-full peer-checked:after:border-white 
                                        after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                                        after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all 
                                        peer-checked:bg-primary"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}