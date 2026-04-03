import { motion } from 'motion/react';
import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { CheckCircle, TrendingUp, Target } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';

export function Evaluation() {
  const [selectedPrediction] = useState({
    id: '1',
    date: '2026-03-28',
    question: 'Will the new product feature increase user engagement by more than 50%?',
    prediction: 78.4,
    submitted: false,
  });

  const [actualOutcome, setActualOutcome] = useState('');

  const calibrationData = [
    { month: 'Oct', calibration: 85.2 },
    { month: 'Nov', calibration: 87.5 },
    { month: 'Dec', calibration: 89.1 },
    { month: 'Jan', calibration: 91.8 },
    { month: 'Feb', calibration: 92.3 },
    { month: 'Mar', calibration: 94.1 },
  ];

  const accuracyData = [
    { range: '0-10%', predicted: 12, actual: 10 },
    { range: '10-20%', predicted: 18, actual: 19 },
    { range: '20-30%', predicted: 25, actual: 27 },
    { range: '30-40%', predicted: 32, actual: 30 },
    { range: '40-50%', predicted: 45, actual: 44 },
    { range: '50-60%', predicted: 52, actual: 55 },
    { range: '60-70%', predicted: 68, actual: 65 },
    { range: '70-80%', predicted: 75, actual: 77 },
    { range: '80-90%', predicted: 85, actual: 82 },
    { range: '90-100%', predicted: 95, actual: 96 },
  ];

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
            <h1 className="text-4xl font-bold mb-2">Evaluation Center</h1>
            <p className="text-muted-foreground">
              Submit actual outcomes and track your calibration trajectory
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-2 gap-6 mb-8">
            {/* Submit Outcome */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <GlassCard variant="elevated" className="p-6 h-full">
                <h3 className="text-lg font-medium mb-4">Submit Actual Outcome</h3>

                <div className="mb-6">
                  <div className="text-sm text-muted-foreground mb-2">Original Prediction</div>
                  <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                    <div className="text-sm mb-2">{selectedPrediction.question}</div>
                    <div className="flex items-center gap-4">
                      <div>
                        <div className="text-xs text-muted-foreground">Predicted</div>
                        <div className="text-2xl font-bold text-primary">
                          {selectedPrediction.prediction}%
                        </div>
                      </div>
                      <div className="text-muted-foreground">•</div>
                      <div>
                        <div className="text-xs text-muted-foreground">Date</div>
                        <div className="text-sm">{selectedPrediction.date}</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mb-6">
                  <label className="block text-sm mb-2">What was the actual outcome?</label>
                  <textarea
                    value={actualOutcome}
                    onChange={(e) => setActualOutcome(e.target.value)}
                    placeholder="Describe what actually happened..."
                    className="w-full h-32 px-4 py-3 bg-white/5 border border-white/10 rounded-lg
                             focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent/50
                             transition-all resize-none placeholder:text-muted-foreground/50"
                  />
                </div>

                <div className="mb-6">
                  <label className="block text-sm mb-2">Actual Result (%)</label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    placeholder="Enter percentage (0-100)"
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg
                             focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent/50
                             transition-all placeholder:text-muted-foreground/50"
                  />
                </div>

                <motion.button
                  className="w-full px-6 py-3 rounded-lg bg-accent text-black font-medium
                           flex items-center justify-center gap-2"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <CheckCircle className="w-5 h-5" />
                  <span>Submit Outcome</span>
                </motion.button>
              </GlassCard>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="space-y-4"
            >
              <GlassCard variant="elevated" glow glowColor="rgba(0, 255, 136, 0.2)" className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-2">
                      Current Calibration Score
                    </div>
                    <div className="text-4xl font-bold text-success">94.1%</div>
                  </div>
                  <TrendingUp className="w-8 h-8 text-success" />
                </div>
                <div className="text-sm text-success">
                  +1.8% from last month
                </div>
              </GlassCard>

              <GlassCard variant="elevated" className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-2">
                      Predictions Evaluated
                    </div>
                    <div className="text-4xl font-bold text-primary">127</div>
                  </div>
                  <Target className="w-8 h-8 text-primary" />
                </div>
                <div className="text-sm text-muted-foreground">
                  18 pending evaluation
                </div>
              </GlassCard>

              <GlassCard variant="elevated" className="p-6">
                <div className="text-sm text-muted-foreground mb-3">
                  Confidence Interval Accuracy
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-2xl font-bold text-primary">90%</div>
                    <div className="text-xs text-muted-foreground">Coverage</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-secondary">7.2</div>
                    <div className="text-xs text-muted-foreground">Avg Width</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-accent">96%</div>
                    <div className="text-xs text-muted-foreground">Precision</div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Calibration Trajectory */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-8"
          >
            <GlassCard variant="elevated" className="p-6">
              <h3 className="text-lg font-medium mb-6">Calibration Trajectory</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={calibrationData}>
                  <defs>
                    <linearGradient id="calibrationGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="month" 
                    stroke="#a0a0a0" 
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    stroke="#a0a0a0" 
                    style={{ fontSize: '12px' }}
                    domain={[80, 100]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(10, 10, 10, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#e8e8e8',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="calibration"
                    stroke="#00ff88"
                    strokeWidth={3}
                    fill="url(#calibrationGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </GlassCard>
          </motion.div>

          {/* Calibration Plot */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <GlassCard variant="elevated" className="p-6">
              <h3 className="text-lg font-medium mb-6">Calibration Plot</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Compare your predicted probabilities with actual outcomes. Perfect calibration follows the diagonal line.
              </p>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="range" 
                    stroke="#a0a0a0" 
                    style={{ fontSize: '11px' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    stroke="#a0a0a0" 
                    style={{ fontSize: '12px' }}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(10, 10, 10, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#e8e8e8',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#00fff2"
                    strokeWidth={3}
                    dot={{ fill: '#00fff2', r: 4 }}
                    name="Your Predictions"
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#9d4eff"
                    strokeWidth={3}
                    dot={{ fill: '#9d4eff', r: 4 }}
                    name="Actual Outcomes"
                  />
                </LineChart>
              </ResponsiveContainer>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}