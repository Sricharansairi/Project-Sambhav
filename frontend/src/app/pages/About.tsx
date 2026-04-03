import { motion } from 'motion/react';
import { Brain, Target, Lightbulb, Users, TrendingUp, Layers } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import { AnimatedIcon } from '../components/AnimatedIcon';
import logoImage from '../../assets/066d6bda782cfe271b2a192b0848783b83987f2e.png';

export function About() {
  const principles = [
    {
      icon: Brain,
      title: 'Probabilistic Thinking',
      description: 'Move beyond binary true/false into the realm of calibrated probabilities and uncertainty quantification.'
    },
    {
      icon: Target,
      title: 'Calibration Over Accuracy',
      description: 'Focus on long-term calibration: when you say 80%, it should happen 80% of the time.'
    },
    {
      icon: Lightbulb,
      title: 'Transparent Reasoning',
      description: 'Every prediction comes with confidence intervals, base rates, and explicit reasoning chains.'
    },
    {
      icon: Users,
      title: 'Multi-Perspective Analysis',
      description: 'Synthesize insights from multiple expert viewpoints and adversarial stress-testing.'
    },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-20 pb-12 px-4">
        <div className="max-w-5xl mx-auto">
          {/* Hero */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <motion.div
              className="w-28 h-28 mx-auto mb-6"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', bounce: 0.4 }}
            >
              <img src={logoImage} alt="Project Sambhav" className="w-full h-full object-contain" />
            </motion.div>

            <h1 className="text-4xl font-bold mb-3">
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Project Sambhav
              </span>
            </h1>

            <p className="text-xl text-muted-foreground mb-3">
              Uncertainty, Quantified
            </p>

            <p className="text-sm text-foreground/70 max-w-2xl mx-auto leading-relaxed">
              A research-grade framework for multi-modal probabilistic inference, designed to help 
              individuals and organizations make better decisions by explicitly quantifying uncertainty 
              and tracking calibration over time.
            </p>
          </motion.div>

          {/* Dual-Layer Architecture */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-16"
          >
            <GlassCard variant="elevated" className="p-8">
              <div className="flex items-center gap-3 mb-4">
                <Layers className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-bold">Dual-Layer Architecture</h2>
              </div>

              <div className="space-y-4 text-foreground/80 leading-relaxed">
                <p className="text-base">
                  Project Sambhav employs a sophisticated <strong className="text-primary">dual-layer architecture</strong> that 
                  combines the strengths of traditional Machine Learning with modern Large Language Models.
                </p>

                <div className="grid md:grid-cols-2 gap-4 mt-6">
                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <h3 className="text-base font-medium mb-2 text-primary">Layer 1: ML Engine</h3>
                    <p className="text-sm text-foreground/70">
                      Specialized machine learning models trained on domain-specific datasets provide quantitative 
                      predictions with statistical rigor. These models excel at pattern recognition and numerical forecasting.
                    </p>
                  </div>

                  <div className="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <h3 className="text-base font-medium mb-2 text-secondary">Layer 2: LLM Reasoning</h3>
                    <p className="text-sm text-foreground/70">
                      Large Language Models provide contextual understanding, qualitative analysis, and natural language 
                      explanations. They help interpret complex scenarios and communicate uncertainty effectively.
                    </p>
                  </div>
                </div>

                <p className="text-sm">
                  This hybrid approach ensures both <em>precision</em> (from ML models) and <em>comprehension</em> (from LLMs), 
                  delivering predictions that are not only accurate but also interpretable and actionable.
                </p>
              </div>
            </GlassCard>
          </motion.div>

          {/* Core Principles */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-16"
          >
            <h2 className="text-2xl font-bold text-center mb-8">Core Principles</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {principles.map((principle, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                >
                  <GlassCard variant="interactive" className="p-5 h-full" whileHover={{ y: -4 }}>
                    <div
                      className="w-12 h-12 rounded-lg flex items-center justify-center mb-3 bg-white/5 border border-white/10"
                    >
                      <AnimatedIcon
                        icon={principle.icon}
                        size={24}
                        color="#c0c0c0"
                        hoverEffect="glow"
                      />
                    </div>
                    <h3 className="text-base font-medium mb-2">{principle.title}</h3>
                    <p className="text-sm text-foreground/70 leading-relaxed">
                      {principle.description}
                    </p>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* How It Works */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="mb-16"
          >
            <h2 className="text-2xl font-bold text-center mb-8">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-4">
              {[
                {
                  step: '01',
                  title: 'Multi-Modal Input',
                  description: 'Provide text, images, documents, or engage in conversation. Choose from 12 operating modes.'
                },
                {
                  step: '02',
                  title: 'Probabilistic Analysis',
                  description: 'Our engine generates calibrated probabilities, confidence intervals, and uncertainty estimates.'
                },
                {
                  step: '03',
                  title: 'Track & Improve',
                  description: 'Submit actual outcomes and watch your calibration score improve over time.'
                }
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 + i * 0.1 }}
                >
                  <GlassCard variant="elevated" className="p-5 h-full">
                    <div
                      className="text-3xl font-bold mb-3 opacity-30 text-primary"
                    >
                      {item.step}
                    </div>
                    <h3 className="text-base font-medium mb-2 text-primary">
                      {item.title}
                    </h3>
                    <p className="text-sm text-foreground/70 leading-relaxed">
                      {item.description}
                    </p>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Achievements */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
            className="mb-16"
          >
            <h2 className="text-2xl font-bold text-center mb-8">Achievements</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { 
                  title: 'High Calibration',
                  desc: 'Consistently achieved <5% Brier score across cross-domain validations.',
                  icon: Target
                },
                {
                  title: 'Multi-Modal Inference',
                  desc: 'Successfully integrated tabular, clinical, linguistic, and behavioral data into a unified predictive layer.',
                  icon: Brain
                },
                {
                  title: 'Robust Safety Engine',
                  desc: 'Developed a 5-tier safety system including adversarial numeric detection and multi-layer PII redaction.',
                  icon: Layers
                }
              ].map((achievement, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.1 + i * 0.1 }}
                >
                  <GlassCard variant="elevated" className="p-6 h-full flex flex-col items-center text-center">
                    <div className="p-3 rounded-full bg-primary/10 text-primary mb-4">
                      <achievement.icon className="w-8 h-8" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2 text-foreground">{achievement.title}</h3>
                    <p className="text-sm text-foreground/70">{achievement.desc}</p>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
          </motion.div>


          {/* Research Foundation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
            className="mb-12"
          >
            <GlassCard variant="elevated" className="p-8">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-6 h-6 text-accent" />
                <h2 className="text-2xl font-bold">Research Foundation</h2>
              </div>

              <div className="space-y-3 text-foreground/80">
                <p className="text-sm">
                  Sambhav is built on decades of research in probabilistic forecasting, Bayesian reasoning, 
                  and calibration training. We draw inspiration from:
                </p>

                <ul className="space-y-2 ml-4">
                  <li className="flex items-start gap-2 text-sm">
                    <div className="w-1 h-1 bg-primary rounded-full mt-1.5 shrink-0" />
                    <span><strong>Philip Tetlock's</strong> work on superforecasting and calibration</span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <div className="w-1 h-1 bg-secondary rounded-full mt-1.5 shrink-0" />
                    <span><strong>Judea Pearl's</strong> causal inference framework</span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <div className="w-1 h-1 bg-accent rounded-full mt-1.5 shrink-0" />
                    <span><strong>Daniel Kahneman's</strong> research on cognitive biases and uncertainty</span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <div className="w-1 h-1 bg-success rounded-full mt-1.5 shrink-0" />
                    <span>Modern advances in <strong>multi-modal AI</strong> and large language models</span>
                  </li>
                </ul>
              </div>
            </GlassCard>
          </motion.div>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.7 }}
            className="text-center"
          >
            <GlassCard variant="elevated" className="p-8">
              <h3 className="text-2xl font-bold mb-3">
                Ready to Explore the Possible?
              </h3>
              <p className="text-base text-muted-foreground mb-6 max-w-xl mx-auto">
                Join researchers, analysts, and decision-makers who are embracing probabilistic thinking
              </p>
              <motion.a
                href="/dashboard"
                className="inline-block px-6 py-2.5 rounded-lg bg-primary text-black font-medium text-sm"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Start Predicting
              </motion.a>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}