import { motion } from 'motion/react';
import { Link } from 'react-router';
import { ArrowRight, Brain, TrendingUp, Shield, Zap } from 'lucide-react';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import { AnimatedIcon } from '../components/AnimatedIcon';
import logoImage from '../../assets/066d6bda782cfe271b2a192b0848783b83987f2e.png';

export function Home() {
  const features = [
    {
      icon: Brain,
      title: 'Multi-Modal Intelligence',
      description: 'Process text, images, video, and documents with unified probabilistic reasoning'
    },
    {
      icon: TrendingUp,
      title: 'Calibrated Uncertainty',
      description: 'Get precise confidence intervals and uncertainty quantification for every prediction'
    },
    {
      icon: Shield,
      title: 'Adversarial Testing',
      description: 'Stress-test your predictions with built-in adversarial mode and edge case analysis'
    },
    {
      icon: Zap,
      title: 'Real-Time Monitoring',
      description: 'Track prediction accuracy over time with continuous calibration updates'
    },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />
      
      <div className="relative z-10 pt-20 pb-12 px-4">
        <div className="max-w-5xl mx-auto">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-center mb-16"
          >
            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="w-32 h-32 mx-auto mb-6"
            >
              <img
                src={logoImage}
                alt="Project Sambhav"
                className="w-full h-full object-contain"
              />
            </motion.div>

            {/* Main Headline */}
            <h1 className="text-5xl md:text-6xl font-bold mb-2">
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Project Sambhav
              </span>
            </h1>

            <motion.p
              className="text-lg md:text-xl text-muted-foreground mb-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35 }}
            >
              A Multi-Modal Probabilistic Inference
            </motion.p>

            <motion.p
              className="text-xl md:text-2xl text-muted-foreground mb-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Uncertainty, Quantified
            </motion.p>

            <motion.p
              className="text-base text-foreground/60 max-w-xl mx-auto mb-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              A research-grade AI framework that transforms complex predictions into calibrated probabilities. 
              Make better decisions with transparent uncertainty quantification.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              className="flex flex-col sm:flex-row items-center justify-center gap-3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Link to="/dashboard">
                <motion.button
                  className="px-6 py-2.5 rounded-lg bg-primary text-black flex items-center gap-2 group text-sm"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="font-medium">Start Predicting</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </motion.button>
              </Link>

              <Link to="/about">
                <motion.button
                  className="px-6 py-2.5 rounded-lg bg-white/5 border border-white/20 text-foreground flex items-center gap-2 text-sm"
                  whileHover={{ scale: 1.05, backgroundColor: 'rgba(255, 255, 255, 0.08)' }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="font-medium">Learn More</span>
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mb-12"
          >
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">
                Premium Research Framework
              </h2>
              <p className="text-sm text-muted-foreground">
                Built for professionals who need precision and transparency
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {features.map((feature, index) => (
                <GlassCard
                  key={index}
                  variant="interactive"
                  className="p-5"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.9 + index * 0.1 }}
                  whileHover={{ y: -4 }}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className="p-2 rounded-lg shrink-0 bg-white/5 border border-white/10"
                    >
                      <AnimatedIcon
                        icon={feature.icon}
                        size={20}
                        color="#c0c0c0"
                        hoverEffect="glow"
                      />
                    </div>
                    <div>
                      <h3 className="text-base font-medium mb-1.5">{feature.title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </GlassCard>
              ))}
            </div>
          </motion.div>

          {/* Final CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.3 }}
            className="text-center"
          >
            <GlassCard
              variant="elevated"
              className="p-8 max-w-2xl mx-auto"
            >
              <h3 className="text-2xl font-bold mb-3">
                Ready to Quantify Uncertainty?
              </h3>
              <p className="text-base text-muted-foreground mb-6">
                Join researchers, analysts, and decision-makers who trust Sambhav for critical predictions
              </p>
              <Link to="/dashboard">
                <motion.button
                  className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-primary to-secondary text-black font-medium text-sm"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Get Started Free
                </motion.button>
              </Link>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}