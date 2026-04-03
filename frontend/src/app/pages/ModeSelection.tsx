import { motion } from 'motion/react';
import { useNavigate } from 'react-router';
import {
  Navigation as NavIcon,
  Type,
  Layers,
  MessageCircle,
  FileText,
  GitCompare,
  History,
  FlaskConical,
  Activity,
  ShieldAlert,
  Users,
  GitBranch
} from 'lucide-react';
import { BackgroundLogo }     from '../components/BackgroundLogo';
import { Navigation }         from '../components/Navigation';
import { GlassCard }          from '../components/GlassCard';
import { AnimatedIcon }       from '../components/AnimatedIcon';
import { MODES_SKIP_DOMAIN, MODE_DEFAULT_DOMAIN } from '../lib/constants';

const iconMap: Record<string, any> = {
  'navigation':   NavIcon,
  'text':         Type,
  'layers':       Layers,
  'message-circle': MessageCircle,
  'file-text':    FileText,
  'git-compare':  GitCompare,
  'history':      History,
  'flask':        FlaskConical,
  'activity':     Activity,
  'shield-alert': ShieldAlert,
  'users':        Users,
  'git-branch':   GitBranch,
};

const modes = [
  { id: 'guided',         name: 'Guided Mode',          description: 'Step-by-step chip parameter collection', icon: 'navigation',   badge: 'All 11 domains' },
  { id: 'free',           name: 'Free Inference',        description: 'Paste any text — no forms needed',       icon: 'text',         badge: 'LLM-primary' },
  { id: 'hybrid',         name: 'Hybrid Mode',           description: 'Text + image or video upload',           icon: 'layers',       badge: 'Vision enabled' },
  { id: 'conversational', name: 'Conversational',        description: 'Dialogue — one question at a time',      icon: 'message-circle', badge: 'Natural' },
  { id: 'document',       name: 'Document Analysis',     description: 'PDF, Word, or CSV file upload',          icon: 'file-text',    badge: '1M ctx' },
  { id: 'comparative',    name: 'Comparative Mode',      description: 'Side-by-side scenario comparison',       icon: 'git-compare',  badge: 'Multi-scenario' },
  { id: 'retrospective',  name: 'Retrospective',         description: 'Post-mortem analysis of past events',    icon: 'history',      badge: 'Hindsight' },
  { id: 'simulation',     name: 'Simulation Mode',       description: 'Hypothetical what-if scenarios',         icon: 'flask',        badge: 'Monte Carlo' },
  { id: 'monitoring',     name: 'Continuous Monitoring', description: 'Track probability over time',            icon: 'activity',     badge: 'Live updates' },
  { id: 'adversarial',    name: 'Adversarial Mode',      description: 'Stress-test with extreme inputs',        icon: 'shield-alert', badge: 'Fail-safe demo' },
  { id: 'expert',         name: 'Expert Consultation',   description: 'Multi-expert simulated perspectives',    icon: 'users',        badge: 'Multi-agent' },
  { id: 'whatif',         name: 'What-If Story',         description: 'Branching scenario probability trees',   icon: 'git-branch',   badge: 'Scenario tree' },
];

export function ModeSelection() {
  const navigate = useNavigate();

  const handleModeSelect = (modeId: string) => {
    if (MODES_SKIP_DOMAIN.includes(modeId)) {
      /**
       * FIX (BUG 3): All non-guided modes skip domain selection.
       * Route directly to /prediction with a sensible default domain.
       *
       * Before this fix: only 'free' and 'document' skipped.
       * 'conversational', 'hybrid', 'simulation' etc. still showed
       * the 11-domain grid — which is only meaningful in guided mode
       * where the chip modal collects domain-specific parameters.
       *
       * Non-guided modes don't use the chip parameter modal,
       * so domain selection has no purpose for them.
       */
      const defaultDomain = MODE_DEFAULT_DOMAIN[modeId] || 'student';
      navigate(`/prediction?mode=${modeId}&domain=${defaultDomain}`);
    } else {
      // Only guided mode reaches the domain selection screen
      navigate(`/domain-selection?mode=${modeId}`);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      <BackgroundLogo />
      <Navigation />

      <div className="relative z-10 pt-20 pb-12 px-4">
        <div className="max-w-6xl mx-auto">

          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <h1 className="text-3xl font-bold mb-2">Select Operating Mode</h1>
            <p className="text-sm text-muted-foreground">
              Choose how you want to interact with the prediction engine
            </p>
            <p className="text-xs text-muted-foreground/60 mt-1.5">
              Only <span className="text-primary font-medium">Guided Mode</span> uses domain selection + chip parameters.
              All other modes go straight to prediction.
            </p>
          </motion.div>

          {/* Modes Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {modes.map((mode, index) => {
              const Icon      = iconMap[mode.icon];
              const isGuided  = mode.id === 'guided';

              return (
                <motion.div
                  key={mode.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <GlassCard
                    variant="interactive"
                    className={`p-5 cursor-pointer h-full ${isGuided ? 'ring-1 ring-primary/30' : ''}`}
                    whileHover={{ y: -4 }}
                    onClick={() => handleModeSelect(mode.id)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg shrink-0 ${
                        isGuided ? 'bg-primary/10 border border-primary/30' : 'bg-white/5 border border-white/10'
                      }`}>
                        <AnimatedIcon
                          icon={Icon}
                          size={20}
                          color={isGuided ? '#C2CD93' : '#c0c0c0'}
                          hoverEffect="pulse"
                        />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-base font-medium">{mode.name}</h3>
                          {isGuided && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                              DOMAIN SELECTION
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">{mode.description}</p>
                        {mode.badge && (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-muted-foreground">
                            {mode.badge}
                          </span>
                        )}
                      </div>
                    </div>
                  </GlassCard>
                </motion.div>
              );
            })}
          </div>

        </div>
      </div>
    </div>
  );
}