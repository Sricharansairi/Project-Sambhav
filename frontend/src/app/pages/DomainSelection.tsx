import { motion } from 'motion/react';
import { useNavigate, useSearchParams } from 'react-router';
import { BackgroundLogo } from '../components/BackgroundLogo';
import { Navigation } from '../components/Navigation';
import { GlassCard } from '../components/GlassCard';
import {
  Brain, FileCheck, Shield, Heart, CreditCard, Dumbbell,
  Users as UsersIcon, GraduationCap, Stethoscope, BookOpen, Eye
} from 'lucide-react';
import { DOMAINS } from '../lib/constants';

const domainIcons: Record<string, any> = {
  pragma:           Shield,
  sarvagna:         Brain,
  claim:            FileCheck,
  behavioral:       Eye,
  mental_health:    Heart,
  financial:        CreditCard,
  fitness:          Dumbbell,
  job_life:         UsersIcon,
  high_school:      GraduationCap,
  health:           Stethoscope,
  student:          BookOpen,
};

export function DomainSelection() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const selectedMode = searchParams.get('mode') || 'guided';

  const handleDomainSelect = (domainId: string) => {
    navigate(`/prediction?mode=${selectedMode}&domain=${domainId}`);
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
            <h1 className="text-3xl font-bold mb-2">Select Domain</h1>
            <p className="text-sm text-muted-foreground">
              Choose the specialised domain for your prediction
            </p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Mode: <span className="text-primary">{selectedMode}</span>
            </p>
          </motion.div>

          {/* Domain Grid — 11 domains */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {DOMAINS.map((domain, index) => {
              const Icon = domainIcons[domain.id] || Brain;
              return (
                <motion.div
                  key={domain.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <GlassCard
                    variant="interactive"
                    className="p-4 cursor-pointer h-full"
                    whileHover={{ y: -4, scale: 1.02 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => handleDomainSelect(domain.id)}
                  >
                    <div className="flex flex-col items-center text-center gap-2">
                      <div className="p-2.5 rounded-lg bg-white/5 border border-white/10">
                        <Icon className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="text-sm font-medium">{domain.name}</h3>
                        <p className="text-[11px] text-muted-foreground mt-0.5">{domain.subtitle}</p>
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
