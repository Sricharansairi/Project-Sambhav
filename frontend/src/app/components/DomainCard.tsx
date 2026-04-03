import { motion } from 'motion/react';
import { Brain, Target, FileCheck, Users, Heart, DollarSign, Dumbbell, Briefcase, GraduationCap, Activity, Home } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { AnimatedIcon } from './AnimatedIcon';

const domainIcons = {
  pragma: Brain,
  sarvagna: Target,
  claim: FileCheck,
  behaviour: Users,
  mental: Heart,
  loan: DollarSign,
  fitness: Dumbbell,
  hr: Briefcase,
  education: GraduationCap,
  medico: Activity,
  daily: Home,
};

interface Domain {
  id: string;
  name: string;
  subtitle: string;
  color: string;
}

interface DomainCardProps {
  domain: Domain;
  selected?: boolean;
  onClick?: () => void;
}

export function DomainCard({ domain, selected = false, onClick }: DomainCardProps) {
  const Icon = domainIcons[domain.id as keyof typeof domainIcons] || Target;

  return (
    <GlassCard
      variant="interactive"
      glow={selected}
      glowColor={`${domain.color}40`}
      className={`p-6 cursor-pointer ${selected ? 'border-2' : 'border'}`}
      style={{
        borderColor: selected ? domain.color : 'rgba(255, 255, 255, 0.1)',
      }}
      onClick={onClick}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-start gap-4">
        <div
          className="p-3 rounded-xl shrink-0"
          style={{
            backgroundColor: `${domain.color}15`,
            border: `1px solid ${domain.color}30`,
          }}
        >
          <AnimatedIcon icon={Icon} size={28} color={domain.color} hoverEffect="twitch" />
        </div>

        <div className="flex-1 min-w-0">
          <h3
            className="text-lg font-medium mb-1"
            style={{ color: selected ? domain.color : 'inherit' }}
          >
            {domain.name}
          </h3>
          <p className="text-sm text-muted-foreground">{domain.subtitle}</p>
        </div>

        {selected && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: domain.color }}
          />
        )}
      </div>
    </GlassCard>
  );
}
