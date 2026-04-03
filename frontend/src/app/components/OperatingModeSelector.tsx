import { motion } from 'motion/react';
import { 
  Navigation, Text, Layers, MessageCircle, FileText, 
  GitCompare, History, FlaskConical, Activity, ShieldAlert, 
  Users, GitBranch, LucideIcon 
} from 'lucide-react';
import { GlassCard } from './GlassCard';
import { AnimatedIcon } from './AnimatedIcon';

const iconMap: Record<string, LucideIcon> = {
  navigation: Navigation,
  text: Text,
  layers: Layers,
  'message-circle': MessageCircle,
  'file-text': FileText,
  'git-compare': GitCompare,
  history: History,
  flask: FlaskConical,
  activity: Activity,
  'shield-alert': ShieldAlert,
  users: Users,
  'git-branch': GitBranch,
};

interface OperatingMode {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
}

interface OperatingModeSelectorProps {
  modes: OperatingMode[];
  selectedMode: string;
  onSelectMode: (modeId: string) => void;
}

export function OperatingModeSelector({ modes, selectedMode, onSelectMode }: OperatingModeSelectorProps) {
  const colorMap: Record<string, string> = {
    primary: '#00fff2',
    secondary: '#9d4eff',
    accent: '#00d9ff',
    success: '#00ff88',
    warning: '#ff6b35',
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {modes.map((mode, index) => {
        const Icon = iconMap[mode.icon] || Text;
        const color = colorMap[mode.color] || '#00fff2';
        const isSelected = selectedMode === mode.id;

        return (
          <motion.div
            key={mode.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <GlassCard
              variant="interactive"
              glow={isSelected}
              glowColor={`${color}40`}
              className={`
                p-4 h-full cursor-pointer
                ${isSelected ? 'border-2' : 'border'}
              `}
              style={{
                borderColor: isSelected ? color : 'rgba(255, 255, 255, 0.1)',
              }}
              onClick={() => onSelectMode(mode.id)}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex flex-col gap-3">
                <div className="flex items-start justify-between">
                  <div 
                    className="p-2 rounded-lg"
                    style={{
                      backgroundColor: `${color}15`,
                      border: `1px solid ${color}30`,
                    }}
                  >
                    <AnimatedIcon 
                      icon={Icon} 
                      size={24} 
                      color={color}
                      hoverEffect="twitch"
                    />
                  </div>
                  
                  {isSelected && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                  )}
                </div>

                <div>
                  <h3 
                    className="font-medium mb-1"
                    style={{ color: isSelected ? color : 'inherit' }}
                  >
                    {mode.name}
                  </h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {mode.description}
                  </p>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        );
      })}
    </div>
  );
}