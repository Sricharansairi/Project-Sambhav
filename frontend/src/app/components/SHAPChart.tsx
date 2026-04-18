import { motion } from 'motion/react';

interface SHAPFeature {
  name: string;
  value: number; // -0.18 to +0.18 normalized
}

interface SHAPChartProps {
  features: SHAPFeature[];
  delay?: number;
}

export function SHAPChart({ features, delay = 0 }: SHAPChartProps) {
  const maxAbsValue = 0.18;
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-2">
        <h5 className="text-[11px] font-medium text-muted-foreground">Feature Contributions (SHAP)</h5>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-sm bg-[#00fff2]/60" />
            Positive
          </span>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-sm bg-[#ff6b6b]/60" />
            Negative
          </span>
        </div>
      </div>
      
      {features.map((feature, idx) => {
        const percentage = (Math.abs(feature.value) / maxAbsValue) * 100;
        const isPositive = feature.value >= 0;
        
        return (
          <motion.div
            key={feature.name}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delay + idx * 0.1, duration: 0.4 }}
            className="space-y-1"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-foreground/80">{feature.name}</span>
              <span className={`text-[10px] font-medium ${isPositive ? 'text-[#00fff2]' : 'text-[#ff6b6b]'}`}>
                {isPositive ? '+' : ''}{feature.value.toFixed(3)}
              </span>
            </div>
            
            {/* Horizontal Bar centered at 0 */}
            <div className="flex items-center gap-1">
              {/* Left side (negative) */}
              <div className="flex-1 flex justify-end">
                {!isPositive && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ 
                      delay: delay + idx * 0.1 + 0.2,
                      duration: 0.8,
                      ease: [0.25, 0.1, 0.25, 1]
                    }}
                    className="h-1.5 bg-[#ff6b6b]/60 rounded-l-full"
                  />
                )}
              </div>
              
              {/* Center line */}
              <div className="w-0.5 h-3 bg-white/20" />
              
              {/* Right side (positive) */}
              <div className="flex-1">
                {isPositive && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ 
                      delay: delay + idx * 0.1 + 0.2,
                      duration: 0.8,
                      ease: [0.25, 0.1, 0.25, 1]
                    }}
                    className="h-1.5 bg-[#00fff2]/60 rounded-r-full"
                  />
                )}
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
