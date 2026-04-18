import { motion } from 'motion/react';
import { useState } from 'react';
import * as Slider from '@radix-ui/react-slider';

interface Parameter {
  id: string;
  name: string;
  current: number;
  min: number;
  max: number;
  unit?: string;
}

interface CounterfactualSlidersProps {
  parameters: Parameter[];
  onParameterChange: (id: string, value: number) => void;
  delay?: number;
}

export function CounterfactualSliders({ parameters, onParameterChange, delay = 0 }: CounterfactualSlidersProps) {
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(parameters.map(p => [p.id, p.current]))
  );

  const handleChange = (id: string, newValue: number) => {
    setValues(prev => ({ ...prev, [id]: newValue }));
    onParameterChange(id, newValue);
  };

  const calculateDelta = (current: number, newValue: number) => {
    const delta = ((newValue - current) / current) * 100;
    return delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1);
  };

  return (
    <div className="space-y-3">
      <h5 className="text-[11px] font-medium text-muted-foreground">What-If Analysis (Counterfactuals)</h5>
      
      {parameters.map((param, idx) => {
        const currentValue = values[param.id];
        const delta = calculateDelta(param.current, currentValue);
        const hasChanged = currentValue !== param.current;
        
        return (
          <motion.div
            key={param.id}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: delay + idx * 0.1 }}
            className="space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-foreground/80">{param.name}</span>
              <div className="flex items-center gap-2 text-[10px]">
                <span className="text-muted-foreground">
                  {param.current}{param.unit}
                </span>
                <span className="text-white/30">→</span>
                <span className={hasChanged ? 'text-[#00fff2] font-medium' : 'text-muted-foreground'}>
                  {currentValue}{param.unit}
                </span>
                {hasChanged && (
                  <motion.span
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-[#ffb7c5] font-medium"
                  >
                    = {delta}%
                  </motion.span>
                )}
              </div>
            </div>
            
            <Slider.Root
              className="relative flex items-center select-none touch-none w-full h-4"
              value={[currentValue]}
              onValueChange={(value) => handleChange(param.id, value[0])}
              min={param.min}
              max={param.max}
              step={(param.max - param.min) / 100}
            >
              <Slider.Track className="bg-white/5 relative grow rounded-full h-1">
                <Slider.Range className="absolute bg-gradient-to-r from-primary/60 to-secondary/40 rounded-full h-full" />
              </Slider.Track>
              <Slider.Thumb 
                className="block w-3 h-3 bg-[#00fff2] rounded-full hover:bg-[#00fff2]/90 
                         focus:outline-none focus:ring-2 focus:ring-primary/50 cursor-grab active:cursor-grabbing
                         transition-colors shadow-lg"
                aria-label={param.name}
              />
            </Slider.Root>
          </motion.div>
        );
      })}
    </div>
  );
}
