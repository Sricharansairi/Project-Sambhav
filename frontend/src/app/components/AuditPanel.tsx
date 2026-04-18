import { motion } from 'motion/react';
import { AlertTriangle, CheckCircle2, AlertCircle } from 'lucide-react';

interface AuditRow {
  type: 'parameter' | 'prediction' | 'confidence';
  status: 'pass' | 'warning' | 'fail';
  label: string;
}

interface AuditPanelProps {
  audits: AuditRow[];
  abnFlags: string[];
  mlLlmAgreement: 'high' | 'moderate' | 'low';
  delay?: number;
}

export function AuditPanel({ audits, abnFlags, mlLlmAgreement, delay = 0 }: AuditPanelProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle2 className="w-3 h-3 text-[#c0c0c0]" />;
      case 'warning':
        return <AlertTriangle className="w-3 h-3 text-[#ffb7c5]" />;
      case 'fail':
        return <AlertCircle className="w-3 h-3 text-[#ff6b6b]" />;
      default:
        return null;
    }
  };

  const getAgreementColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'text-[#00fff2] border-[#00fff2]/30 bg-[#00fff2]/10';
      case 'moderate':
        return 'text-[#c0c0c0] border-white/20 bg-white/5';
      case 'low':
        return 'text-[#ff6b6b] border-[#ff6b6b]/30 bg-[#ff6b6b]/10';
      default:
        return 'text-muted-foreground border-white/10 bg-white/5';
    }
  };

  return (
    <div className="space-y-4">
      {/* Audit Rows */}
      <div className="space-y-2">
        <h5 className="text-[11px] font-medium text-muted-foreground">Forensic Audit System</h5>
        
        {audits.map((audit, idx) => (
          <motion.div
            key={audit.type}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delay + idx * 0.1 }}
            className="flex items-center justify-between p-2 rounded-lg bg-white/5 border border-white/10"
          >
            <div className="flex items-center gap-2">
              {getStatusIcon(audit.status)}
              <span className="text-[10px] text-foreground/80">{audit.label}</span>
            </div>
            <div className="flex items-center gap-1">
              <div className={`w-1.5 h-1.5 rounded-full ${
                audit.status === 'pass' ? 'bg-[#c0c0c0]' :
                audit.status === 'warning' ? 'bg-[#ffb7c5]' : 'bg-[#ff6b6b]'
              }`} />
              <span className={`text-[9px] uppercase font-medium ${
                audit.status === 'pass' ? 'text-[#c0c0c0]' :
                audit.status === 'warning' ? 'text-[#ffb7c5]' : 'text-[#ff6b6b]'
              }`}>
                {audit.status}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* ABN Flags */}
      {abnFlags.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: delay + 0.3 }}
          className="space-y-1.5"
        >
          <h5 className="text-[11px] font-medium text-muted-foreground">Active System Warnings</h5>
          <div className="flex flex-wrap gap-1.5">
            {abnFlags.map((flag, idx) => (
              <motion.span
                key={flag}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: delay + 0.4 + idx * 0.05 }}
                className="px-2 py-0.5 text-[9px] font-medium rounded-full 
                         bg-[#ffb7c5]/20 border border-[#ffb7c5]/40 text-[#ffb7c5]"
              >
                {flag}
              </motion.span>
            ))}
          </div>
        </motion.div>
      )}

      {/* ML vs LLM Consistency */}
      <motion.div
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: delay + 0.5 }}
        className="space-y-1.5"
      >
        <h5 className="text-[11px] font-medium text-muted-foreground">ML vs LLM Agreement</h5>
        <div className={`
          inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[10px] font-medium
          ${getAgreementColor(mlLlmAgreement)}
        `}>
          <div className={`w-1.5 h-1.5 rounded-full ${
            mlLlmAgreement === 'high' ? 'bg-[#00fff2]' :
            mlLlmAgreement === 'moderate' ? 'bg-[#c0c0c0]' : 'bg-[#ff6b6b]'
          }`} />
          <span className="uppercase tracking-wide">
            {mlLlmAgreement} Consistency
          </span>
        </div>
      </motion.div>

      {/* Failure Scenarios (Full Breakdown) */}
      <motion.div
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: delay + 0.6 }}
        className="space-y-1.5"
      >
        <h5 className="text-[11px] font-medium text-muted-foreground">Failure Scenarios</h5>
        <div className="space-y-1">
          {[
            'Missing contextual variables may reduce accuracy',
            'Temporal drift detected in historical patterns',
            'Low sample size in edge case scenarios'
          ].map((scenario, idx) => (
            <div key={idx} className="flex gap-2 text-[10px] text-foreground/70">
              <div className="w-1 h-1 rounded-full bg-[#ff6b6b]/60 mt-1 shrink-0" />
              <span>{scenario}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
