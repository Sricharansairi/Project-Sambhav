import { motion } from 'motion/react';
import { FileText, FileSpreadsheet, FileJson, FileCode, Image, Share2, Download, Database, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { exports, type ExportPayload, SambhavAPIError } from '../lib/api';

interface ExportPanelProps {
  payload:  ExportPayload;   // prediction data to export
  delay?:   number;
}

const exportFormats = [
  { id: 'pdf',   label: 'PDF',    icon: FileText,        handler: (p: ExportPayload) => exports.pdf(p) },
  { id: 'word',  label: 'Word',   icon: FileText,        handler: (p: ExportPayload) => exports.word(p) },
  { id: 'excel', label: 'Excel',  icon: FileSpreadsheet, handler: (p: ExportPayload) => exports.excel(p) },
  { id: 'json',  label: 'JSON',   icon: FileJson,        handler: (p: ExportPayload) => exports.json(p) },
  { id: 'csv',   label: 'CSV',    icon: FileSpreadsheet, handler: (p: ExportPayload) => exports.csv(p) },
  { id: 'xml',   label: 'XML',    icon: FileCode,        handler: (p: ExportPayload) => exports.xml(p) },
  { id: 'png',   label: 'PNG',    icon: Image,           handler: (p: ExportPayload) => exports.png(p) },
  { id: 'api',   label: 'API Link',icon: Database,       handler: async (p: ExportPayload) => {
      const res = await exports.apiLink(p);
      await navigator.clipboard.writeText(res.api_url || '').catch(() => {});
      return res;
    }
  },
];

export function ExportPanel({ payload, delay = 0 }: ExportPanelProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error,   setError]   = useState<string | null>(null);
  const [copied,  setCopied]  = useState(false);

  const handleExport = async (format: typeof exportFormats[0]) => {
    setLoading(format.id);
    setError(null);
    try {
      const result = await format.handler(payload);
      if (format.id === 'api' && result?.api_url) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch (err) {
      const msg = err instanceof SambhavAPIError ? err.message : 'Export failed';
      setError(`${format.label}: ${msg}`);
      setTimeout(() => setError(null), 4000);
    } finally {
      setLoading(null);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="p-3 rounded-lg bg-white/5 border border-white/10"
    >
      <div className="flex items-center justify-between mb-2">
        <h5 className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5">
          <Download className="w-3 h-3" />
          Export Results
        </h5>
        {copied && (
          <span className="text-[10px] text-primary">API link copied!</span>
        )}
        {error && (
          <span className="text-[10px] text-destructive truncate max-w-[140px]">{error}</span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-1.5">
        {exportFormats.map((format, idx) => {
          const Icon      = format.icon;
          const isLoading = loading === format.id;
          return (
            <motion.button
              key={format.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: delay + idx * 0.05 }}
              onClick={() => handleExport(format)}
              disabled={!!loading}
              className="flex flex-col items-center gap-1 p-2 rounded-lg bg-white/5
                       border border-white/10 hover:bg-white/10 hover:border-primary/30
                       transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={!loading ? { scale: 1.05 } : {}}
              whileTap={!loading ? { scale: 0.95 } : {}}
            >
              {isLoading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
              ) : (
                <Icon className="w-3.5 h-3.5 text-muted-foreground group-hover:text-primary transition-colors" />
              )}
              <span className="text-[9px] text-muted-foreground group-hover:text-foreground transition-colors">
                {format.label}
              </span>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}
