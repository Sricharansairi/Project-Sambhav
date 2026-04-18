import { motion } from 'motion/react';
import { FileText, FileSpreadsheet, FileJson, FileCode, Image, Share2, Download, Database, Loader2, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';
import { exports, type ExportPayload, SambhavAPIError } from '../lib/api';

interface ExportPanelProps {
  payload: ExportPayload;
  delay?: number;
}

const exportGroups = [
  {
    title: 'Document Reports',
    description: 'Beautifully formatted documents for presentations and sharing.',
    formats: [
      { id: 'pdf', label: 'PDF Report', icon: FileText, handler: exports.pdf },
      { id: 'word', label: 'Word Doc', icon: FileText, handler: exports.word },
      { id: 'excel', label: 'Excel Sheet', icon: FileSpreadsheet, handler: exports.excel },
    ],
  },
  {
    title: 'Raw Data & Code',
    description: 'Structured data for engineering and external pipelines.',
    formats: [
      { id: 'json', label: 'JSON Data', icon: FileJson, handler: exports.json },
      { id: 'csv', label: 'CSV Table', icon: FileSpreadsheet, handler: exports.csv },
      { id: 'xml', label: 'XML File', icon: FileCode, handler: exports.xml },
    ],
  },
  {
    title: 'Visuals & Cloud',
    description: 'Export as images or integrate via live API URLs.',
    formats: [
      { id: 'png', label: 'PNG Image', icon: Image, handler: exports.png },
      {
        id: 'api',
        label: 'API Link',
        icon: Database,
        handler: async (p: ExportPayload) => {
          const res = await exports.apiLink(p);
          if (res.api_url) await navigator.clipboard.writeText(res.api_url).catch(() => {});
          return res;
        },
      },
    ],
  },
];

export function ExportPanel({ payload, delay = 0 }: ExportPanelProps) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleExport = async (formatId: string, handler: (p: ExportPayload) => Promise<any>) => {
    setLoading(formatId);
    setError(null);
    setSuccess(null);
    try {
      await handler(payload);
      setSuccess(formatId);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof SambhavAPIError ? err.message : 'Export failed due to a sudden error.');
      setTimeout(() => setError(null), 4000);
    } finally {
      setLoading(null);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-2xl overflow-hidden border border-white/10"
      style={{ background: 'linear-gradient(135deg, rgba(15,15,30,0.8) 0%, rgba(20,20,40,0.8) 100%)' }}
    >
      <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-foreground">Export Results</h3>
          <p className="text-[10px] text-muted-foreground mt-0.5">Download your prediction data in multiple robust formats.</p>
        </div>
        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20">
          <Download className="w-5 h-5 text-primary" />
        </div>
      </div>

      <div className="p-5 space-y-5">
        {error && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-[11px] text-destructive font-medium text-center">
            {error}
          </motion.div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {exportGroups.map((group, gIdx) => (
            <motion.div
              key={group.title}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: delay + gIdx * 0.1 + 0.2 }}
              className="space-y-3"
            >
              <div>
                <p className="text-[11px] font-bold text-foreground uppercase tracking-wider">{group.title}</p>
                <p className="text-[9px] text-muted-foreground/70 mt-0.5 leading-tight">{group.description}</p>
              </div>
              <div className="space-y-2">
                {group.formats.map((fmt) => {
                  const Icon = fmt.icon;
                  const isLoading = loading === fmt.id;
                  const isSuccess = success === fmt.id;

                  return (
                    <motion.button
                      key={fmt.id}
                      onClick={() => handleExport(fmt.id, fmt.handler)}
                      disabled={!!loading}
                      className={`w-full flex items-center justify-between px-3 py-2 rounded-xl border transition-all group disabled:opacity-50 disabled:cursor-not-allowed
                        ${isSuccess ? 'bg-success/10 border-success/30' : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-primary/30'}`}
                      whileHover={!loading && !isSuccess ? { scale: 1.02 } : {}}
                      whileTap={!loading && !isSuccess ? { scale: 0.98 } : {}}
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0
                          ${isSuccess ? 'bg-success/20' : 'bg-white/5 group-hover:bg-primary/20 transition-colors'}`}>
                          {isLoading ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
                          ) : isSuccess ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                          ) : (
                            <Icon className="w-3.5 h-3.5 text-muted-foreground group-hover:text-primary transition-colors" />
                          )}
                        </div>
                        <span className={`text-[11px] font-medium truncate transition-colors
                          ${isSuccess ? 'text-success' : 'text-muted-foreground group-hover:text-foreground'}`}>
                          {isSuccess ? (fmt.id === 'api' ? 'Copied Link!' : 'Exported!') : fmt.label}
                        </span>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
