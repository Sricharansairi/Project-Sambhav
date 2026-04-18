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
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-2xl border border-white/10 bg-[#0a0a0f] backdrop-blur-md overflow-hidden"
    >
      <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between bg-white/5">
        <div>
          <h3 className="text-sm font-bold text-foreground">Export Data Matrix</h3>
          <p className="text-[10px] text-muted-foreground mt-0.5">Download full datasets, metrics, and documents.</p>
        </div>
        <div className="w-9 h-9 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
          <Download className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>

      <div className="p-4 space-y-4">
        {error && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-[11px] text-destructive font-medium text-center">
            {error}
          </motion.div>
        )}

        <div className="flex flex-col gap-2 relative">
          <div className="absolute left-4 top-4 bottom-4 w-px bg-white/5 z-0 hidden sm:block" />
          
          {exportGroups.flatMap(g => g.formats).map((fmt, fIdx) => {
            const Icon = fmt.icon;
            const isLoading = loading === fmt.id;
            const isSuccess = success === fmt.id;

            return (
              <motion.button
                key={fmt.id}
                onClick={() => handleExport(fmt.id, fmt.handler)}
                disabled={!!loading}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: delay + fIdx * 0.05 + 0.1 }}
                className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all z-10 group disabled:opacity-50 disabled:cursor-not-allowed
                  ${isSuccess ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-[#0f0f15] border-white/5 hover:bg-white/5 hover:border-white/10'}`}
                whileHover={!loading && !isSuccess ? { scale: 1.01 } : {}}
                whileTap={!loading && !isSuccess ? { scale: 0.99 } : {}}
              >
                <div className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0
                    ${isSuccess ? 'bg-emerald-500/20' : 'bg-white/5 group-hover:bg-white/10 transition-colors'}`}>
                    {isLoading ? (
                      <Loader2 className="w-4 h-4 animate-spin text-foreground" />
                    ) : isSuccess ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Icon className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                    )}
                  </div>
                  <div className="text-left">
                    <p className={`text-[12px] font-semibold transition-colors
                      ${isSuccess ? 'text-emerald-400' : 'text-foreground/90 group-hover:text-foreground'}`}>
                      {fmt.label}
                    </p>
                    <p className="text-[10px] text-muted-foreground capitalize">
                      {fmt.id === 'api' ? 'Generate Live API JSON URL' : `Download ${fmt.id.toUpperCase()} file`}
                    </p>
                  </div>
                </div>

                <div className="pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Download className="w-4 h-4 text-muted-foreground" />
                </div>
              </motion.button>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}
