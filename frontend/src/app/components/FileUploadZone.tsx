import { motion, AnimatePresence } from 'motion/react';
import { Upload, File, CheckCircle, X } from 'lucide-react';
import { useState, useCallback } from 'react';

interface FileUploadZoneProps {
  onFileSelect: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // in MB
}

export function FileUploadZone({ 
  onFileSelect, 
  accept = '*',
  multiple = false,
  maxSize = 50 
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [inputId] = useState(`file-input-${Math.random().toString(36).substr(2, 9)}`);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFiles = useCallback((fileList: FileList | null) => {
    if (!fileList) return;
    
    const filesArray = Array.from(fileList);
    setFiles(filesArray);
    setIsUploading(true);
    
    // Simulate upload animation
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 30;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setTimeout(() => {
          setIsUploading(false);
          onFileSelect(filesArray);
        }, 500);
      }
      setUploadProgress(progress);
    }, 200);
  }, [onFileSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    processFiles(e.dataTransfer.files);
  }, [processFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(e.target.files);
  }, [processFiles]);

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      <motion.div
        className={`
          relative border-2 border-dashed rounded-xl p-12 
          transition-all duration-300 cursor-pointer
          ${isDragging 
            ? 'border-primary bg-[#ffb7c5]/5 scale-[1.02]' 
            : 'border-white/20 hover:border-white/40 bg-white/[0.02]'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById(inputId)?.click()}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <input
          id={inputId}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleFileInput}
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex flex-col items-center gap-6"
            >
              {/* Liquid absorption animation */}
              <div className="relative w-32 h-32">
                <motion.div
                  className="absolute inset-0 rounded-full"
                  style={{
                    background: 'conic-gradient(from 0deg, #00fff2, #9d4eff, #00ff88, #00fff2)',
                  }}
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
                <div className="absolute inset-2 bg-black rounded-full flex items-center justify-center">
                  <File className="w-12 h-12 text-[#ffb7c5]" />
                </div>
              </div>
              
              <div className="w-full max-w-xs space-y-2">
                <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary via-secondary to-accent rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-center text-sm text-muted-foreground">
                  Processing... {Math.round(uploadProgress)}%
                </p>
              </div>
            </motion.div>
          ) : files.length > 0 ? (
            <motion.div
              key="completed"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center justify-center gap-3"
            >
              <CheckCircle className="w-8 h-8 text-[#c0c0c0]" />
              <span className="text-[#c0c0c0]">
                {files.length} file{files.length > 1 ? 's' : ''} ready
              </span>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <motion.div
                animate={{
                  y: [0, -10, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              >
                <Upload className="w-16 h-16 text-[#ffb7c5]/50" />
              </motion.div>
              
              <div className="text-center space-y-2">
                <p className="text-lg">
                  Drop files here or{' '}
                  <span className="text-[#ffb7c5] cursor-pointer hover:underline">
                    browse
                  </span>
                </p>
                <p className="text-sm text-muted-foreground">
                  {accept === '*' ? 'Any file type' : accept} • Max {maxSize}MB
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Glow effect when dragging */}
        {isDragging && (
          <motion.div
            className="absolute inset-0 rounded-xl pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              boxShadow: '0 0 40px rgba(0, 255, 242, 0.3), inset 0 0 40px rgba(0, 255, 242, 0.1)',
            }}
          />
        )}
      </motion.div>

      {/* File list */}
      <AnimatePresence>
        {files.length > 0 && !isUploading && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2"
          >
            {files.map((file, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="flex items-center justify-between p-3 bg-white/[0.03] border border-white/10 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <File className="w-5 h-5 text-[#ffb7c5]" />
                  <div>
                    <p className="text-sm">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                  className="p-1 hover:bg-white/10 rounded transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
