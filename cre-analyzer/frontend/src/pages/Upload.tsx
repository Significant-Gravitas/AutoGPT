import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload as UploadIcon, FileText, Loader2, Sparkles, AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { useDealStore } from '../store/dealStore';
import { parseDocument, getDemoData } from '../api/client';
import type { WizardStep } from '../types/deal';

type DocType = 'om' | 't12' | 'rent_roll';

interface DocSlot {
  type: DocType;
  label: string;
  desc: string;
  extensions: string[];
}

const SLOTS: DocSlot[] = [
  { type: 'om', label: 'Offering Memorandum', desc: 'PDF with property details, pricing, sponsor projections', extensions: ['.pdf'] },
  { type: 't12', label: 'T12 Financials', desc: 'PDF or Excel with trailing 12-month income & expenses', extensions: ['.pdf', '.xlsx', '.xls'] },
  { type: 'rent_roll', label: 'Rent Roll', desc: 'PDF or Excel with unit-by-unit lease data', extensions: ['.pdf', '.xlsx', '.xls'] },
];

interface UploadedFile { file: File; docType: DocType; status: 'pending' | 'parsing' | 'done' | 'error'; result?: Record<string, unknown>; error?: string }

export function UploadPage() {
  const { deal, updateDeal, setStep } = useDealStore();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [demoLoading, setDemoLoading] = useState(false);
  const [parseAll, setParseAll] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const addFile = (file: File, docType: DocType) => {
    setFiles((prev) => {
      const filtered = prev.filter((f) => f.docType !== docType);
      return [...filtered, { file, docType, status: 'pending' }];
    });
  };

  const parseFile = async (entry: UploadedFile): Promise<Record<string, unknown>> => {
    setFiles((prev) => prev.map((f) => f.docType === entry.docType ? { ...f, status: 'parsing' } : f));
    try {
      const res = await parseDocument(entry.file, entry.docType);
      setFiles((prev) => prev.map((f) => f.docType === entry.docType ? { ...f, status: 'done', result: res.data } : f));
      return res.data;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Parse error';
      setFiles((prev) => prev.map((f) => f.docType === entry.docType ? { ...f, status: 'error', error: msg } : f));
      throw e;
    }
  };

  const handleParseAll = async () => {
    if (files.length === 0) { setGlobalError('Upload at least one document first.'); return; }
    setParseAll(true);
    setGlobalError(null);
    const merged: Record<string, unknown> = {};
    for (const f of files) {
      try {
        const data = await parseFile(f);
        Object.assign(merged, data);
      } catch { /* handled in parseFile */ }
    }
    applyExtraction(merged);
    setParseAll(false);
    setStep('review');
  };

  const applyExtraction = (data: Record<string, unknown>) => {
    const updates: Partial<typeof deal> = { raw_extraction: data };
    if (data.property_info) updates.property_info = { ...deal.property_info, ...(data.property_info as object) };
    if (data.t12_data) updates.t12_data = { ...deal.t12_data, ...(data.t12_data as object) };
    if (Array.isArray(data.rent_roll)) updates.rent_roll = data.rent_roll as typeof deal.rent_roll;
    updateDeal(updates);
  };

  const handleDemo = async () => {
    setDemoLoading(true);
    setGlobalError(null);
    try {
      const data = await getDemoData();
      applyExtraction(data);
      setStep('review');
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : 'Failed to load demo');
    } finally {
      setDemoLoading(false);
    }
  };

  const skipToReview = () => setStep('review');

  return (
    <div className="max-w-2xl mx-auto space-y-6 py-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Upload Documents</h2>
        <p className="text-sm text-gray-500 mt-1">Upload one or more deal documents. Claude will extract and normalize the data automatically.</p>
      </div>

      {globalError && (
        <div className="flex items-center gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle size={16} /> {globalError}
        </div>
      )}

      <div className="space-y-3">
        {SLOTS.map((slot) => (
          <DocDropzone
            key={slot.type}
            slot={slot}
            file={files.find((f) => f.docType === slot.type)}
            onDrop={(f) => addFile(f, slot.type)}
          />
        ))}
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <Button onClick={handleParseAll} loading={parseAll} icon={<UploadIcon size={14} />} disabled={files.length === 0}>
          Parse Documents with Claude
        </Button>
        <Button variant="secondary" onClick={handleDemo} loading={demoLoading} icon={<Sparkles size={14} />}>
          Load Demo Deal
        </Button>
        <Button variant="ghost" onClick={skipToReview}>
          Skip → Use Defaults
        </Button>
      </div>

      <p className="text-xs text-gray-400">
        Supported: PDF, XLSX, XLS. Documents are sent to Claude for extraction. Ensure you have your ANTHROPIC_API_KEY set.
      </p>
    </div>
  );
}

function DocDropzone({ slot, file, onDrop }: { slot: DocSlot; file?: UploadedFile; onDrop: (f: File) => void }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (accepted) => accepted[0] && onDrop(accepted[0]),
    accept: slot.extensions.reduce((acc, ext) => {
      if (ext === '.pdf') acc['application/pdf'] = ['.pdf'];
      if (ext === '.xlsx') acc['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] = ['.xlsx'];
      if (ext === '.xls') acc['application/vnd.ms-excel'] = ['.xls'];
      return acc;
    }, {} as Record<string, string[]>),
    maxFiles: 1,
  });

  const statusIcon = () => {
    if (!file) return null;
    if (file.status === 'parsing') return <Loader2 size={14} className="animate-spin text-brand-500" />;
    if (file.status === 'done') return <span className="text-emerald-500 text-xs font-medium">✓ Parsed</span>;
    if (file.status === 'error') return <span className="text-red-500 text-xs">{file.error}</span>;
    return <span className="text-xs text-gray-500">{file.file.name}</span>;
  };

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'card p-4 flex items-center gap-4 cursor-pointer transition',
        isDragActive && 'ring-2 ring-brand-400 bg-brand-50 dark:bg-brand-900/20',
        file?.status === 'done' && 'ring-1 ring-emerald-400'
      )}
    >
      <input {...getInputProps()} />
      <div className="w-10 h-10 rounded-lg bg-brand-100 dark:bg-brand-900/30 flex items-center justify-center flex-shrink-0">
        <FileText size={18} className="text-brand-600 dark:text-brand-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">{slot.label}</p>
        <p className="text-xs text-gray-500 truncate">{file ? statusIcon() : slot.desc}</p>
      </div>
      <div className="text-xs text-gray-400 flex-shrink-0">
        {isDragActive ? 'Drop here' : slot.extensions.join(', ')}
      </div>
    </div>
  );
}
