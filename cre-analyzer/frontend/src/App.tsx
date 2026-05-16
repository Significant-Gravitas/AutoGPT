import React, { useEffect } from 'react';
import clsx from 'clsx';
import { Moon, Sun, Building2, RotateCcw, Upload } from 'lucide-react';
import { useDealStore } from './store/dealStore';
import { WizardNav } from './components/wizard/WizardNav';
import { UploadPage } from './pages/Upload';
import { ReviewDataPage } from './pages/ReviewData';
import { AssumptionsPage } from './pages/Assumptions';
import { FinancingPage } from './pages/Financing';
import { WaterfallConfigPage } from './pages/WaterfallConfig';
import { ResultsPage } from './pages/Results';
import type { WizardStep } from './types/deal';

const STEP_ORDER: WizardStep[] = ['upload', 'review', 'assumptions', 'financing', 'waterfall', 'results'];

export default function App() {
  const { step, darkMode, toggleDarkMode, setStep, resetDeal, deal, loadFromJson } = useDealStore();

  // Apply dark mode class to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const completedSteps = new Set<WizardStep>(
    STEP_ORDER.slice(0, STEP_ORDER.indexOf(step))
  );

  const handleNavigate = (s: WizardStep) => setStep(s);

  const handleFileLoad = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      loadFromJson(text);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <div className={clsx('min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors', darkMode && 'dark')}>
      {/* Top Nav */}
      <header className="sticky top-0 z-50 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center gap-4">
          <div className="flex items-center gap-2 flex-shrink-0">
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
              <Building2 size={16} className="text-white" />
            </div>
            <span className="font-bold text-gray-900 dark:text-white hidden sm:block">CRE Deal Analyzer</span>
          </div>

          <div className="flex-1 overflow-hidden">
            <WizardNav current={step} completed={completedSteps} onNavigate={handleNavigate} />
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <label className="btn-secondary cursor-pointer text-xs px-2 py-1.5">
              <Upload size={12} />
              <span className="hidden sm:inline">Load JSON</span>
              <input type="file" accept=".json" className="sr-only" onChange={handleFileLoad} />
            </label>
            <button onClick={resetDeal} className="btn-secondary px-2 py-1.5 text-xs" title="New Deal">
              <RotateCcw size={12} />
            </button>
            <button onClick={toggleDarkMode} className="btn-secondary px-2 py-1.5">
              {darkMode ? <Sun size={14} /> : <Moon size={14} />}
            </button>
          </div>
        </div>
      </header>

      {/* Deal name subtitle */}
      <div className="bg-brand-600 dark:bg-brand-800 px-4 sm:px-6 py-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <span className="text-sm text-brand-100 font-medium truncate">{deal.name || deal.property_info.name}</span>
          <span className="text-xs text-brand-200">{deal.property_info.units} units · ${(deal.property_info.purchase_price / 1_000_000).toFixed(1)}M</span>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6">
        {step === 'upload' && <UploadPage />}
        {step === 'review' && <ReviewDataPage />}
        {step === 'assumptions' && <AssumptionsPage />}
        {step === 'financing' && <FinancingPage />}
        {step === 'waterfall' && <WaterfallConfigPage />}
        {step === 'results' && <ResultsPage />}
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-200 dark:border-gray-800 px-4 py-6 text-center text-xs text-gray-400">
        CRE Deal Analyzer · Powered by Claude · For professional use only · Not financial advice
      </footer>
    </div>
  );
}
