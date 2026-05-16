import React from 'react';
import clsx from 'clsx';
import { Check } from 'lucide-react';
import type { WizardStep } from '../../types/deal';

const STEPS: { id: WizardStep; label: string; short: string }[] = [
  { id: 'upload', label: 'Upload Docs', short: 'Upload' },
  { id: 'review', label: 'Review Data', short: 'Review' },
  { id: 'assumptions', label: 'Assumptions', short: 'Assume' },
  { id: 'financing', label: 'Financing', short: 'Finance' },
  { id: 'waterfall', label: 'Waterfall', short: 'Waterfall' },
  { id: 'results', label: 'Results', short: 'Results' },
];

interface Props {
  current: WizardStep;
  completed: Set<WizardStep>;
  onNavigate: (step: WizardStep) => void;
}

export function WizardNav({ current, completed, onNavigate }: Props) {
  return (
    <nav className="flex items-center gap-0 overflow-x-auto">
      {STEPS.map((step, i) => {
        const isCompleted = completed.has(step.id);
        const isCurrent = step.id === current;
        const isAccessible = isCompleted || isCurrent;

        return (
          <React.Fragment key={step.id}>
            {i > 0 && (
              <div className={clsx(
                'h-0.5 flex-1 min-w-4',
                isCompleted ? 'bg-brand-500' : 'bg-gray-200 dark:bg-gray-700'
              )} />
            )}
            <button
              onClick={() => isAccessible && onNavigate(step.id)}
              disabled={!isAccessible}
              className={clsx(
                'flex flex-col items-center gap-1 flex-shrink-0',
                isAccessible ? 'cursor-pointer' : 'cursor-not-allowed opacity-40'
              )}
            >
              <div className={clsx(
                'w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold transition',
                isCurrent && 'bg-brand-600 text-white ring-2 ring-brand-300',
                isCompleted && !isCurrent && 'bg-brand-600 text-white',
                !isCurrent && !isCompleted && 'bg-gray-200 dark:bg-gray-700 text-gray-500'
              )}>
                {isCompleted && !isCurrent ? <Check size={14} /> : i + 1}
              </div>
              <span className={clsx(
                'text-xs font-medium hidden sm:block',
                isCurrent ? 'text-brand-600 dark:text-brand-400' : 'text-gray-500'
              )}>
                {step.short}
              </span>
            </button>
          </React.Fragment>
        );
      })}
    </nav>
  );
}
