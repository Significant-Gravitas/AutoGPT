import React from 'react';
import clsx from 'clsx';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  suffix?: string;
  prefix?: string;
  hint?: string;
  error?: string;
}

export function Input({ label, suffix, prefix, hint, error, className, ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label className="text-xs font-medium text-gray-600 dark:text-gray-400">{label}</label>
      )}
      <div className="relative flex items-center">
        {prefix && (
          <span className="absolute left-3 text-sm text-gray-400 pointer-events-none">{prefix}</span>
        )}
        <input
          className={clsx(
            'input-field',
            prefix && 'pl-7',
            suffix && 'pr-10',
            error && 'border-red-400 focus:ring-red-400',
            className
          )}
          {...props}
        />
        {suffix && (
          <span className="absolute right-3 text-sm text-gray-400 pointer-events-none">{suffix}</span>
        )}
      </div>
      {hint && !error && <span className="text-xs text-gray-400">{hint}</span>}
      {error && <span className="text-xs text-red-500">{error}</span>}
    </div>
  );
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: { value: string | number; label: string }[];
}

export function Select({ label, options, className, ...props }: SelectProps) {
  return (
    <div className="flex flex-col gap-1">
      {label && <label className="text-xs font-medium text-gray-600 dark:text-gray-400">{label}</label>}
      <select className={clsx('input-field', className)} {...props}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}

export function Toggle({ label, checked, onChange, hint }: ToggleProps) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <div className="relative">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <div className={clsx(
          'w-10 h-6 rounded-full transition',
          checked ? 'bg-brand-600' : 'bg-gray-300 dark:bg-gray-600'
        )} />
        <div className={clsx(
          'absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform',
          checked && 'translate-x-4'
        )} />
      </div>
      <div>
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
        {hint && <p className="text-xs text-gray-500">{hint}</p>}
      </div>
    </label>
  );
}
