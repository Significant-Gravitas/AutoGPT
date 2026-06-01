import React, { useState, useEffect } from 'react';
import clsx from 'clsx';

interface NumericInputProps {
  label?: string;
  value: number;
  onChange: (value: number) => void;
  prefix?: string;
  suffix?: string;
  hint?: string;
  decimals?: number;
  className?: string;
  disabled?: boolean;
}

function formatWithCommas(n: number, decimals = 0): string {
  if (!isFinite(n)) return '';
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function parseFormatted(s: string): number {
  const cleaned = s.replace(/,/g, '').trim();
  const n = parseFloat(cleaned);
  return isNaN(n) ? 0 : n;
}

/**
 * A text input that displays numbers with thousands commas when not focused,
 * and allows free-form numeric entry when focused.
 */
export function NumericInput({
  label,
  value,
  onChange,
  prefix,
  suffix,
  hint,
  decimals = 0,
  className,
  disabled,
}: NumericInputProps) {
  const [focused, setFocused] = useState(false);
  const [raw, setRaw] = useState('');

  // When focus is gained, populate raw with the plain number string
  const handleFocus = () => {
    setRaw(value === 0 ? '' : String(value));
    setFocused(true);
  };

  const handleBlur = () => {
    setFocused(false);
    const parsed = parseFormatted(raw);
    onChange(parsed);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Allow digits, commas, dots, minus
    const v = e.target.value.replace(/[^0-9.,\-]/g, '');
    setRaw(v);
  };

  const displayValue = focused ? raw : formatWithCommas(value, decimals);

  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label className="text-xs font-medium text-gray-600 dark:text-gray-400">{label}</label>
      )}
      <div className="relative flex items-center">
        {prefix && (
          <span className="absolute left-3 text-sm text-gray-400 pointer-events-none select-none">
            {prefix}
          </span>
        )}
        <input
          type="text"
          inputMode="numeric"
          className={clsx(
            'input-field tabular-nums',
            prefix && 'pl-7',
            suffix && 'pr-10',
            className
          )}
          value={displayValue}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onChange={handleChange}
          disabled={disabled}
        />
        {suffix && (
          <span className="absolute right-3 text-sm text-gray-400 pointer-events-none select-none">
            {suffix}
          </span>
        )}
      </div>
      {hint && <span className="text-xs text-gray-400">{hint}</span>}
    </div>
  );
}
