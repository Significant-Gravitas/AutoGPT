import React from 'react';
import clsx from 'clsx';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  padding?: boolean;
}

export function Card({ title, subtitle, action, padding = true, className, children, ...props }: CardProps) {
  return (
    <div className={clsx('card', className)} {...props}>
      {(title || action) && (
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div>
            {title && <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">{title}</h3>}
            {subtitle && <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{subtitle}</p>}
          </div>
          {action && <div className="flex-shrink-0">{action}</div>}
        </div>
      )}
      <div className={padding ? 'p-5 pt-2' : ''}>{children}</div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  highlight?: 'green' | 'yellow' | 'red' | 'blue' | 'default';
}

export function MetricCard({ label, value, sub, highlight = 'default' }: MetricCardProps) {
  const colors: Record<string, string> = {
    green: 'text-emerald-600 dark:text-emerald-400',
    yellow: 'text-amber-500 dark:text-amber-400',
    red: 'text-red-500 dark:text-red-400',
    blue: 'text-brand-600 dark:text-brand-400',
    default: 'text-gray-900 dark:text-white',
  };
  return (
    <div className="metric-card">
      <span className="label">{label}</span>
      <span className={clsx('text-2xl font-bold', colors[highlight])}>{value}</span>
      {sub && <span className="text-xs text-gray-500 dark:text-gray-400">{sub}</span>}
    </div>
  );
}
