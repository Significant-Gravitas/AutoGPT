import React from 'react';
import clsx from 'clsx';
import type { SensitivityTable } from '../../types/deal';

interface Props {
  table: SensitivityTable;
  title: string;
  rowLabel: string;
  colLabel: string;
  formatRow?: (v: number) => string;
  formatCol?: (v: number) => string;
  formatCell?: (v: number) => string;
  thresholds?: { green: number; yellow: number }; // above green=green, above yellow=yellow, else red
}

function cellColor(val: number | null, thresholds: { green: number; yellow: number }): string {
  if (val == null) return 'bg-gray-100 dark:bg-gray-700 text-gray-400';
  if (val >= thresholds.green) return 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-800 dark:text-emerald-300 font-semibold';
  if (val >= thresholds.yellow) return 'bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300';
  return 'bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-300';
}

const fmtDefault = (v: number) => v.toFixed(2);

export function SensitivityHeatmap({
  table,
  title,
  rowLabel,
  colLabel,
  formatRow = fmtDefault,
  formatCol = fmtDefault,
  formatCell = fmtDefault,
  thresholds = { green: 14, yellow: 10 },
}: Props) {
  return (
    <div className="overflow-x-auto">
      <div className="mb-2">
        <h4 className="text-sm font-semibold text-gray-800 dark:text-gray-200">{title}</h4>
        <p className="text-xs text-gray-500">
          Row: {rowLabel} &nbsp;|&nbsp; Col: {colLabel}
        </p>
      </div>
      <table className="text-xs border-collapse w-full">
        <thead>
          <tr>
            <th className="px-2 py-1.5 text-left text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 font-medium">
              {rowLabel} ↓ / {colLabel} →
            </th>
            {table.col_values.map((cv) => (
              <th
                key={cv}
                className="px-2 py-1.5 text-center text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 font-medium whitespace-nowrap"
              >
                {formatCol(cv)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.row_values.map((rv, ri) => (
            <tr key={rv}>
              <td className="px-2 py-1.5 text-left text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 font-medium whitespace-nowrap">
                {formatRow(rv)}
              </td>
              {table.col_values.map((_, ci) => {
                const val = table.table[ri]?.[ci] ?? null;
                return (
                  <td
                    key={ci}
                    className={clsx(
                      'px-2 py-1.5 text-center border border-gray-200 dark:border-gray-700 tabular-nums',
                      cellColor(val, thresholds)
                    )}
                  >
                    {val != null ? formatCell(val) : '—'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-4 mt-2">
        {[
          { cls: 'bg-emerald-100 dark:bg-emerald-900/40', label: `≥ ${thresholds.green}%` },
          { cls: 'bg-amber-100 dark:bg-amber-900/40', label: `${thresholds.yellow}–${thresholds.green}%` },
          { cls: 'bg-red-100 dark:bg-red-900/40', label: `< ${thresholds.yellow}%` },
        ].map((leg) => (
          <div key={leg.label} className="flex items-center gap-1.5">
            <div className={clsx('w-3 h-3 rounded-sm', leg.cls)} />
            <span className="text-xs text-gray-500">{leg.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
