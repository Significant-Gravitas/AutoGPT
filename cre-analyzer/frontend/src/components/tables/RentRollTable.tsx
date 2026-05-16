import React, { useState } from 'react';
import clsx from 'clsx';
import type { RentRollUnit } from '../../types/deal';
import { fmtCurrency } from '../../utils/calculations';

interface Props {
  units: RentRollUnit[];
  onEdit?: (idx: number, field: keyof RentRollUnit, value: string | number) => void;
}

const statusColors: Record<string, string> = {
  Occupied: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400',
  Vacant: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400',
  MTM: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400',
};

export function RentRollTable({ units, onEdit }: Props) {
  const [page, setPage] = useState(0);
  const PER_PAGE = 20;
  const pages = Math.ceil(units.length / PER_PAGE);
  const visible = units.slice(page * PER_PAGE, (page + 1) * PER_PAGE);

  const summary = {
    total: units.length,
    occupied: units.filter((u) => u.status === 'Occupied').length,
    vacant: units.filter((u) => u.status === 'Vacant').length,
    mtm: units.filter((u) => u.status === 'MTM').length,
    avgMarket: units.length ? units.reduce((s, u) => s + u.market_rent, 0) / units.length : 0,
    avgCurrent: units.length ? units.reduce((s, u) => s + u.current_rent, 0) / units.length : 0,
  };

  return (
    <div>
      {/* Summary bar */}
      <div className="flex flex-wrap gap-4 mb-3 text-xs">
        {[
          { label: 'Total', value: summary.total },
          { label: 'Occupied', value: summary.occupied, cls: 'text-emerald-600' },
          { label: 'Vacant', value: summary.vacant, cls: 'text-red-500' },
          { label: 'MTM', value: summary.mtm, cls: 'text-amber-500' },
          { label: 'Occ %', value: `${summary.total ? ((summary.occupied / summary.total) * 100).toFixed(1) : 0}%` },
          { label: 'Avg Market', value: fmtCurrency(summary.avgMarket) },
          { label: 'Avg Current', value: fmtCurrency(summary.avgCurrent) },
        ].map((s) => (
          <div key={s.label} className="flex gap-1">
            <span className="text-gray-500">{s.label}:</span>
            <span className={clsx('font-semibold', s.cls)}>{s.value}</span>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              {['Unit', 'Type', 'SqFt', 'Market Rent', 'Current Rent', 'Lease End', 'Status'].map((h) => (
                <th key={h} className="table-header">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((u, i) => (
              <tr key={i} className={i % 2 === 0 ? 'table-row-alt' : ''}>
                <td className="table-cell text-left font-mono">{u.unit_number}</td>
                <td className="table-cell text-left">{u.unit_type}</td>
                <td className="table-cell">{u.sqft.toLocaleString()}</td>
                <td className="table-cell">{fmtCurrency(u.market_rent)}</td>
                <td className={clsx('table-cell', u.current_rent < u.market_rent ? 'text-amber-600 dark:text-amber-400' : '')}>
                  {fmtCurrency(u.current_rent)}
                </td>
                <td className="table-cell">{u.lease_end || '—'}</td>
                <td className="table-cell text-center">
                  <span className={clsx('px-1.5 py-0.5 rounded-full text-xs font-medium', statusColors[u.status])}>
                    {u.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {pages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-3">
          <button
            className="btn-secondary px-2 py-1 text-xs"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            ←
          </button>
          <span className="text-xs text-gray-500">Page {page + 1} of {pages}</span>
          <button
            className="btn-secondary px-2 py-1 text-xs"
            disabled={page === pages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            →
          </button>
        </div>
      )}
    </div>
  );
}
