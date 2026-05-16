import React from 'react';
import clsx from 'clsx';
import type { WaterfallYear } from '../../types/deal';
import { fmtCurrency } from '../../utils/calculations';

interface Props {
  years: WaterfallYear[];
}

export function WaterfallTable({ years }: Props) {
  const totals = years.reduce(
    (acc, y) => ({
      distributable: acc.distributable + y.distributable,
      lp: acc.lp + y.lp_distribution,
      gp: acc.gp + y.gp_distribution,
      roc_lp: acc.roc_lp + y.tier_roc_lp,
      roc_gp: acc.roc_gp + y.tier_roc_gp,
      pref: acc.pref + y.tier_preferred_return,
      catchup: acc.catchup + y.tier_gp_catchup,
      lp_prom: acc.lp_prom + y.tier_lp_promote,
      gp_prom: acc.gp_prom + y.tier_gp_promote,
    }),
    { distributable: 0, lp: 0, gp: 0, roc_lp: 0, roc_gp: 0, pref: 0, catchup: 0, lp_prom: 0, gp_prom: 0 }
  );

  const cols = [
    { key: 'distributable', label: 'Distributable' },
    { key: 'lp_distribution', label: 'LP Total' },
    { key: 'gp_distribution', label: 'GP Total' },
    { key: 'tier_roc_lp', label: 'RoC (LP)' },
    { key: 'tier_roc_gp', label: 'RoC (GP)' },
    { key: 'tier_preferred_return', label: 'Pref Return' },
    { key: 'tier_gp_catchup', label: 'GP Catch-up' },
    { key: 'tier_lp_promote', label: 'LP Promote' },
    { key: 'tier_gp_promote', label: 'GP Promote' },
  ] as const;

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <th className="table-header text-left">Period</th>
            {cols.map((c) => (
              <th key={c.key} className="table-header">{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map((y, i) => (
            <tr
              key={y.year}
              className={clsx(
                i % 2 === 0 ? 'table-row-alt' : '',
                y.is_exit && 'bg-brand-50 dark:bg-brand-900/20 font-semibold'
              )}
            >
              <td className="px-3 py-1.5 text-left font-medium">
                {y.is_exit ? `Year ${y.year} (Exit)` : `Year ${y.year}`}
              </td>
              {cols.map((c) => (
                <td key={c.key} className="table-cell">
                  {fmtCurrency(y[c.key as keyof WaterfallYear] as number)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-gray-300 dark:border-gray-600 font-bold bg-gray-100 dark:bg-gray-700">
            <td className="px-3 py-2 text-left">Total</td>
            <td className="table-cell">{fmtCurrency(totals.distributable)}</td>
            <td className="table-cell text-brand-600 dark:text-brand-400">{fmtCurrency(totals.lp)}</td>
            <td className="table-cell text-purple-600 dark:text-purple-400">{fmtCurrency(totals.gp)}</td>
            <td className="table-cell">{fmtCurrency(totals.roc_lp)}</td>
            <td className="table-cell">{fmtCurrency(totals.roc_gp)}</td>
            <td className="table-cell">{fmtCurrency(totals.pref)}</td>
            <td className="table-cell">{fmtCurrency(totals.catchup)}</td>
            <td className="table-cell">{fmtCurrency(totals.lp_prom)}</td>
            <td className="table-cell">{fmtCurrency(totals.gp_prom)}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
