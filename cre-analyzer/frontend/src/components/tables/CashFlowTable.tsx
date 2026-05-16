import React, { useState } from 'react';
import clsx from 'clsx';
import type { AnnualRow, ExitSummary } from '../../types/deal';
import { fmtCurrency, fmtPct, fmtNum } from '../../utils/calculations';

interface Props {
  rows: AnnualRow[];
  exit: ExitSummary;
}

type Section = 'income' | 'expenses' | 'summary';

interface RowDef {
  key: keyof AnnualRow | string;
  label: string;
  bold?: boolean;
  indent?: boolean;
  negative?: boolean;
  pct?: boolean;
  separator?: boolean;
}

const INCOME_ROWS: RowDef[] = [
  { key: 'gross_potential_rent', label: 'Gross Potential Rent', bold: false },
  { key: 'vacancy_loss', label: 'Less: Vacancy Loss', indent: true, negative: true },
  { key: 'credit_loss', label: 'Less: Credit Loss', indent: true, negative: true },
  { key: 'other_income', label: 'Plus: Other Income', indent: true },
  { key: 'egi', label: 'Effective Gross Income', bold: true, separator: true },
];

const EXPENSE_ROWS: RowDef[] = [
  { key: 'property_taxes', label: 'Property Taxes', indent: true, negative: true },
  { key: 'insurance', label: 'Insurance', indent: true, negative: true },
  { key: 'management_fee', label: 'Management Fee', indent: true, negative: true },
  { key: 'maintenance', label: 'Maintenance & Repairs', indent: true, negative: true },
  { key: 'utilities', label: 'Utilities', indent: true, negative: true },
  { key: 'payroll', label: 'Payroll', indent: true, negative: true },
  { key: 'general_admin', label: 'General & Admin', indent: true, negative: true },
  { key: 'marketing', label: 'Marketing', indent: true, negative: true },
  { key: 'capex_reserves', label: 'CapEx Reserves', indent: true, negative: true },
  { key: 'total_expenses', label: 'Total Operating Expenses', bold: true, negative: true },
];

const SUMMARY_ROWS: RowDef[] = [
  { key: 'noi', label: 'Net Operating Income', bold: true, separator: true },
  { key: 'debt_service', label: 'Less: Debt Service', indent: true, negative: true },
  { key: 'levered_cf', label: 'Levered Cash Flow', bold: true },
  { key: 'dscr', label: 'DSCR', pct: false },
];

export function CashFlowTable({ rows, exit }: Props) {
  const [collapsed, setCollapsed] = useState<Set<Section>>(new Set());

  const toggle = (s: Section) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      next.has(s) ? next.delete(s) : next.add(s);
      return next;
    });

  const renderRow = (def: RowDef) => (
    <tr
      key={def.key}
      className={clsx(
        'hover:bg-gray-50 dark:hover:bg-gray-700/30',
        def.separator && 'border-t-2 border-gray-300 dark:border-gray-600'
      )}
    >
      <td className={clsx('px-3 py-1.5 text-sm text-left', def.indent && 'pl-6', def.bold && 'font-semibold')}>
        {def.label}
      </td>
      {rows.map((r) => {
        const raw = r[def.key as keyof AnnualRow] as number | null;
        const display =
          def.pct === false && def.key === 'dscr'
            ? fmtNum(raw)
            : fmtCurrency(raw);
        return (
          <td
            key={r.year}
            className={clsx(
              'table-cell',
              def.bold && 'font-semibold',
              def.negative && typeof raw === 'number' && raw > 0 && 'text-red-500'
            )}
          >
            {def.negative && typeof raw === 'number' && raw > 0 ? `(${fmtCurrency(raw)})` : display}
          </td>
        );
      })}
    </tr>
  );

  const headerCls = 'bg-gray-100 dark:bg-gray-700 text-left px-3 py-1.5 text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600 select-none';

  return (
    <div className="overflow-x-auto text-sm">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="table-header text-left px-3 py-2 text-gray-600 dark:text-gray-300">Line Item</th>
            {rows.map((r) => (
              <th key={r.year} className="table-header">Year {r.year}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {/* Income */}
          <tr>
            <td
              colSpan={rows.length + 1}
              className={headerCls}
              onClick={() => toggle('income')}
            >
              {collapsed.has('income') ? '▶' : '▼'} Income
            </td>
          </tr>
          {!collapsed.has('income') && INCOME_ROWS.map(renderRow)}

          {/* Expenses */}
          <tr>
            <td
              colSpan={rows.length + 1}
              className={headerCls}
              onClick={() => toggle('expenses')}
            >
              {collapsed.has('expenses') ? '▶' : '▼'} Operating Expenses
            </td>
          </tr>
          {!collapsed.has('expenses') && EXPENSE_ROWS.map(renderRow)}

          {/* Summary */}
          <tr>
            <td
              colSpan={rows.length + 1}
              className={headerCls}
              onClick={() => toggle('summary')}
            >
              {collapsed.has('summary') ? '▶' : '▼'} Cash Flow Summary
            </td>
          </tr>
          {!collapsed.has('summary') && SUMMARY_ROWS.map(renderRow)}

          {/* Exit row */}
          <tr className="border-t-2 border-gray-300 dark:border-gray-600 bg-brand-50 dark:bg-brand-900/20">
            <td className="px-3 py-2 text-sm font-bold text-brand-800 dark:text-brand-300">Exit / Reversion</td>
            <td colSpan={rows.length - 1} />
            <td className="table-cell font-bold text-brand-700 dark:text-brand-300">
              {fmtCurrency(exit.net_sale_proceeds)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
