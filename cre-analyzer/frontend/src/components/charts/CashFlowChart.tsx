import React from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { AnnualRow } from '../../types/deal';

interface Props {
  rows: AnnualRow[];
}

const fmt = (v: number) =>
  Math.abs(v) >= 1_000_000
    ? `$${(v / 1_000_000).toFixed(1)}M`
    : `$${(v / 1_000).toFixed(0)}K`;

export function CashFlowChart({ rows }: Props) {
  const data = rows.map((r) => ({
    year: `Yr ${r.year}`,
    NOI: r.noi,
    'Debt Service': -r.debt_service,
    'Levered CF': r.levered_cf,
    EGI: r.egi,
    OpEx: -r.total_expenses,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: 16, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="year" tick={{ fontSize: 11 }} />
        <YAxis tickFormatter={fmt} tick={{ fontSize: 11 }} width={60} />
        <Tooltip
          formatter={(v: number) => fmt(v)}
          contentStyle={{ fontSize: 12 }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="NOI" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        <Bar dataKey="Debt Service" fill="#f87171" radius={[3, 3, 0, 0]} />
        <Line
          type="monotone"
          dataKey="Levered CF"
          stroke="#10b981"
          strokeWidth={2}
          dot={{ r: 4 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function NOIGrowthChart({ rows }: Props) {
  const data = rows.map((r) => ({
    year: `Yr ${r.year}`,
    GPR: r.gross_potential_rent,
    EGI: r.egi,
    NOI: r.noi,
  }));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: 16, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="year" tick={{ fontSize: 11 }} />
        <YAxis tickFormatter={fmt} tick={{ fontSize: 11 }} width={60} />
        <Tooltip formatter={(v: number) => fmt(v)} contentStyle={{ fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="GPR" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="EGI" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="monotone" dataKey="NOI" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
