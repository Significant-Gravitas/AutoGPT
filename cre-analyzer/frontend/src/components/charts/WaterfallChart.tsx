import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { WaterfallYear } from '../../types/deal';

interface Props {
  years: WaterfallYear[];
}

const fmt = (v: number) =>
  Math.abs(v) >= 1_000_000
    ? `$${(v / 1_000_000).toFixed(2)}M`
    : `$${(v / 1_000).toFixed(0)}K`;

export function WaterfallDistributionChart({ years }: Props) {
  const data = years.map((y) => ({
    name: y.is_exit ? `Yr ${y.year}\n(Exit)` : `Yr ${y.year}`,
    'RoC (LP)': y.tier_roc_lp,
    'Pref Return': y.tier_preferred_return,
    'GP Catch-up': y.tier_gp_catchup,
    'LP Promote': y.tier_lp_promote,
    'GP Promote': y.tier_gp_promote,
    'RoC (GP)': y.tier_roc_gp,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 16, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="name" tick={{ fontSize: 10 }} />
        <YAxis tickFormatter={fmt} tick={{ fontSize: 11 }} width={70} />
        <Tooltip formatter={(v: number) => fmt(v)} contentStyle={{ fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="RoC (LP)" stackId="a" fill="#93c5fd" />
        <Bar dataKey="RoC (GP)" stackId="a" fill="#c4b5fd" />
        <Bar dataKey="Pref Return" stackId="a" fill="#6ee7b7" />
        <Bar dataKey="GP Catch-up" stackId="a" fill="#fcd34d" />
        <Bar dataKey="LP Promote" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} />
        <Bar dataKey="GP Promote" stackId="a" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function LPGPSplitChart({ years }: Props) {
  const data = years.map((y) => ({
    name: y.is_exit ? `Exit` : `Yr ${y.year}`,
    LP: y.lp_distribution,
    GP: y.gp_distribution,
  }));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 16, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
        <YAxis tickFormatter={fmt} tick={{ fontSize: 11 }} width={70} />
        <Tooltip formatter={(v: number) => fmt(v)} contentStyle={{ fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="LP" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        <Bar dataKey="GP" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
