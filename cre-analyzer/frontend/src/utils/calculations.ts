/**
 * Client-side calculation utilities for instant feedback.
 * The authoritative calculations run on the backend.
 */

export function fmtCurrency(n: number | null | undefined, compact = false): string {
  if (n == null) return '—';
  if (compact) {
    if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
    if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  }
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
}

export function fmtPct(n: number | null | undefined, decimals = 2): string {
  if (n == null) return '—';
  return `${n.toFixed(decimals)}%`;
}

export function fmtNum(n: number | null | undefined, decimals = 2): string {
  if (n == null) return '—';
  return n.toFixed(decimals);
}

export function fmtMultiple(n: number | null | undefined): string {
  if (n == null) return '—';
  return `${n.toFixed(2)}x`;
}

export function computeEGI(t12: {
  gross_potential_rent: number;
  vacancy_loss: number;
  concessions: number;
  bad_debt: number;
  other_income: number;
}): number {
  return (
    t12.gross_potential_rent -
    t12.vacancy_loss -
    t12.concessions -
    t12.bad_debt +
    t12.other_income
  );
}

export function computeT12Expenses(t12: {
  property_taxes: number;
  insurance: number;
  management_fee: number;
  maintenance_repairs: number;
  utilities: number;
  payroll: number;
  general_admin: number;
  marketing: number;
  capex_reserves: number;
  other_expenses: number;
}): number {
  return (
    t12.property_taxes +
    t12.insurance +
    t12.management_fee +
    t12.maintenance_repairs +
    t12.utilities +
    t12.payroll +
    t12.general_admin +
    t12.marketing +
    t12.capex_reserves +
    t12.other_expenses
  );
}

export function computeNOI(egi: number, expenses: number): number {
  return egi - expenses;
}

export function computeGoingInCap(noi: number, purchasePrice: number): number {
  return purchasePrice > 0 ? (noi / purchasePrice) * 100 : 0;
}

export function rentRollSummary(units: { market_rent: number; current_rent: number; status: string }[]) {
  const total = units.length;
  const occupied = units.filter((u) => u.status === 'Occupied').length;
  const vacant = units.filter((u) => u.status === 'Vacant').length;
  const mtm = units.filter((u) => u.status === 'MTM').length;
  const avgMarket = total ? units.reduce((s, u) => s + u.market_rent, 0) / total : 0;
  const avgCurrent = total ? units.reduce((s, u) => s + u.current_rent, 0) / total : 0;
  return {
    total,
    occupied,
    vacant,
    mtm,
    occupancy_pct: total ? (occupied / total) * 100 : 0,
    avg_market_rent: avgMarket,
    avg_current_rent: avgCurrent,
    total_market_gpr: units.reduce((s, u) => s + u.market_rent * 12, 0),
    total_current_gpr: units.reduce((s, u) => s + u.current_rent * 12, 0),
  };
}
