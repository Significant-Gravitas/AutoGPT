import * as XLSX from 'xlsx';
import type { DealState, AnalysisResults } from '../types/deal';
import { fmtCurrency, fmtPct, fmtMultiple } from './calculations';

export function exportExcel(deal: DealState, results: AnalysisResults) {
  const wb = XLSX.utils.book_new();

  // --- Summary sheet ---
  const m = results.metrics;
  const wf = results.waterfall;
  const summaryData = [
    ['CRE Deal Analyzer — Deal Summary'],
    [],
    ['Property', deal.property_info.name],
    ['Address', deal.property_info.address],
    ['Units', deal.property_info.units],
    ['Purchase Price', m.purchase_price],
    ['Loan Amount', m.loan_amount],
    ['Equity Required', m.equity_required],
    [],
    ['--- Return Metrics ---'],
    ['Going-in Cap Rate', fmtPct(m.going_in_cap_rate)],
    ['Stabilized Cap Rate', fmtPct(m.stabilized_cap_rate)],
    ['Year 1 NOI', fmtCurrency(m.year1_noi)],
    ['DSCR (Year 1)', fmtNum(m.dscr_year1)],
    ['Levered CoC', fmtPct(m.levered_coc)],
    ['Levered IRR', fmtPct(m.levered_irr)],
    ['Unlevered IRR', fmtPct(m.unlevered_irr)],
    ['Levered EM', fmtMultiple(m.levered_em)],
    ['Unlevered EM', fmtMultiple(m.unlevered_em)],
    [],
    ['--- Waterfall ---'],
    ['LP IRR', fmtPct(wf.lp_irr)],
    ['GP IRR', fmtPct(wf.gp_irr)],
    ['LP EM', fmtMultiple(wf.lp_em)],
    ['GP EM', fmtMultiple(wf.gp_em)],
    ['LP Total Distributions', fmtCurrency(wf.lp_total_distributions)],
    ['GP Total Distributions', fmtCurrency(wf.gp_total_distributions)],
    ['GP Promote Earned', fmtCurrency(wf.gp_promote_earned)],
  ];
  const ws1 = XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(wb, ws1, 'Summary');

  // --- Pro Forma sheet ---
  const pf = results.proforma;
  const pfHeaders = [
    'Year', 'GPR', 'Vacancy', 'Credit Loss', 'Other Income', 'EGI',
    'Taxes', 'Insurance', 'Mgmt Fee', 'Maintenance', 'Utilities',
    'Payroll', 'G&A', 'Marketing', 'CapEx', 'Total Exp', 'NOI',
    'Debt Service', 'Levered CF', 'DSCR',
  ];
  const pfRows = pf.annual_rows.map((r) => [
    r.year, r.gross_potential_rent, r.vacancy_loss, r.credit_loss, r.other_income, r.egi,
    r.property_taxes, r.insurance, r.management_fee, r.maintenance, r.utilities,
    r.payroll, r.general_admin, r.marketing, r.capex_reserves, r.total_expenses, r.noi,
    r.debt_service, r.levered_cf, r.dscr,
  ]);
  const ws2 = XLSX.utils.aoa_to_sheet([pfHeaders, ...pfRows]);
  XLSX.utils.book_append_sheet(wb, ws2, 'Pro Forma');

  // --- Waterfall sheet ---
  const wfHeaders = [
    'Year', 'Distributable', 'LP Dist', 'GP Dist',
    'RoC LP', 'RoC GP', 'Pref Return', 'GP Catch-up', 'LP Promote', 'GP Promote',
  ];
  const wfRows = wf.yearly.map((r) => [
    r.is_exit ? `Year ${r.year} (Exit)` : `Year ${r.year}`,
    r.distributable, r.lp_distribution, r.gp_distribution,
    r.tier_roc_lp, r.tier_roc_gp, r.tier_preferred_return,
    r.tier_gp_catchup, r.tier_lp_promote, r.tier_gp_promote,
  ]);
  const ws3 = XLSX.utils.aoa_to_sheet([wfHeaders, ...wfRows]);
  XLSX.utils.book_append_sheet(wb, ws3, 'Waterfall');

  // --- Rent Roll sheet ---
  if (deal.rent_roll.length > 0) {
    const rrHeaders = ['Unit', 'Type', 'SqFt', 'Market Rent', 'Current Rent', 'Lease Start', 'Lease End', 'Status'];
    const rrRows = deal.rent_roll.map((u) => [
      u.unit_number, u.unit_type, u.sqft, u.market_rent, u.current_rent,
      u.lease_start, u.lease_end, u.status,
    ]);
    const ws4 = XLSX.utils.aoa_to_sheet([rrHeaders, ...rrRows]);
    XLSX.utils.book_append_sheet(wb, ws4, 'Rent Roll');
  }

  XLSX.writeFile(wb, `${deal.name || 'cre-deal'}.xlsx`);
}

function fmtNum(n: number | null | undefined, d = 2): string {
  if (n == null) return '—';
  return n.toFixed(d);
}

export function exportDealJson(deal: DealState) {
  const blob = new Blob([JSON.stringify(deal, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${deal.name || 'deal'}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
