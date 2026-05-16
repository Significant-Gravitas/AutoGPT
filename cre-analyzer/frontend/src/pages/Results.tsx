import React, { useEffect, useState } from 'react';
import {
  Download, FileSpreadsheet, BarChart3, RefreshCw,
  TrendingUp, DollarSign, Building, Target, AlertCircle,
} from 'lucide-react';
import { Card, MetricCard } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { CashFlowChart, NOIGrowthChart } from '../components/charts/CashFlowChart';
import { WaterfallDistributionChart, LPGPSplitChart } from '../components/charts/WaterfallChart';
import { SensitivityHeatmap } from '../components/charts/SensitivityHeatmap';
import { CashFlowTable } from '../components/tables/CashFlowTable';
import { WaterfallTable } from '../components/tables/WaterfallTable';
import { RentRollTable } from '../components/tables/RentRollTable';
import { useDealStore } from '../store/dealStore';
import { runAnalysis, runSensitivity, solveDeal } from '../api/client';
import { exportExcel, exportDealJson } from '../utils/export';
import { fmtCurrency, fmtPct, fmtMultiple, fmtNum } from '../utils/calculations';
import type { SensitivityResults } from '../types/deal';

type ResultTab = 'overview' | 'proforma' | 'waterfall' | 'sensitivity' | 'rentroll';

export function ResultsPage() {
  const { deal, setResults, setSensitivity, isLoading, isSensLoading, setLoading, setSensLoading, setError, error, sensitivity } = useDealStore();
  const [tab, setTab] = useState<ResultTab>('overview');
  const [solveResult, setSolveResult] = useState<{ value: number; label: string } | null>(null);
  const [targetIrr, setTargetIrr] = useState(14);
  const [solveFor, setSolveFor] = useState<'purchase_price' | 'exit_cap_rate'>('purchase_price');

  const results = deal.results;

  useEffect(() => {
    if (!results) handleRunAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRunAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runAnalysis(deal);
      setResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSensitivity = async () => {
    setSensLoading(true);
    try {
      const res = await runSensitivity(deal);
      setSensitivity(res as SensitivityResults);
      setTab('sensitivity');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Sensitivity failed');
    } finally {
      setSensLoading(false);
    }
  };

  const handleSolve = async () => {
    try {
      const res = await solveDeal(deal, targetIrr, solveFor);
      setSolveResult({
        value: res.value,
        label: solveFor === 'purchase_price'
          ? `Max Purchase Price for ${targetIrr}% LP IRR: ${fmtCurrency(res.value)}`
          : `Min Exit Cap Rate for ${targetIrr}% LP IRR: ${res.value.toFixed(2)}%`,
      });
    } catch { /* ignore */ }
  };

  const tabs: { id: ResultTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <BarChart3 size={14} /> },
    { id: 'proforma', label: '10-Year Pro Forma', icon: <TrendingUp size={14} /> },
    { id: 'waterfall', label: 'LP/GP Waterfall', icon: <DollarSign size={14} /> },
    { id: 'sensitivity', label: 'Sensitivity', icon: <Target size={14} /> },
    { id: 'rentroll', label: 'Rent Roll', icon: <Building size={14} /> },
  ];

  return (
    <div className="space-y-6 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">{deal.property_info.name}</h2>
          <p className="text-sm text-gray-500">{deal.property_info.address} · {deal.property_info.units} units · {fmtCurrency(deal.property_info.purchase_price, true)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={handleRunAnalysis} loading={isLoading} icon={<RefreshCw size={14} />}>
            Recalculate
          </Button>
          <Button variant="secondary" onClick={handleRunSensitivity} loading={isSensLoading} icon={<Target size={14} />}>
            Run Sensitivity
          </Button>
          {results && (
            <>
              <Button variant="secondary" onClick={() => exportExcel(deal, results)} icon={<FileSpreadsheet size={14} />}>
                Export Excel
              </Button>
              <Button variant="secondary" onClick={() => exportDealJson(deal)} icon={<Download size={14} />}>
                Save JSON
              </Button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 text-sm text-red-700 dark:text-red-400">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <div>
            <strong>Error:</strong> {error}
            <p className="text-xs mt-1 opacity-80">Make sure the backend is running on port 8000 and ANTHROPIC_API_KEY is set.</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="text-center py-16 text-gray-400">
          <RefreshCw size={32} className="animate-spin mx-auto mb-3" />
          <p>Running analysis...</p>
        </div>
      )}

      {results && !isLoading && (
        <>
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            <MetricCard label="LP IRR" value={fmtPct(results.lp_irr)} highlight={results.lp_irr != null && results.lp_irr >= 14 ? 'green' : results.lp_irr != null && results.lp_irr >= 10 ? 'yellow' : 'red'} />
            <MetricCard label="Levered IRR" value={fmtPct(results.levered_irr)} highlight="blue" />
            <MetricCard label="Unlev IRR" value={fmtPct(results.metrics.unlevered_irr)} />
            <MetricCard label="LP EM" value={fmtMultiple(results.waterfall.lp_em)} highlight="green" />
            <MetricCard label="Levered EM" value={fmtMultiple(results.levered_em)} />
            <MetricCard label="CoC (Yr 1)" value={fmtPct(results.levered_coc)} />
            <MetricCard label="Going-in Cap" value={fmtPct(results.metrics.going_in_cap_rate)} />
            <MetricCard label="DSCR (Yr 1)" value={fmtNum(results.metrics.dscr_year1)} highlight={results.metrics.dscr_year1 != null && results.metrics.dscr_year1 >= 1.25 ? 'green' : 'yellow'} />
          </div>

          {/* Tabs */}
          <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
            {tabs.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition whitespace-nowrap ${
                  tab === t.id ? 'border-brand-500 text-brand-600 dark:text-brand-400' : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          {/* Overview Tab */}
          {tab === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card title="10-Year Cash Flow" padding>
                  <CashFlowChart rows={results.proforma.annual_rows} />
                </Card>
                <Card title="NOI Growth" padding>
                  <NOIGrowthChart rows={results.proforma.annual_rows} />
                </Card>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card title="Deal Summary" padding>
                  <dl className="space-y-2 text-sm">
                    {[
                      ['Purchase Price', fmtCurrency(results.metrics.purchase_price)],
                      ['Loan Amount', fmtCurrency(results.metrics.loan_amount)],
                      ['Equity Required', fmtCurrency(results.metrics.equity_required)],
                      ['Year 1 NOI', fmtCurrency(results.metrics.year1_noi)],
                      ['Year 1 EGI', fmtCurrency(results.metrics.year1_egi)],
                      ['NOI Margin', fmtPct(results.metrics.noi_margin)],
                      ['Going-in Cap', fmtPct(results.metrics.going_in_cap_rate)],
                      ['Stabilized Cap', fmtPct(results.metrics.stabilized_cap_rate)],
                      ['NPV @ 10%', fmtCurrency(results.metrics.npv_10pct)],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1 border-b border-gray-100 dark:border-gray-700">
                        <dt className="text-gray-500">{k}</dt>
                        <dd className="font-medium">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </Card>

                <Card title="Exit / Disposition" padding>
                  <dl className="space-y-2 text-sm">
                    {[
                      ['Exit Cap Rate', fmtPct(results.proforma.exit.exit_cap_rate)],
                      ['Exit Year NOI', fmtCurrency(results.proforma.exit.exit_year_noi)],
                      ['Gross Sale Price', fmtCurrency(results.proforma.exit.gross_sale_price)],
                      ['Selling Costs', `(${fmtCurrency(results.proforma.exit.selling_costs)})`],
                      ['Net Sale Price', fmtCurrency(results.proforma.exit.net_sale_price)],
                      ['Loan Payoff', `(${fmtCurrency(results.proforma.exit.loan_payoff)})`],
                      ['Net Proceeds to Equity', fmtCurrency(results.proforma.exit.net_sale_proceeds)],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1 border-b border-gray-100 dark:border-gray-700">
                        <dt className="text-gray-500">{k}</dt>
                        <dd className="font-medium">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </Card>

                <Card title="Waterfall Summary" padding>
                  <dl className="space-y-2 text-sm">
                    {[
                      ['LP Invested', fmtCurrency(results.waterfall.lp_invested)],
                      ['LP Total Distributions', fmtCurrency(results.waterfall.lp_total_distributions)],
                      ['LP IRR', fmtPct(results.waterfall.lp_irr)],
                      ['LP Equity Multiple', fmtMultiple(results.waterfall.lp_em)],
                      ['GP Invested', fmtCurrency(results.waterfall.gp_invested)],
                      ['GP Total Distributions', fmtCurrency(results.waterfall.gp_total_distributions)],
                      ['GP IRR', fmtPct(results.waterfall.gp_irr)],
                      ['GP Equity Multiple', fmtMultiple(results.waterfall.gp_em)],
                      ['GP Promote Earned', fmtCurrency(results.waterfall.gp_promote_earned)],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1 border-b border-gray-100 dark:border-gray-700">
                        <dt className="text-gray-500">{k}</dt>
                        <dd className="font-medium">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </Card>
              </div>

              {/* Solve Mode */}
              <Card title="Solve Mode" subtitle="Back-solve for target LP IRR" padding>
                <div className="flex flex-wrap gap-3 items-end">
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Target LP IRR (%)</label>
                    <input type="number" className="input-field w-28" value={targetIrr} onChange={(e) => setTargetIrr(+e.target.value)} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Solve For</label>
                    <select className="input-field" value={solveFor} onChange={(e) => setSolveFor(e.target.value as 'purchase_price' | 'exit_cap_rate')}>
                      <option value="purchase_price">Max Purchase Price</option>
                      <option value="exit_cap_rate">Min Exit Cap Rate</option>
                    </select>
                  </div>
                  <Button onClick={handleSolve} icon={<Target size={14} />}>Solve</Button>
                  {solveResult && (
                    <div className="px-4 py-2 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-sm text-emerald-700 dark:text-emerald-400 font-medium">
                      {solveResult.label}
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}

          {/* Pro Forma Tab */}
          {tab === 'proforma' && (
            <div className="space-y-6">
              <Card title="10-Year Cash Flow Model" subtitle="Click section headers to collapse" padding={false}>
                <CashFlowTable rows={results.proforma.annual_rows} exit={results.proforma.exit} />
              </Card>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card title="NOI & Cash Flow by Year" padding>
                  <CashFlowChart rows={results.proforma.annual_rows} />
                </Card>
                <Card title="Revenue Growth" padding>
                  <NOIGrowthChart rows={results.proforma.annual_rows} />
                </Card>
              </div>
            </div>
          )}

          {/* Waterfall Tab */}
          {tab === 'waterfall' && (
            <div className="space-y-6">
              <Card title="Waterfall Distribution Table" subtitle="Year-by-year cash flow through each tier" padding={false}>
                <div className="p-4">
                  <WaterfallTable years={results.waterfall.yearly} />
                </div>
              </Card>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card title="Distributions by Tier" padding>
                  <WaterfallDistributionChart years={results.waterfall.yearly} />
                </Card>
                <Card title="LP vs GP Distributions" padding>
                  <LPGPSplitChart years={results.waterfall.yearly} />
                </Card>
              </div>
            </div>
          )}

          {/* Sensitivity Tab */}
          {tab === 'sensitivity' && (
            <div className="space-y-6">
              {!sensitivity ? (
                <div className="text-center py-16">
                  <p className="text-gray-400 mb-4">Click "Run Sensitivity" to generate heatmap tables.</p>
                  <Button onClick={handleRunSensitivity} loading={isSensLoading} icon={<Target size={14} />}>
                    Run Sensitivity Analysis
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                  <Card title="LP IRR vs. Purchase Price × Exit Cap Rate" padding>
                    <SensitivityHeatmap
                      table={sensitivity.lp_irr_vs_price_x_exit_cap}
                      title=""
                      rowLabel="Purchase Price"
                      colLabel="Exit Cap Rate"
                      formatRow={(v) => `$${(v / 1_000_000).toFixed(1)}M`}
                      formatCol={(v) => `${v.toFixed(2)}%`}
                      formatCell={(v) => `${v.toFixed(1)}%`}
                      thresholds={{ green: 14, yellow: 10 }}
                    />
                  </Card>
                  <Card title="LP IRR vs. Rent Growth × Vacancy" padding>
                    <SensitivityHeatmap
                      table={sensitivity.lp_irr_vs_rent_x_vacancy}
                      title=""
                      rowLabel="Rent Growth"
                      colLabel="Vacancy"
                      formatRow={(v) => `${v.toFixed(1)}%`}
                      formatCol={(v) => `${v.toFixed(0)}%`}
                      formatCell={(v) => `${v.toFixed(1)}%`}
                      thresholds={{ green: 14, yellow: 10 }}
                    />
                  </Card>
                  <Card title="Cash-on-Cash vs. LTV × Interest Rate" padding>
                    <SensitivityHeatmap
                      table={sensitivity.coc_vs_ltv_x_rate}
                      title=""
                      rowLabel="LTV"
                      colLabel="Interest Rate"
                      formatRow={(v) => `${v.toFixed(0)}%`}
                      formatCol={(v) => `${v.toFixed(1)}%`}
                      formatCell={(v) => `${v.toFixed(1)}%`}
                      thresholds={{ green: 7, yellow: 4 }}
                    />
                  </Card>
                  <Card title="Equity Multiple vs. Hold Period × Exit Cap" padding>
                    <SensitivityHeatmap
                      table={sensitivity.em_vs_hold_x_exit_cap}
                      title=""
                      rowLabel="Hold Period"
                      colLabel="Exit Cap Rate"
                      formatRow={(v) => `${v}yr`}
                      formatCol={(v) => `${v.toFixed(2)}%`}
                      formatCell={(v) => `${v.toFixed(2)}x`}
                      thresholds={{ green: 1.8, yellow: 1.4 }}
                    />
                  </Card>
                </div>
              )}
            </div>
          )}

          {/* Rent Roll Tab */}
          {tab === 'rentroll' && (
            <Card title="Rent Roll" subtitle={`${deal.rent_roll.length} units`} padding>
              {deal.rent_roll.length > 0 ? (
                <RentRollTable units={deal.rent_roll} />
              ) : (
                <div className="text-center py-8 text-gray-400 text-sm">No rent roll data. Upload a rent roll document in the Upload step.</div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
