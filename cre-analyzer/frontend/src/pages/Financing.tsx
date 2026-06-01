import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Input, Select, Toggle } from '../components/ui/Input';
import { NumericInput } from '../components/ui/NumericInput';
import { Button } from '../components/ui/Button';
import { useDealStore } from '../store/dealStore';
import { fmtCurrency } from '../utils/calculations';
import type { FinancingAssumptions } from '../types/deal';

export function FinancingPage() {
  const { deal, updateDeal, setStep } = useDealStore();
  const f = deal.financing;
  const pp = deal.property_info.purchase_price;

  const upF = <K extends keyof FinancingAssumptions>(k: K, v: FinancingAssumptions[K]) =>
    updateDeal({ financing: { ...f, [k]: v } });

  const loanAmt = pp * f.ltv_pct / 100;
  const equity = pp - loanAmt;
  const ioPayment = loanAmt * f.interest_rate / 100;
  const r = f.interest_rate / 100 / 12;
  const n = f.amortization_years * 12;
  const monthlyAm = r > 0 ? loanAmt * (r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1) : loanAmt / (f.amortization_years * 12);
  const amPayment = monthlyAm * 12;

  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Financing Structure</h2>
          <p className="text-sm text-gray-500 mt-1">Configure loan terms and debt service schedule.</p>
        </div>
        <Button onClick={() => setStep('waterfall')} icon={<ArrowRight size={14} />}>Continue to Waterfall</Button>
      </div>

      {/* Quick summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Purchase Price', value: fmtCurrency(pp) },
          { label: 'Loan Amount', value: fmtCurrency(loanAmt) },
          { label: 'Equity Required', value: fmtCurrency(equity) },
          { label: 'I/O Debt Service / yr', value: fmtCurrency(ioPayment) },
        ].map((m) => (
          <div key={m.label} className="card p-3">
            <div className="label">{m.label}</div>
            <div className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{m.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Loan Terms" padding>
          <div className="space-y-4">
            <Select label="Loan Type" value={f.loan_type}
              options={[{ value: 'Agency', label: 'Agency (Fannie/Freddie)' }, { value: 'Bridge', label: 'Bridge' }, { value: 'CMBS', label: 'CMBS' }]}
              onChange={(e) => upF('loan_type', e.target.value)} />
            <Input label="LTV (%)" type="number" suffix="%" value={f.ltv_pct}
              onChange={(e) => upF('ltv_pct', +e.target.value)}
              hint={`Loan: ${fmtCurrency(loanAmt)} | Equity: ${fmtCurrency(equity)}`} />
            <Input label="Interest Rate (%)" type="number" suffix="%" value={f.interest_rate}
              onChange={(e) => upF('interest_rate', +e.target.value)} />
            <Input label="I/O Period (years)" type="number" value={f.io_period_years}
              onChange={(e) => upF('io_period_years', +e.target.value)}
              hint={`I/O payment: ${fmtCurrency(ioPayment)}/yr`} />
            <Input label="Amortization (years)" type="number" value={f.amortization_years}
              onChange={(e) => upF('amortization_years', +e.target.value)}
              hint={`Am payment: ${fmtCurrency(amPayment)}/yr`} />
            <Input label="Loan Term (years)" type="number" value={f.loan_term_years}
              onChange={(e) => upF('loan_term_years', +e.target.value)} />
          </div>
        </Card>

        <Card title="Refi / Cash-Out Scenario" padding>
          <div className="space-y-4">
            <Toggle label="Enable Refi Scenario" checked={f.enable_refi} onChange={(v) => upF('enable_refi', v)}
              hint="Model an optional cash-out refinance at stabilization" />
            {f.enable_refi && (
              <div className="space-y-4 pt-3 border-t border-gray-100 dark:border-gray-700">
                <Input label="Refi Year" type="number" value={f.refi_year}
                  onChange={(e) => upF('refi_year', +e.target.value)} />
                <Input label="Refi LTV (%)" type="number" suffix="%" value={f.refi_ltv_pct}
                  onChange={(e) => upF('refi_ltv_pct', +e.target.value)} />
                <Input label="Refi Rate (%)" type="number" suffix="%" value={f.refi_rate}
                  onChange={(e) => upF('refi_rate', +e.target.value)} />
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Debt service table preview */}
      <Card title="Debt Service Preview" padding={false}>
        <div className="overflow-x-auto p-4">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr>
                {['Year', 'Type', 'Annual Payment', 'Interest', 'Principal'].map((h) => (
                  <th key={h} className="table-header">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: Math.min(deal.exit_assumptions.hold_period_years, 10) }, (_, i) => {
                const yr = i + 1;
                const isIO = yr <= f.io_period_years;
                const pmt = isIO ? ioPayment : amPayment;
                const interest = loanAmt * f.interest_rate / 100;
                const principal = isIO ? 0 : pmt - interest;
                return (
                  <tr key={yr} className={i % 2 === 0 ? 'table-row-alt' : ''}>
                    <td className="table-cell text-left pl-3">Year {yr}</td>
                    <td className="table-cell text-left">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${isIO ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'}`}>
                        {isIO ? 'I/O' : 'Am'}
                      </span>
                    </td>
                    <td className="table-cell font-medium">{fmtCurrency(pmt)}</td>
                    <td className="table-cell text-red-500">{fmtCurrency(interest)}</td>
                    <td className="table-cell">{fmtCurrency(principal > 0 ? principal : 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
