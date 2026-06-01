import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Input, Toggle } from '../components/ui/Input';
import { NumericInput } from '../components/ui/NumericInput';
import { Button } from '../components/ui/Button';
import { useDealStore } from '../store/dealStore';
import type { Assumptions, ValueAddAssumptions } from '../types/deal';

export function AssumptionsPage() {
  const { deal, updateDeal, setStep } = useDealStore();
  const a = deal.assumptions;
  const va = a.value_add;

  const upA = <K extends keyof Assumptions>(k: K, v: Assumptions[K]) =>
    updateDeal({ assumptions: { ...a, [k]: v } });

  const upVA = <K extends keyof ValueAddAssumptions>(k: K, v: ValueAddAssumptions[K]) =>
    updateDeal({ assumptions: { ...a, value_add: { ...va, [k]: v } } });

  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Underwriting Assumptions</h2>
          <p className="text-sm text-gray-500 mt-1">Set growth rates, vacancy, and value-add parameters.</p>
        </div>
        <Button onClick={() => setStep('financing')} icon={<ArrowRight size={14} />}>Continue to Financing</Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Revenue Assumptions" padding>
          <div className="space-y-4">
            <Input label="Rent Growth Rate (%/yr)" type="number" suffix="%" value={a.rent_growth_rate}
              onChange={(e) => upA('rent_growth_rate', +e.target.value)} hint="Annual market rent growth" />
            <Input label="Vacancy Rate (%)" type="number" suffix="%" value={a.vacancy_rate}
              onChange={(e) => upA('vacancy_rate', +e.target.value)} hint="Physical vacancy assumption" />
            <Input label="Credit Loss (%)" type="number" suffix="%" value={a.credit_loss_rate}
              onChange={(e) => upA('credit_loss_rate', +e.target.value)} hint="Bad debt / collection loss" />
            <Input label="Other Income Growth (%/yr)" type="number" suffix="%" value={a.other_income_growth}
              onChange={(e) => upA('other_income_growth', +e.target.value)} />
          </div>
        </Card>

        <Card title="Expense Assumptions" padding>
          <div className="space-y-4">
            <Input label="Expense Growth Rate (%/yr)" type="number" suffix="%" value={a.expense_growth_rate}
              onChange={(e) => upA('expense_growth_rate', +e.target.value)} />
            <Input label="Management Fee (% of EGI)" type="number" suffix="%" value={a.management_fee_pct}
              onChange={(e) => upA('management_fee_pct', +e.target.value)} hint="Replaces T12 management fee" />
            <NumericInput label="CapEx Reserves ($/unit/yr)" prefix="$" value={a.capex_reserves_per_unit}
              onChange={(v) => upA('capex_reserves_per_unit', v)} />
          </div>
        </Card>
      </div>

      <Card title="Value-Add Underwriting" padding>
        <div className="space-y-4">
          <Toggle
            label="Enable Value-Add Scenario"
            checked={va.enabled}
            onChange={(v) => upVA('enabled', v)}
            hint="Model unit renovation program with rent premiums"
          />
          {va.enabled && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-2 pt-4 border-t border-gray-100 dark:border-gray-700">
              <Input label="Units to Renovate" type="number" value={va.units_to_renovate}
                onChange={(e) => upVA('units_to_renovate', +e.target.value)}
                hint={`of ${deal.property_info.units} total`} />
              <NumericInput label="Reno Cost / Unit" prefix="$" value={va.renovation_cost_per_unit}
                onChange={(v) => upVA('renovation_cost_per_unit', v)} />
              <NumericInput label="Rent Premium / Unit / Mo" prefix="$" value={va.rent_premium_per_unit}
                onChange={(v) => upVA('rent_premium_per_unit', v)} />
              <Input label="Absorption Period (yrs)" type="number" value={va.absorption_years}
                onChange={(e) => upVA('absorption_years', +e.target.value)} />
            </div>
          )}
        </div>
      </Card>

      <Card title="Exit Assumptions" padding>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Input label="Hold Period (years)" type="number" value={deal.exit_assumptions.hold_period_years}
            onChange={(e) => updateDeal({ exit_assumptions: { ...deal.exit_assumptions, hold_period_years: +e.target.value } })} />
          <Input label="Exit Cap Rate (%)" type="number" suffix="%" value={deal.exit_assumptions.exit_cap_rate}
            onChange={(e) => updateDeal({ exit_assumptions: { ...deal.exit_assumptions, exit_cap_rate: +e.target.value } })} />
          <Input label="Selling Costs (%)" type="number" suffix="%" value={deal.exit_assumptions.selling_costs_pct}
            onChange={(e) => updateDeal({ exit_assumptions: { ...deal.exit_assumptions, selling_costs_pct: +e.target.value } })} />
        </div>
      </Card>
    </div>
  );
}
