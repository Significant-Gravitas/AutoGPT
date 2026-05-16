import React, { useState } from 'react';
import { Card } from '../components/ui/Card';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { RentRollTable } from '../components/tables/RentRollTable';
import { useDealStore } from '../store/dealStore';
import { fmtCurrency } from '../utils/calculations';
import type { T12Data, PropertyInfo, RentRollUnit } from '../types/deal';
import { ArrowRight, Building2, DollarSign, Users } from 'lucide-react';

type Tab = 'property' | 't12' | 'rentroll';

export function ReviewDataPage() {
  const { deal, updateDeal, setStep } = useDealStore();
  const [tab, setTab] = useState<Tab>('property');

  const updateProp = <K extends keyof PropertyInfo>(k: K, v: PropertyInfo[K]) =>
    updateDeal({ property_info: { ...deal.property_info, [k]: v } });

  const updateT12 = <K extends keyof T12Data>(k: K, v: T12Data[K]) =>
    updateDeal({ t12_data: { ...deal.t12_data, [k]: v } });

  const updateUnit = (idx: number, field: keyof RentRollUnit, value: string | number) => {
    const rr = [...deal.rent_roll];
    rr[idx] = { ...rr[idx], [field]: value };
    updateDeal({ rent_roll: rr });
  };

  const egi =
    deal.t12_data.gross_potential_rent -
    deal.t12_data.vacancy_loss -
    deal.t12_data.concessions -
    deal.t12_data.bad_debt +
    deal.t12_data.other_income;

  const totalExp =
    deal.t12_data.property_taxes +
    deal.t12_data.insurance +
    deal.t12_data.management_fee +
    deal.t12_data.maintenance_repairs +
    deal.t12_data.utilities +
    deal.t12_data.payroll +
    deal.t12_data.general_admin +
    deal.t12_data.marketing +
    deal.t12_data.capex_reserves +
    deal.t12_data.other_expenses;

  const noi = egi - totalExp;
  const goingIn = deal.property_info.purchase_price > 0 ? (noi / deal.property_info.purchase_price) * 100 : 0;

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'property', label: 'Property Info', icon: <Building2 size={14} /> },
    { id: 't12', label: 'T12 Financials', icon: <DollarSign size={14} /> },
    { id: 'rentroll', label: `Rent Roll (${deal.rent_roll.length})`, icon: <Users size={14} /> },
  ];

  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Review Extracted Data</h2>
          <p className="text-sm text-gray-500 mt-1">Edit any values extracted from your documents. Live NOI: <strong className="text-gray-800 dark:text-gray-200">{fmtCurrency(noi)}</strong> ({goingIn.toFixed(2)}% cap)</p>
        </div>
        <Button onClick={() => setStep('assumptions')} icon={<ArrowRight size={14} />}>
          Continue to Assumptions
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition ${
              tab === t.id
                ? 'border-brand-500 text-brand-600 dark:text-brand-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {tab === 'property' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Input label="Property Name" value={deal.property_info.name} onChange={(e) => updateProp('name', e.target.value)} />
          <Input label="Address" value={deal.property_info.address} onChange={(e) => updateProp('address', e.target.value)} />
          <Input label="Market" value={deal.property_info.market} onChange={(e) => updateProp('market', e.target.value)} />
          <Input label="Asset Type" value={deal.property_info.asset_type} onChange={(e) => updateProp('asset_type', e.target.value)} />
          <Input label="Total Units" type="number" value={deal.property_info.units} onChange={(e) => updateProp('units', +e.target.value)} />
          <Input label="Total SqFt" type="number" value={deal.property_info.sqft} onChange={(e) => updateProp('sqft', +e.target.value)} />
          <Input label="Year Built" type="number" value={deal.property_info.year_built} onChange={(e) => updateProp('year_built', +e.target.value)} />
          <Input label="Purchase Price" type="number" prefix="$" value={deal.property_info.purchase_price} onChange={(e) => updateProp('purchase_price', +e.target.value)} />
          <Input label="Asking Price" type="number" prefix="$" value={deal.property_info.asking_price} onChange={(e) => updateProp('asking_price', +e.target.value)} />
        </div>
      )}

      {tab === 't12' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Income" padding>
            <div className="space-y-3">
              {(
                [
                  ['gross_potential_rent', 'Gross Potential Rent'],
                  ['vacancy_loss', 'Vacancy Loss'],
                  ['concessions', 'Concessions'],
                  ['bad_debt', 'Bad Debt'],
                  ['other_income', 'Other Income'],
                ] as [keyof T12Data, string][]
              ).map(([k, label]) => (
                <Input key={k} label={label} type="number" prefix="$" value={deal.t12_data[k] as number}
                  onChange={(e) => updateT12(k, +e.target.value)} />
              ))}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-600">
                <div className="flex justify-between text-sm">
                  <span className="font-semibold text-gray-700 dark:text-gray-300">EGI</span>
                  <span className="font-bold">{fmtCurrency(egi)}</span>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Expenses" padding>
            <div className="space-y-3">
              {(
                [
                  ['property_taxes', 'Property Taxes'],
                  ['insurance', 'Insurance'],
                  ['management_fee', 'Management Fee'],
                  ['maintenance_repairs', 'Maintenance & Repairs'],
                  ['utilities', 'Utilities'],
                  ['payroll', 'Payroll'],
                  ['general_admin', 'General & Admin'],
                  ['marketing', 'Marketing'],
                  ['capex_reserves', 'CapEx Reserves'],
                  ['other_expenses', 'Other Expenses'],
                ] as [keyof T12Data, string][]
              ).map(([k, label]) => (
                <Input key={k} label={label} type="number" prefix="$" value={deal.t12_data[k] as number}
                  onChange={(e) => updateT12(k, +e.target.value)} />
              ))}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-600 space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Total OpEx</span>
                  <span className="font-medium text-red-500">({fmtCurrency(totalExp)})</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="font-semibold text-gray-700 dark:text-gray-300">NOI</span>
                  <span className={`font-bold ${noi > 0 ? 'text-emerald-600' : 'text-red-500'}`}>{fmtCurrency(noi)}</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {tab === 'rentroll' && (
        <Card title="Rent Roll" subtitle="Unit-by-unit lease data" padding={false}>
          {deal.rent_roll.length > 0 ? (
            <div className="p-4">
              <RentRollTable units={deal.rent_roll} onEdit={updateUnit} />
            </div>
          ) : (
            <div className="p-8 text-center text-gray-400 text-sm">
              No rent roll data extracted. Upload a rent roll document or add units manually.
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
