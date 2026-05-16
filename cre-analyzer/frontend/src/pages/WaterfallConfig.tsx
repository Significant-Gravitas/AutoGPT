import React from 'react';
import { ArrowRight, Plus, Trash2 } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Input, Toggle } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { useDealStore } from '../store/dealStore';
import type { WaterfallConfig, WaterfallTier } from '../types/deal';

export function WaterfallConfigPage() {
  const { deal, updateDeal, setStep } = useDealStore();
  const wf = deal.waterfall_config;

  const upWF = <K extends keyof WaterfallConfig>(k: K, v: WaterfallConfig[K]) =>
    updateDeal({ waterfall_config: { ...wf, [k]: v } });

  const updateTier = (i: number, field: keyof WaterfallTier, value: number) => {
    const tiers = wf.tiers.map((t, idx) => idx === i ? { ...t, [field]: value } : t);
    upWF('tiers', tiers);
  };

  const addTier = () => {
    const last = wf.tiers[wf.tiers.length - 1];
    upWF('tiers', [
      ...wf.tiers.slice(0, -1),
      { ...last, irr_max: last.irr_min + 4, lp_split: 70, gp_split: 30 },
      { irr_min: last.irr_min + 4, irr_max: 999, lp_split: 60, gp_split: 40 },
    ]);
  };

  const removeTier = (i: number) => {
    if (wf.tiers.length <= 1) return;
    upWF('tiers', wf.tiers.filter((_, idx) => idx !== i));
  };

  const equity = deal.property_info.purchase_price * (1 - deal.financing.ltv_pct / 100);
  const lpEquity = equity * wf.lp_equity_pct / 100;
  const gpEquity = equity * wf.gp_equity_pct / 100;

  return (
    <div className="space-y-6 py-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">LP/GP Waterfall Structure</h2>
          <p className="text-sm text-gray-500 mt-1">Configure equity split, preferred return, catch-up, and promote tiers.</p>
        </div>
        <Button onClick={() => setStep('results')} icon={<ArrowRight size={14} />}>Run Analysis</Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Equity Split" padding>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Input label="LP Equity (%)" type="number" suffix="%" value={wf.lp_equity_pct}
                onChange={(e) => { const v = +e.target.value; upWF('lp_equity_pct', v); upWF('gp_equity_pct', 100 - v); }}
                hint={`LP invests ~$${(lpEquity / 1_000_000).toFixed(1)}M`} />
              <Input label="GP Equity (%)" type="number" suffix="%" value={wf.gp_equity_pct}
                onChange={(e) => { const v = +e.target.value; upWF('gp_equity_pct', v); upWF('lp_equity_pct', 100 - v); }}
                hint={`GP invests ~$${(gpEquity / 1_000_000).toFixed(1)}M`} />
            </div>
            <div className="w-full h-4 rounded-full bg-gray-100 dark:bg-gray-700 overflow-hidden flex">
              <div className="h-full bg-brand-500 transition-all" style={{ width: `${wf.lp_equity_pct}%` }} />
              <div className="h-full bg-purple-500 transition-all" style={{ width: `${wf.gp_equity_pct}%` }} />
            </div>
            <div className="flex gap-4 text-xs text-gray-500">
              <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-sm bg-brand-500" />LP {wf.lp_equity_pct}%</div>
              <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-sm bg-purple-500" />GP {wf.gp_equity_pct}%</div>
            </div>
          </div>
        </Card>

        <Card title="Preferred Return & Catch-up" padding>
          <div className="space-y-4">
            <Input label="Preferred Return (%/yr)" type="number" suffix="%" value={wf.preferred_return}
              onChange={(e) => upWF('preferred_return', +e.target.value)}
              hint="Annual preferred return to LP on invested capital" />
            <Toggle label="Compounding Pref" checked={wf.pref_compounding} onChange={(v) => upWF('pref_compounding', v)}
              hint="Unchecked = simple (non-compounding)" />
            <Toggle label="GP Catch-up" checked={wf.gp_catchup} onChange={(v) => upWF('gp_catchup', v)}
              hint="GP receives catch-up distribution after LP pref" />
            {wf.gp_catchup && (
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-100 dark:border-gray-700">
                <Input label="Catch-up Rate (% to GP)" type="number" suffix="%" value={wf.gp_catchup_rate}
                  onChange={(e) => upWF('gp_catchup_rate', +e.target.value)}
                  hint="% of distributions to GP during catch-up" />
                <Input label="GP Promote Target (%)" type="number" suffix="%" value={wf.gp_target_promote_pct}
                  onChange={(e) => upWF('gp_target_promote_pct', +e.target.value)}
                  hint="GP's target % of total profits" />
              </div>
            )}
          </div>
        </Card>
      </div>

      <Card
        title="Promote Tiers"
        subtitle="IRR-based profit splits after preferred return & catch-up"
        action={
          <Button variant="secondary" size="sm" icon={<Plus size={12} />} onClick={addTier}>
            Add Tier
          </Button>
        }
        padding
      >
        <div className="space-y-3 mt-2">
          {wf.tiers.map((tier, i) => (
            <div key={i} className="grid grid-cols-5 gap-3 items-end p-3 rounded-lg bg-gray-50 dark:bg-gray-700/30">
              <Input label="LP IRR Floor (%)" type="number" suffix="%" value={tier.irr_min}
                onChange={(e) => updateTier(i, 'irr_min', +e.target.value)} />
              <Input label="LP IRR Ceiling (%)" type="number" suffix="%" value={tier.irr_max >= 999 ? 999 : tier.irr_max}
                onChange={(e) => updateTier(i, 'irr_max', +e.target.value)}
                hint={tier.irr_max >= 999 ? '(uncapped)' : undefined} />
              <Input label="LP Split (%)" type="number" suffix="%" value={tier.lp_split}
                onChange={(e) => { updateTier(i, 'lp_split', +e.target.value); updateTier(i, 'gp_split', 100 - +e.target.value); }} />
              <Input label="GP Split (%)" type="number" suffix="%" value={tier.gp_split}
                onChange={(e) => { updateTier(i, 'gp_split', +e.target.value); updateTier(i, 'lp_split', 100 - +e.target.value); }} />
              <Button variant="ghost" size="sm" icon={<Trash2 size={14} />} onClick={() => removeTier(i)}
                disabled={wf.tiers.length <= 1} className="text-red-500 hover:text-red-600" />
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 rounded-lg bg-brand-50 dark:bg-brand-900/20 border border-brand-100 dark:border-brand-800">
          <p className="text-xs text-brand-700 dark:text-brand-300 font-medium mb-1">Waterfall Order:</p>
          <ol className="text-xs text-brand-600 dark:text-brand-400 space-y-0.5 list-decimal list-inside">
            <li>Return of capital (pro-rata {wf.lp_equity_pct}/{wf.gp_equity_pct})</li>
            <li>LP preferred return ({wf.preferred_return}% {wf.pref_compounding ? 'compounding' : 'simple'}, cumulative)</li>
            {wf.gp_catchup && <li>GP catch-up ({wf.gp_target_promote_pct}% of total profits)</li>}
            {wf.tiers.map((t, i) => (
              <li key={i}>
                IRR {t.irr_min}–{t.irr_max >= 999 ? '∞' : t.irr_max}%: {t.lp_split}/{t.gp_split} LP/GP
              </li>
            ))}
          </ol>
        </div>
      </Card>
    </div>
  );
}
