import axios from 'axios';
import type { DealState, AnalysisResults, SensitivityResults } from '../types/deal';

const api = axios.create({ baseURL: '/api' });

export async function parseDocument(
  file: File,
  docType: 'om' | 't12' | 'rent_roll'
) {
  const form = new FormData();
  form.append('file', file);
  form.append('doc_type', docType);
  const { data } = await api.post('/documents/parse', form);
  return data;
}

export async function getDemoData() {
  const { data } = await api.get('/documents/demo');
  return data;
}

export async function runAnalysis(deal: DealState): Promise<AnalysisResults> {
  const { data } = await api.post('/analysis/run', { deal });
  return data;
}

export async function runSensitivity(deal: DealState): Promise<SensitivityResults> {
  const { data } = await api.post('/analysis/sensitivity', { deal });
  return data;
}

export async function solveDeal(
  deal: DealState,
  targetLpIrr: number,
  solveFor: 'purchase_price' | 'exit_cap_rate'
) {
  const { data } = await api.post('/analysis/solve', {
    deal,
    target_lp_irr: targetLpIrr,
    solve_for: solveFor,
  });
  return data;
}

export async function saveDeal(deal: DealState) {
  const { data } = await api.post('/deals/', deal);
  return data;
}

export async function loadDeal(dealId: string): Promise<DealState> {
  const { data } = await api.get(`/deals/${dealId}`);
  return data;
}
