import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  DealState,
  WizardStep,
  AnalysisResults,
  SensitivityResults,
} from '../types/deal';

const DEFAULT_DEAL: DealState = {
  deal_id: crypto.randomUUID(),
  name: 'New Deal',
  property_info: {
    name: 'Sunset Ridge Apartments',
    address: '4500 Sunset Ridge Dr, Austin, TX 78741',
    asset_type: 'Multifamily',
    units: 100,
    sqft: 85000,
    year_built: 1998,
    purchase_price: 15_000_000,
    asking_price: 15_500_000,
    market: 'Austin, TX',
    submarket: 'South Austin',
    sponsor_projected_noi: 825_000,
  },
  t12_data: {
    gross_potential_rent: 1_260_000,
    vacancy_loss: 63_000,
    concessions: 12_600,
    bad_debt: 6_300,
    other_income: 60_000,
    effective_gross_income: 1_238_100,
    property_taxes: 138_000,
    insurance: 42_000,
    management_fee: 49_524,
    maintenance_repairs: 65_000,
    utilities: 38_000,
    payroll: 78_000,
    general_admin: 28_000,
    marketing: 15_000,
    capex_reserves: 25_000,
    other_expenses: 7_576,
    total_expenses: 486_100,
    noi: 752_000,
  },
  rent_roll: [],
  assumptions: {
    rent_growth_rate: 3.0,
    vacancy_rate: 5.0,
    credit_loss_rate: 0.5,
    other_income_growth: 3.0,
    expense_growth_rate: 3.0,
    management_fee_pct: 4.0,
    capex_reserves_per_unit: 250,
    value_add: {
      enabled: false,
      units_to_renovate: 50,
      renovation_cost_per_unit: 15_000,
      rent_premium_per_unit: 150,
      absorption_years: 2,
    },
  },
  financing: {
    ltv_pct: 65.0,
    interest_rate: 6.5,
    io_period_years: 2,
    amortization_years: 30,
    loan_term_years: 5,
    loan_type: 'Agency',
    enable_refi: false,
    refi_year: 3,
    refi_ltv_pct: 70.0,
    refi_rate: 6.0,
  },
  exit_assumptions: {
    hold_period_years: 5,
    exit_cap_rate: 5.25,
    selling_costs_pct: 2.0,
  },
  waterfall_config: {
    lp_equity_pct: 90,
    gp_equity_pct: 10,
    preferred_return: 8.0,
    pref_compounding: false,
    gp_catchup: true,
    gp_catchup_rate: 50,
    gp_target_promote_pct: 20,
    tiers: [
      { irr_min: 0, irr_max: 14, lp_split: 80, gp_split: 20 },
      { irr_min: 14, irr_max: 18, lp_split: 70, gp_split: 30 },
      { irr_min: 18, irr_max: 999, lp_split: 60, gp_split: 40 },
    ],
  },
  results: null,
  raw_extraction: null,
};

interface StoreState {
  deal: DealState;
  step: WizardStep;
  isLoading: boolean;
  isSensLoading: boolean;
  error: string | null;
  sensitivity: SensitivityResults | null;
  darkMode: boolean;

  setDeal: (deal: DealState) => void;
  updateDeal: (partial: Partial<DealState>) => void;
  setStep: (step: WizardStep) => void;
  setResults: (results: AnalysisResults) => void;
  setSensitivity: (s: SensitivityResults) => void;
  setLoading: (v: boolean) => void;
  setSensLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  resetDeal: () => void;
  toggleDarkMode: () => void;
  loadFromJson: (json: string) => void;
  exportJson: () => string;
}

export const useDealStore = create<StoreState>()(
  persist(
    (set, get) => ({
      deal: DEFAULT_DEAL,
      step: 'upload',
      isLoading: false,
      isSensLoading: false,
      error: null,
      sensitivity: null,
      darkMode: false,

      setDeal: (deal) => set({ deal }),
      updateDeal: (partial) => set((s) => ({ deal: { ...s.deal, ...partial } })),
      setStep: (step) => set({ step }),
      setResults: (results) =>
        set((s) => ({ deal: { ...s.deal, results } })),
      setSensitivity: (sensitivity) => set({ sensitivity }),
      setLoading: (isLoading) => set({ isLoading }),
      setSensLoading: (isSensLoading) => set({ isSensLoading }),
      setError: (error) => set({ error }),
      resetDeal: () => set({ deal: { ...DEFAULT_DEAL, deal_id: crypto.randomUUID() }, step: 'upload', sensitivity: null }),
      toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
      loadFromJson: (json) => {
        try {
          const deal = JSON.parse(json) as DealState;
          set({ deal, step: 'results' });
        } catch {
          set({ error: 'Invalid deal file' });
        }
      },
      exportJson: () => JSON.stringify(get().deal, null, 2),
    }),
    { name: 'cre-deal-store', partialize: (s) => ({ deal: s.deal, darkMode: s.darkMode }) }
  )
);
