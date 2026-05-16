export interface PropertyInfo {
  name: string;
  address: string;
  asset_type: string;
  units: number;
  sqft: number;
  year_built: number;
  purchase_price: number;
  asking_price: number;
  market: string;
  submarket: string;
  sponsor_projected_noi: number;
}

export interface T12Data {
  gross_potential_rent: number;
  vacancy_loss: number;
  concessions: number;
  bad_debt: number;
  other_income: number;
  effective_gross_income: number;
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
  total_expenses: number;
  noi: number;
}

export interface RentRollUnit {
  unit_number: string;
  unit_type: string;
  sqft: number;
  market_rent: number;
  current_rent: number;
  lease_start: string | null;
  lease_end: string | null;
  status: 'Occupied' | 'Vacant' | 'MTM';
}

export interface ValueAddAssumptions {
  enabled: boolean;
  units_to_renovate: number;
  renovation_cost_per_unit: number;
  rent_premium_per_unit: number;
  absorption_years: number;
}

export interface Assumptions {
  rent_growth_rate: number;
  vacancy_rate: number;
  credit_loss_rate: number;
  other_income_growth: number;
  expense_growth_rate: number;
  management_fee_pct: number;
  capex_reserves_per_unit: number;
  value_add: ValueAddAssumptions;
}

export interface FinancingAssumptions {
  ltv_pct: number;
  interest_rate: number;
  io_period_years: number;
  amortization_years: number;
  loan_term_years: number;
  loan_type: string;
  enable_refi: boolean;
  refi_year: number;
  refi_ltv_pct: number;
  refi_rate: number;
}

export interface ExitAssumptions {
  hold_period_years: number;
  exit_cap_rate: number;
  selling_costs_pct: number;
}

export interface WaterfallTier {
  irr_min: number;
  irr_max: number;
  lp_split: number;
  gp_split: number;
}

export interface WaterfallConfig {
  lp_equity_pct: number;
  gp_equity_pct: number;
  preferred_return: number;
  pref_compounding: boolean;
  gp_catchup: boolean;
  gp_catchup_rate: number;
  gp_target_promote_pct: number;
  tiers: WaterfallTier[];
}

export interface DealState {
  deal_id: string;
  name: string;
  property_info: PropertyInfo;
  t12_data: T12Data;
  rent_roll: RentRollUnit[];
  assumptions: Assumptions;
  financing: FinancingAssumptions;
  exit_assumptions: ExitAssumptions;
  waterfall_config: WaterfallConfig;
  results?: AnalysisResults | null;
  raw_extraction?: Record<string, unknown> | null;
}

// Analysis result types
export interface AnnualRow {
  year: number;
  gross_potential_rent: number;
  vacancy_loss: number;
  credit_loss: number;
  other_income: number;
  egi: number;
  property_taxes: number;
  insurance: number;
  management_fee: number;
  maintenance: number;
  utilities: number;
  payroll: number;
  general_admin: number;
  marketing: number;
  capex_reserves: number;
  other_expenses: number;
  total_expenses: number;
  noi: number;
  debt_service: number;
  levered_cf: number;
  dscr: number | null;
}

export interface Metrics {
  purchase_price: number;
  loan_amount: number;
  equity_required: number;
  closing_costs: number;
  going_in_cap_rate: number;
  stabilized_cap_rate: number;
  year1_noi: number;
  year1_egi: number;
  noi_margin: number | null;
  dscr_year1: number | null;
  levered_coc: number | null;
  unlevered_coc: number;
  levered_irr: number | null;
  unlevered_irr: number | null;
  levered_em: number | null;
  unlevered_em: number | null;
  npv_10pct: number;
}

export interface ExitSummary {
  exit_year_noi: number;
  exit_cap_rate: number;
  gross_sale_price: number;
  selling_costs: number;
  net_sale_price: number;
  loan_payoff: number;
  net_sale_proceeds: number;
}

export interface WaterfallYear {
  year: number;
  is_exit: boolean;
  distributable: number;
  lp_distribution: number;
  gp_distribution: number;
  tier_roc_lp: number;
  tier_roc_gp: number;
  tier_preferred_return: number;
  tier_gp_catchup: number;
  tier_lp_promote: number;
  tier_gp_promote: number;
}

export interface WaterfallResults {
  yearly: WaterfallYear[];
  lp_total_distributions: number;
  gp_total_distributions: number;
  lp_irr: number | null;
  gp_irr: number | null;
  lp_em: number | null;
  gp_em: number | null;
  gp_promote_earned: number;
  lp_invested: number;
  gp_invested: number;
  pref_paid: number;
}

export interface SensitivityTable {
  row_param: string;
  col_param: string;
  row_values: number[];
  col_values: number[];
  output_metric: string;
  table: (number | null)[][];
}

export interface AnalysisResults {
  proforma: {
    annual_rows: AnnualRow[];
    metrics: Metrics;
    exit: ExitSummary;
    cash_flows: { levered: number[]; unlevered: number[] };
    loan_amount: number;
    total_equity: number;
  };
  waterfall: WaterfallResults;
  metrics: Metrics;
  lp_irr: number | null;
  levered_irr: number | null;
  levered_em: number | null;
  levered_coc: number | null;
}

export interface SensitivityResults {
  lp_irr_vs_price_x_exit_cap: SensitivityTable;
  lp_irr_vs_rent_x_vacancy: SensitivityTable;
  coc_vs_ltv_x_rate: SensitivityTable;
  em_vs_hold_x_exit_cap: SensitivityTable;
}

export type WizardStep =
  | 'upload'
  | 'review'
  | 'assumptions'
  | 'financing'
  | 'waterfall'
  | 'results';
