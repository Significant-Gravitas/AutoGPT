from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid


class PropertyInfo(BaseModel):
    name: str = "Sample Multifamily Asset"
    address: str = "123 Main Street, Austin, TX 78701"
    asset_type: str = "Multifamily"
    units: int = 100
    sqft: float = 85000
    year_built: int = 1995
    purchase_price: float = 15_000_000
    asking_price: float = 15_500_000
    market: str = "Austin, TX"
    submarket: str = ""
    sponsor_projected_noi: float = 0


class T12LineItem(BaseModel):
    label: str
    t12_actual: float = 0
    underwritten: float = 0
    notes: str = ""


class T12Data(BaseModel):
    # Income
    gross_potential_rent: float = 1_200_000
    vacancy_loss: float = 60_000
    concessions: float = 12_000
    bad_debt: float = 6_000
    other_income: float = 48_000
    effective_gross_income: float = 0  # computed

    # Expenses
    property_taxes: float = 120_000
    insurance: float = 36_000
    management_fee: float = 48_000
    maintenance_repairs: float = 60_000
    utilities: float = 36_000
    payroll: float = 72_000
    general_admin: float = 24_000
    marketing: float = 12_000
    capex_reserves: float = 25_000
    other_expenses: float = 0
    total_expenses: float = 0  # computed
    noi: float = 750_000

    def compute(self) -> "T12Data":
        self.effective_gross_income = (
            self.gross_potential_rent
            - self.vacancy_loss
            - self.concessions
            - self.bad_debt
            + self.other_income
        )
        self.total_expenses = (
            self.property_taxes
            + self.insurance
            + self.management_fee
            + self.maintenance_repairs
            + self.utilities
            + self.payroll
            + self.general_admin
            + self.marketing
            + self.capex_reserves
            + self.other_expenses
        )
        self.noi = self.effective_gross_income - self.total_expenses
        return self


class RentRollUnit(BaseModel):
    unit_number: str = ""
    unit_type: str = "1BR/1BA"
    sqft: float = 850
    market_rent: float = 1200
    current_rent: float = 1150
    lease_start: Optional[str] = None
    lease_end: Optional[str] = None
    status: str = "Occupied"  # Occupied, Vacant, MTM


class ValueAddAssumptions(BaseModel):
    enabled: bool = False
    units_to_renovate: int = 50
    renovation_cost_per_unit: float = 15_000
    rent_premium_per_unit: float = 150  # $/month
    absorption_years: int = 2  # years to complete all renovations


class Assumptions(BaseModel):
    # Revenue
    rent_growth_rate: float = 3.0  # %/yr
    vacancy_rate: float = 5.0  # %
    credit_loss_rate: float = 0.5  # %
    other_income_growth: float = 3.0  # %/yr

    # Expenses
    expense_growth_rate: float = 3.0  # %/yr
    management_fee_pct: float = 4.0  # % of EGI
    capex_reserves_per_unit: float = 250  # $/unit/yr

    # Value-add
    value_add: ValueAddAssumptions = Field(default_factory=ValueAddAssumptions)


class FinancingAssumptions(BaseModel):
    ltv_pct: float = 65.0
    interest_rate: float = 6.5  # %
    io_period_years: int = 2
    amortization_years: int = 30
    loan_term_years: int = 5
    loan_type: str = "Agency"  # Agency, Bridge

    # Refi scenario
    enable_refi: bool = False
    refi_year: int = 3
    refi_ltv_pct: float = 70.0
    refi_rate: float = 6.0


class ExitAssumptions(BaseModel):
    hold_period_years: int = 5
    exit_cap_rate: float = 5.25  # %
    selling_costs_pct: float = 2.0  # %


class WaterfallTier(BaseModel):
    irr_min: float = 0.0   # % (LP IRR floor for this tier)
    irr_max: float = 14.0  # % (LP IRR ceiling; 999 = unlimited)
    lp_split: float = 80.0
    gp_split: float = 20.0


class WaterfallConfig(BaseModel):
    lp_equity_pct: float = 90.0
    gp_equity_pct: float = 10.0
    preferred_return: float = 8.0  # % annual
    pref_compounding: bool = False
    gp_catchup: bool = True
    gp_catchup_rate: float = 50.0  # % to GP during catch-up phase
    gp_target_promote_pct: float = 20.0  # GP % of total profits in catch-up target
    tiers: List[WaterfallTier] = Field(
        default_factory=lambda: [
            WaterfallTier(irr_min=0, irr_max=14, lp_split=80, gp_split=20),
            WaterfallTier(irr_min=14, irr_max=18, lp_split=70, gp_split=30),
            WaterfallTier(irr_min=18, irr_max=999, lp_split=60, gp_split=40),
        ]
    )


class DealState(BaseModel):
    deal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Deal"
    property_info: PropertyInfo = Field(default_factory=PropertyInfo)
    t12_data: T12Data = Field(default_factory=T12Data)
    rent_roll: List[RentRollUnit] = Field(default_factory=list)
    assumptions: Assumptions = Field(default_factory=Assumptions)
    financing: FinancingAssumptions = Field(default_factory=FinancingAssumptions)
    exit_assumptions: ExitAssumptions = Field(default_factory=ExitAssumptions)
    waterfall_config: WaterfallConfig = Field(default_factory=WaterfallConfig)
    results: Optional[Dict[str, Any]] = None
    raw_extraction: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    deal: DealState


class SolveRequest(BaseModel):
    deal: DealState
    target_lp_irr: float = 14.0  # %
    solve_for: str = "purchase_price"  # or "exit_cap_rate"
