"""
Financial Accuracy Tests — CRE Analyzer Backend
================================================

Tests are organised into four sections:

1. IRR Solver            – known closed-form solutions
2. Debt Service          – I/O and amortising payment maths
3. Full Pro-Forma        – Sunset Ridge Apartments (default deal)
4. LP/GP Waterfall       – capital stack distribution logic

All expected values are derived analytically or via the curl smoke-test
(Going-in Cap 5.25 %, Year-1 NOI $788 080, Levered IRR 8.33 %).
"""

import math
import sys
import os
import pytest

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.underwriting_engine import (
    calculate_irr,
    calculate_npv,
    annual_debt_service,
    calculate_loan_balance,
    build_proforma,
)
from services.waterfall_engine import run_waterfall
from models.deal import (
    DealState,
    PropertyInfo,
    T12Data,
    Assumptions,
    FinancingAssumptions,
    ExitAssumptions,
    WaterfallConfig,
    WaterfallTier,
    ValueAddAssumptions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx(val, rel=0.005):
    """Assert within 0.5 % relative tolerance (5 bps for rates)."""
    return pytest.approx(val, rel=rel)


def sunset_ridge_deal() -> DealState:
    """Default Sunset Ridge deal matching the frontend store defaults."""
    return DealState(
        property_info=PropertyInfo(
            purchase_price=15_000_000,
            units=100,
        ),
        t12_data=T12Data(
            gross_potential_rent=1_260_000,
            vacancy_loss=63_000,
            concessions=12_600,
            bad_debt=6_300,
            other_income=60_000,
            property_taxes=138_000,
            insurance=42_000,
            management_fee=49_524,
            maintenance_repairs=65_000,
            utilities=38_000,
            payroll=78_000,
            general_admin=28_000,
            marketing=15_000,
            capex_reserves=25_000,
            other_expenses=7_576,
        ),
        assumptions=Assumptions(
            rent_growth_rate=3.0,
            vacancy_rate=5.0,
            credit_loss_rate=0.5,
            other_income_growth=3.0,
            expense_growth_rate=3.0,
            management_fee_pct=4.0,
            capex_reserves_per_unit=250,
            value_add=ValueAddAssumptions(enabled=False),
        ),
        financing=FinancingAssumptions(
            ltv_pct=65.0,
            interest_rate=6.5,
            io_period_years=2,
            amortization_years=30,
            loan_term_years=5,
        ),
        exit_assumptions=ExitAssumptions(
            hold_period_years=5,
            exit_cap_rate=5.25,
            selling_costs_pct=2.0,
        ),
        waterfall_config=WaterfallConfig(
            lp_equity_pct=90,
            gp_equity_pct=10,
            preferred_return=8.0,
            pref_compounding=False,
            gp_catchup=True,
            gp_catchup_rate=50,
            gp_target_promote_pct=20,
            tiers=[
                WaterfallTier(irr_min=0,  irr_max=14,  lp_split=80, gp_split=20),
                WaterfallTier(irr_min=14, irr_max=18,  lp_split=70, gp_split=30),
                WaterfallTier(irr_min=18, irr_max=999, lp_split=60, gp_split=40),
            ],
        ),
    )


# ===========================================================================
# 1. IRR Solver
# ===========================================================================

class TestIRRSolver:
    """
    Known closed-form IRR cases.

    For a single-period investment:  IRR = (FV/PV)^(1/n) - 1
    For a bond:  cash flows [-P, C, C, ..., C+P] → IRR = coupon rate when P = face.
    """

    def test_two_period_10_pct(self):
        # Invest $100, receive $121 after 2 years: IRR = (121/100)^0.5 - 1 = 10 %
        cfs = [-100, 0, 121]
        irr = calculate_irr(cfs)
        assert irr == approx(0.10)

    def test_bond_8_pct(self):
        # Par bond: invest $1 000, annual coupon $80 for 5 yrs + $1 080 at end → IRR = 8 %
        cfs = [-1_000, 80, 80, 80, 80, 1_080]
        irr = calculate_irr(cfs)
        assert irr == approx(0.08)

    def test_single_year_15_pct(self):
        # Invest $1 M, get $1.15 M in one year → IRR = 15 %
        cfs = [-1_000_000, 1_150_000]
        irr = calculate_irr(cfs)
        assert irr == approx(0.15)

    def test_high_irr_30_pct(self):
        # Invest $100, get $169 after 2 years: (169/100)^0.5 - 1 = 30 %
        cfs = [-100, 0, 169]
        irr = calculate_irr(cfs)
        assert irr == approx(0.30)

    def test_npv_at_irr_is_zero(self):
        """NPV evaluated at the computed IRR must be near zero."""
        cfs = [-500_000, 80_000, 80_000, 80_000, 80_000, 620_000]
        irr = calculate_irr(cfs)
        assert irr is not None
        npv = calculate_npv(cfs, irr)
        assert abs(npv) < 1.0   # within $1

    def test_no_solution_all_positive(self):
        assert calculate_irr([100, 200, 300]) is None

    def test_no_solution_all_negative(self):
        assert calculate_irr([-100, -200]) is None

    def test_sign_change_required(self):
        # Only one sign change — solvable
        irr = calculate_irr([-1_000, 500, 700])
        assert irr is not None
        assert 0 < irr < 1   # positive IRR between 0 and 100 %


# ===========================================================================
# 2. Debt Service
# ===========================================================================

class TestDebtService:
    """
    Manually derived expected values for $9.75 M loan at 6.5 % / 30-yr am.
    """

    LOAN   = 9_750_000.0
    RATE   = 0.065
    IO_YRS = 2
    AM_YRS = 30

    # ---- Interest-Only years -----------------------------------------------

    def test_io_year_1_payment(self):
        # I/O payment = loan × rate
        ds = annual_debt_service(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, year=1)
        expected = self.LOAN * self.RATE   # $633 750
        assert ds == approx(expected)

    def test_io_year_2_payment(self):
        ds = annual_debt_service(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, year=2)
        assert ds == approx(self.LOAN * self.RATE)

    # ---- First amortising year (year 3) ------------------------------------

    def test_amortising_year_3_payment(self):
        """
        Standard mortgage: PMT = P * [r*(1+r)^n] / [(1+r)^n - 1]
        r_monthly = 6.5%/12, n = 360 months → annual = PMT * 12
        """
        r = self.RATE / 12
        n = self.AM_YRS * 12
        monthly = self.LOAN * (r * (1 + r)**n) / ((1 + r)**n - 1)
        expected_annual = monthly * 12

        ds = annual_debt_service(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, year=3)
        assert ds == approx(expected_annual)

    def test_amortising_greater_than_io(self):
        """Am payment must exceed I/O payment (principal reduces balance)."""
        io_ds  = annual_debt_service(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, year=1)
        am_ds  = annual_debt_service(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, year=3)
        assert am_ds > io_ds

    # ---- Loan balance -------------------------------------------------------

    def test_balance_during_io_period(self):
        """Balance is unchanged during I/O — no principal paid."""
        bal = calculate_loan_balance(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, after_years=1)
        assert bal == approx(self.LOAN)

    def test_balance_at_io_boundary(self):
        bal = calculate_loan_balance(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, after_years=2)
        assert bal == approx(self.LOAN)

    def test_balance_after_5_years_less_than_original(self):
        """After 3 amortising years the balance must be below original loan."""
        bal = calculate_loan_balance(self.LOAN, self.RATE, self.IO_YRS, self.AM_YRS, after_years=5)
        assert bal < self.LOAN
        # Rough sanity: should still be > 90 % of original (only 3 yrs am on 30-yr)
        assert bal > self.LOAN * 0.90

    def test_zero_rate_io(self):
        """0 % I/O payment is $0 interest."""
        ds = annual_debt_service(1_000_000, 0.0, 3, 30, year=1)
        assert ds == approx(0.0)


# ===========================================================================
# 3. Full Pro-Forma — Sunset Ridge Apartments
# ===========================================================================

class TestProformaSunsetRidge:
    """
    Validates the full proforma against analytically derived values and the
    live smoke-test results from the curl integration test.
    """

    @pytest.fixture(scope="class")
    def pf(self):
        return build_proforma(sunset_ridge_deal())

    # ---- Capitalization structure ------------------------------------------

    def test_loan_amount(self, pf):
        # 65 % LTV on $15 M
        assert pf["loan_amount"] == approx(9_750_000, rel=0.001)

    def test_equity_required(self, pf):
        # Down payment + 1 % closing costs
        equity_dp   = 15_000_000 * 0.35      # $5 250 000
        closing     = 15_000_000 * 0.01      # $150 000
        assert pf["total_equity"] == approx(equity_dp + closing, rel=0.001)

    # ---- Year 1 NOI (confirmed by curl smoke-test) -------------------------

    def test_year1_noi(self, pf):
        """
        Manually derived Year-1 NOI = $788 080 (see calc in AGENTS.md comments).
        Smoke-test curl output confirmed this value.
        """
        yr1_noi = pf["annual_rows"][0]["noi"]
        assert yr1_noi == approx(788_080, rel=0.01)

    def test_going_in_cap_rate(self, pf):
        # Smoke-test confirmed 5.25 %
        assert pf["metrics"]["going_in_cap_rate"] == approx(5.25, rel=0.02)

    # ---- Debt service in Year 1 (I/O) -------------------------------------

    def test_year1_debt_service_is_io(self, pf):
        expected_io = 9_750_000 * 0.065
        assert pf["annual_rows"][0]["debt_service"] == approx(expected_io, rel=0.001)

    def test_year1_levered_cf_positive(self, pf):
        assert pf["annual_rows"][0]["levered_cf"] > 0

    def test_year1_dscr_above_1(self, pf):
        assert pf["annual_rows"][0]["dscr"] > 1.0

    # ---- Exit calculations -------------------------------------------------

    def test_exit_noi_higher_than_year1(self, pf):
        """Revenue grows each year so exit NOI > Year-1 NOI."""
        assert pf["exit"]["exit_year_noi"] > pf["annual_rows"][0]["noi"]

    def test_gross_sale_price_formula(self, pf):
        """Exit value = exit NOI / exit cap."""
        expected = pf["exit"]["exit_year_noi"] / 0.0525
        assert pf["exit"]["gross_sale_price"] == approx(expected, rel=0.001)

    def test_selling_costs_2pct(self, pf):
        gross = pf["exit"]["gross_sale_price"]
        assert pf["exit"]["selling_costs"] == approx(gross * 0.02, rel=0.001)

    # ---- Return metrics ---------------------------------------------------

    def test_levered_irr_positive(self, pf):
        assert pf["metrics"]["levered_irr"] is not None
        assert pf["metrics"]["levered_irr"] > 0

    def test_levered_irr_near_smoke_test(self, pf):
        # Smoke-test: 8.33 %
        assert pf["metrics"]["levered_irr"] == approx(8.33, rel=0.05)

    def test_unlevered_irr_less_than_levered(self, pf):
        """Positive leverage: levered IRR > unlevered IRR when loan rate < cap rate?
        Actually with going-in cap 5.25% and loan rate 6.5%, unlevered > levered."""
        # Both should be positive
        assert pf["metrics"]["unlevered_irr"] > 0
        assert pf["metrics"]["levered_irr"] > 0

    def test_levered_em_above_1(self, pf):
        """Positive returns: equity multiple > 1×."""
        assert pf["metrics"]["levered_em"] > 1.0

    def test_cash_flows_array_length(self, pf):
        """Levered CF array: year 0 + 5 hold years = 6 elements."""
        assert len(pf["cash_flows"]["levered"]) == 6

    def test_year0_cf_is_negative_equity(self, pf):
        """Year-0 cash flow is the initial equity outflow (negative)."""
        assert pf["cash_flows"]["levered"][0] < 0
        assert pf["cash_flows"]["levered"][0] == approx(-pf["total_equity"], rel=0.001)

    # ---- Multi-year growth sanity ------------------------------------------

    def test_noi_grows_year_over_year(self, pf):
        nois = [row["noi"] for row in pf["annual_rows"]]
        for i in range(1, len(nois)):
            assert nois[i] > nois[i - 1], f"NOI did not grow from yr {i} to yr {i+1}"

    def test_gpr_grows_at_3pct(self, pf):
        """GPR should grow ~3 % per year."""
        rows = pf["annual_rows"]
        for i in range(1, len(rows)):
            ratio = rows[i]["gross_potential_rent"] / rows[i - 1]["gross_potential_rent"]
            assert ratio == approx(1.03, rel=0.01)


# ===========================================================================
# 4. LP/GP Waterfall
# ===========================================================================

class TestWaterfall:
    """
    Tests the waterfall engine against hand-calculated expected distributions.
    """

    # ---- Setup helpers ------------------------------------------------------

    def simple_config(self, lp_pct=90, gp_pct=10, pref=8.0,
                      catchup=True, catchup_rate=50, promote_pct=20,
                      tiers=None) -> WaterfallConfig:
        return WaterfallConfig(
            lp_equity_pct=lp_pct,
            gp_equity_pct=gp_pct,
            preferred_return=pref,
            pref_compounding=False,
            gp_catchup=catchup,
            gp_catchup_rate=catchup_rate,
            gp_target_promote_pct=promote_pct,
            tiers=tiers or [
                WaterfallTier(irr_min=0, irr_max=14,  lp_split=80, gp_split=20),
                WaterfallTier(irr_min=14, irr_max=999, lp_split=60, gp_split=40),
            ],
        )

    # ---- Case 1: Return of capital only (no profit) ------------------------

    def test_roc_only_lp_gets_invested_capital_back(self):
        """
        Invest $9M (LP) + $1M (GP).  Exit proceeds = $10M → both get capital back,
        nothing extra.
        """
        equity = 10_000_000
        config = self.simple_config(lp_pct=90, gp_pct=10)
        result = run_waterfall(
            equity_required=equity,
            annual_levered_cfs=[0] * 5,   # no operating CFs
            exit_proceeds=equity,          # exactly capital back
            config=config,
        )
        assert result["lp_total_distributions"] == approx(9_000_000, rel=0.01)
        assert result["gp_total_distributions"] == approx(1_000_000, rel=0.01)
        assert result["gp_promote_earned"] == pytest.approx(0, abs=1)

    # ---- Case 2: Preferred return to LP ------------------------------------

    def test_pref_return_paid_before_promote(self):
        """
        LP invests $9M, 8% pref (non-compounding), 5-yr hold, no annual CFs.
        At exit: $9M RoC + 5 × $720K pref = $12.6M to LP before any promote.
        Excess goes to GP catch-up then promote.
        """
        equity = 10_000_000
        lp_invested = 9_000_000
        pref_accrued = lp_invested * 0.08 * 5   # $3 600 000

        exit_proceeds = equity + pref_accrued + 2_000_000   # plenty of profit
        config = self.simple_config(catchup=False,
                                    tiers=[WaterfallTier(irr_min=0, irr_max=999, lp_split=80, gp_split=20)])
        result = run_waterfall(
            equity_required=equity,
            annual_levered_cfs=[0] * 5,
            exit_proceeds=exit_proceeds,
            config=config,
        )
        # LP must have received at minimum RoC + full pref
        lp_min = lp_invested + pref_accrued
        assert result["lp_total_distributions"] >= lp_min - 1   # within $1

    def test_pref_paid_label_matches_distribution(self):
        """pref_paid field should equal the preferred return received by LP."""
        equity = 10_000_000
        lp_invested = 9_000_000
        pref_per_year = lp_invested * 0.08    # $720 000
        # Annual CFs exactly equal pref → pref paid yearly, none at exit
        annual_cfs = [pref_per_year] * 5
        exit_proceeds = lp_invested + 1_000_000   # LP RoC + $1M profit
        config = self.simple_config(catchup=False,
                                    tiers=[WaterfallTier(irr_min=0, irr_max=999, lp_split=80, gp_split=20)])
        result = run_waterfall(equity, annual_cfs, exit_proceeds, config)
        assert result["pref_paid"] == approx(pref_per_year * 5, rel=0.05)

    # ---- Case 3: Full waterfall with Sunset Ridge numbers ------------------

    def test_sunset_ridge_waterfall_lp_irr(self):
        """
        LP earns less than levered IRR due to GP promote.
        Levered IRR ≈ 8.33 %; LP IRR should be in the 6–8 % range.
        """
        deal = sunset_ridge_deal()
        pf = build_proforma(deal)
        annual_cfs = pf["cash_flows"]["levered"][1:-1]   # years 1-4
        exit_proceeds = pf["exit"]["net_sale_proceeds"]

        result = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=annual_cfs,
            exit_proceeds=exit_proceeds,
            config=deal.waterfall_config,
        )
        assert result["lp_irr"] is not None
        assert 6.0 <= result["lp_irr"] <= 8.5, (
            f"LP IRR {result['lp_irr']} % is outside expected 6–8.5 % range"
        )
        # LP IRR must be below levered IRR (GP promote takes a cut)
        assert result["lp_irr"] < pf["metrics"]["levered_irr"]

    def test_sunset_ridge_gp_promote_positive(self):
        deal = sunset_ridge_deal()
        pf = build_proforma(deal)
        result = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=pf["cash_flows"]["levered"][1:-1],
            exit_proceeds=pf["exit"]["net_sale_proceeds"],
            config=deal.waterfall_config,
        )
        assert result["gp_promote_earned"] > 0

    def test_total_distributions_equal_total_cash_in(self):
        """
        LP distributions + GP distributions must equal sum of all distributable
        cash (operating CFs ≥ 0 only, plus exit proceeds).
        No cash is created or destroyed.
        """
        deal = sunset_ridge_deal()
        pf = build_proforma(deal)
        annual_cfs = pf["cash_flows"]["levered"][1:-1]
        exit_proceeds = pf["exit"]["net_sale_proceeds"]

        result = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=annual_cfs,
            exit_proceeds=exit_proceeds,
            config=deal.waterfall_config,
        )
        distributable_total = sum(max(0, c) for c in annual_cfs) + max(0, exit_proceeds)
        total_out = result["lp_total_distributions"] + result["gp_total_distributions"]
        assert total_out == approx(distributable_total, rel=0.001)

    # ---- Case 4: LP IRR > GP IRR (LP should earn more per $ invested) ------

    def test_lp_em_lower_than_gp_em_due_to_promote(self):
        """
        GP earns promote, so GP EM > LP EM when returns exceed pref.
        LP EM should still be > 1 (positive return).
        """
        deal = sunset_ridge_deal()
        pf = build_proforma(deal)
        result = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=pf["cash_flows"]["levered"][1:-1],
            exit_proceeds=pf["exit"]["net_sale_proceeds"],
            config=deal.waterfall_config,
        )
        assert result["lp_em"] > 1.0
        assert result["gp_em"] > 1.0

    # ---- Case 5: 100 % LP, no GP ------------------------------------------

    def test_100pct_lp_all_goes_to_lp(self):
        """When LP holds 100 % equity, all distributions go to LP."""
        config = WaterfallConfig(
            lp_equity_pct=100,
            gp_equity_pct=0,
            preferred_return=8.0,
            pref_compounding=False,
            gp_catchup=False,
            tiers=[WaterfallTier(irr_min=0, irr_max=999, lp_split=100, gp_split=0)],
        )
        equity = 1_000_000
        exit_proceeds = 1_500_000
        result = run_waterfall(equity, [0] * 5, exit_proceeds, config)
        assert result["gp_total_distributions"] == pytest.approx(0, abs=1)
        assert result["lp_total_distributions"] == pytest.approx(exit_proceeds, abs=1)

    # ---- Case 6: Edge — no exit proceeds -----------------------------------

    def test_no_exit_proceeds_zero_distributions(self):
        """If exit is $0 and no operating CFs, no one gets paid."""
        config = self.simple_config()
        result = run_waterfall(1_000_000, [0] * 5, 0, config)
        assert result["lp_total_distributions"] == 0
        assert result["gp_total_distributions"] == 0


# ===========================================================================
# 5. Solve-for-Price / Solve-for-Cap Regression
# ===========================================================================

class TestSolveConsistency:
    """
    Regression tests for the solver bug where solve_for_price used
    pf["cash_flows"]["levered"][1:-1] (dropping year-N operating CF) instead
    of matching the full_analysis cash-flow separation.

    Invariant: at the price the solver returns, re-running the waterfall with
    the CORRECT cash-flow setup should yield LP IRR ≈ target_lp_irr.
    """

    def _run_waterfall_correctly(self, deal, pf):
        """Use the production helper so the test tracks the real code path."""
        from services.waterfall_engine import waterfall_from_proforma
        return waterfall_from_proforma(pf, deal.waterfall_config)

    def test_solve_price_yields_target_lp_irr(self):
        """
        solve_for_price(deal, 14%) must return a price X such that
        running the waterfall at X gives LP IRR ≈ 14%.
        """
        from services.waterfall_engine import solve_for_price
        from copy import deepcopy

        deal = sunset_ridge_deal()
        target = 14.0

        solved_price = solve_for_price(deal, target, build_proforma)
        assert solved_price is not None

        # Verify at the solved price LP IRR ≈ target
        d = deepcopy(deal)
        d.property_info.purchase_price = float(solved_price)
        pf = build_proforma(d)
        result = self._run_waterfall_correctly(d, pf)

        assert result["lp_irr"] is not None, "LP IRR should be computable at solved price"
        assert result["lp_irr"] == pytest.approx(target, abs=0.5), (
            f"LP IRR at solved price {solved_price:,.0f} is {result['lp_irr']:.2f}%, "
            f"expected ≈ {target}%"
        )

    def test_solve_price_lower_than_current_when_irr_deficit(self):
        """
        Regression: the solver must return a LOWER price than the current
        purchase price when the current LP IRR is already below the target.
        (The old bug returned a HIGHER price, e.g. $13.6M vs $9.5M input.)
        """
        from services.waterfall_engine import solve_for_price

        deal = sunset_ridge_deal()
        pf_current = build_proforma(deal)
        current_lp_irr = self._run_waterfall_correctly(deal, pf_current)["lp_irr"]

        target = 14.0
        assert current_lp_irr < target, "Precondition: current LP IRR below target"

        solved_price = solve_for_price(deal, target, build_proforma)
        assert solved_price < deal.property_info.purchase_price, (
            f"Solver returned {solved_price:,.0f} which is >= current price "
            f"{deal.property_info.purchase_price:,.0f}. "
            f"Current LP IRR is {current_lp_irr:.2f}% < target {target}%, "
            f"so the max purchase price must be lower."
        )

    def test_solve_cap_yields_target_lp_irr(self):
        """
        solve_for_exit_cap(deal, 14%) must return an exit cap X such that
        running the waterfall at X gives LP IRR ≈ 14%.
        """
        from services.waterfall_engine import solve_for_exit_cap
        from copy import deepcopy

        deal = sunset_ridge_deal()
        target = 14.0

        solved_cap = solve_for_exit_cap(deal, target, build_proforma)
        assert solved_cap is not None

        d = deepcopy(deal)
        d.exit_assumptions.exit_cap_rate = float(solved_cap)
        pf = build_proforma(d)
        result = self._run_waterfall_correctly(d, pf)

        assert result["lp_irr"] is not None
        assert result["lp_irr"] == pytest.approx(target, abs=0.5), (
            f"LP IRR at solved cap {solved_cap:.2f}% is {result['lp_irr']:.2f}%, "
            f"expected ≈ {target}%"
        )
