"""
Underwriting engine: builds pro forma cash flows and computes return metrics.
"""

from typing import List, Optional, Dict, Any
import math
from models.deal import DealState


# ---------------------------------------------------------------------------
# IRR / NPV helpers
# ---------------------------------------------------------------------------

def calculate_irr(cash_flows: List[float], max_iter: int = 500) -> Optional[float]:
    """Newton-Raphson IRR solver. Returns None if no solution found."""
    if not cash_flows:
        return None
    has_neg = any(c < 0 for c in cash_flows)
    has_pos = any(c > 0 for c in cash_flows)
    if not (has_neg and has_pos):
        return None

    def npv(r: float) -> float:
        return sum(c / (1 + r) ** t for t, c in enumerate(cash_flows))

    def dnpv(r: float) -> float:
        return sum(-t * c / (1 + r) ** (t + 1) for t, c in enumerate(cash_flows))

    for guess in [0.10, 0.05, 0.20, 0.01, 0.30, -0.05]:
        rate = guess
        try:
            for _ in range(max_iter):
                f = npv(rate)
                fp = dnpv(rate)
                if abs(fp) < 1e-14:
                    break
                new_rate = rate - f / fp
                if new_rate <= -1:
                    new_rate = -0.9999
                if abs(new_rate - rate) < 1e-10:
                    rate = new_rate
                    break
                rate = new_rate
            if abs(npv(rate)) < 10:
                return rate
        except Exception:
            continue
    return None


def calculate_npv(cash_flows: List[float], discount_rate: float) -> float:
    return sum(c / (1 + discount_rate) ** t for t, c in enumerate(cash_flows))


def calculate_loan_balance(
    original_balance: float,
    annual_rate: float,
    io_years: int,
    am_years: int,
    after_years: int,
) -> float:
    """Remaining loan balance after `after_years` years."""
    if after_years <= io_years:
        return original_balance

    am_months = am_years * 12
    r = annual_rate / 12
    if r == 0:
        return original_balance - original_balance * (after_years - io_years) / am_years

    monthly_pmt = original_balance * (r * (1 + r) ** am_months) / ((1 + r) ** am_months - 1)
    # Remaining balance after N amortizing months
    am_months_elapsed = (after_years - io_years) * 12
    balance = original_balance * (1 + r) ** am_months_elapsed - monthly_pmt * (
        ((1 + r) ** am_months_elapsed - 1) / r
    )
    return max(0, balance)


def annual_debt_service(
    loan_amount: float,
    annual_rate: float,
    io_years: int,
    am_years: int,
    year: int,  # 1-indexed
) -> float:
    """Annual debt service for a given year (1-indexed)."""
    if year <= io_years:
        return loan_amount * annual_rate
    r = annual_rate / 12
    if r == 0:
        return loan_amount / am_years
    n = am_years * 12
    monthly = loan_amount * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return monthly * 12


# ---------------------------------------------------------------------------
# Pro Forma Builder
# ---------------------------------------------------------------------------

def build_proforma(deal: DealState) -> Dict[str, Any]:
    pp = deal.property_info.purchase_price
    units = deal.property_info.units or 100
    t12 = deal.t12_data
    asmp = deal.assumptions
    fin = deal.financing
    ex = deal.exit_assumptions
    hold = ex.hold_period_years

    # Financing
    loan_amount = pp * fin.ltv_pct / 100
    closing_costs = pp * 0.01  # 1% closing costs
    total_equity = (pp - loan_amount) + closing_costs
    rate = fin.interest_rate / 100

    # Base revenue (annualized)
    gpr_base = t12.gross_potential_rent or (units * 1050 * 12)

    annual_rows = []
    for yr in range(1, hold + 1):
        rg = (1 + asmp.rent_growth_rate / 100) ** yr
        eg = (1 + asmp.expense_growth_rate / 100) ** yr

        gpr = gpr_base * rg

        # Value-add renovation premium
        va = asmp.value_add
        if va.enabled and yr <= va.absorption_years:
            frac = yr / va.absorption_years
            reno_units = round(va.units_to_renovate * frac)
            gpr += reno_units * va.rent_premium_per_unit * 12

        vac_loss = gpr * asmp.vacancy_rate / 100
        credit_loss = (gpr - vac_loss) * asmp.credit_loss_rate / 100
        other_inc = (t12.other_income or 0) * (1 + asmp.other_income_growth / 100) ** yr
        egi = gpr - vac_loss - credit_loss + other_inc

        # Expenses
        taxes = t12.property_taxes * eg
        insurance = t12.insurance * eg
        mgmt = egi * asmp.management_fee_pct / 100
        maint = t12.maintenance_repairs * eg
        utils = t12.utilities * eg
        payroll = t12.payroll * eg
        gen_admin = t12.general_admin * eg
        marketing = t12.marketing * eg
        capex = units * asmp.capex_reserves_per_unit
        other_exp = (t12.other_expenses or 0) * eg

        total_exp = taxes + insurance + mgmt + maint + utils + payroll + gen_admin + marketing + capex + other_exp
        noi = egi - total_exp

        ds = annual_debt_service(loan_amount, rate, fin.io_period_years, fin.amortization_years, yr)
        levered_cf = noi - ds
        dscr = noi / ds if ds > 0 else None

        annual_rows.append(
            {
                "year": yr,
                "gross_potential_rent": round(gpr),
                "vacancy_loss": round(vac_loss),
                "credit_loss": round(credit_loss),
                "other_income": round(other_inc),
                "egi": round(egi),
                "property_taxes": round(taxes),
                "insurance": round(insurance),
                "management_fee": round(mgmt),
                "maintenance": round(maint),
                "utilities": round(utils),
                "payroll": round(payroll),
                "general_admin": round(gen_admin),
                "marketing": round(marketing),
                "capex_reserves": round(capex),
                "other_expenses": round(other_exp),
                "total_expenses": round(total_exp),
                "noi": round(noi),
                "debt_service": round(ds),
                "levered_cf": round(levered_cf),
                "dscr": round(dscr, 2) if dscr else None,
            }
        )

    # Exit
    exit_noi = annual_rows[-1]["noi"]
    exit_cap = ex.exit_cap_rate / 100
    gross_sale = exit_noi / exit_cap
    sell_costs = gross_sale * ex.selling_costs_pct / 100
    net_sale = gross_sale - sell_costs
    loan_payoff = calculate_loan_balance(
        loan_amount, rate, fin.io_period_years, fin.amortization_years, hold
    )
    net_proceeds = net_sale - loan_payoff

    # Cash flow arrays (year 0 = equity outflow)
    levered_cfs = [-total_equity] + [r["levered_cf"] for r in annual_rows]
    levered_cfs[-1] = levered_cfs[-1] + net_proceeds  # add exit to final year

    unlevered_cfs = [-(pp + closing_costs)] + [r["noi"] for r in annual_rows]
    unlevered_cfs[-1] = unlevered_cfs[-1] + (gross_sale - sell_costs)

    levered_irr = calculate_irr(levered_cfs)
    unlevered_irr = calculate_irr(unlevered_cfs)

    # Equity multiple
    total_lev_in = abs(levered_cfs[0])
    total_lev_out = sum(c for c in levered_cfs[1:])
    levered_em = total_lev_out / total_lev_in if total_lev_in > 0 else None

    total_unlev_in = abs(unlevered_cfs[0])
    total_unlev_out = sum(c for c in unlevered_cfs[1:])
    unlevered_em = total_unlev_out / total_unlev_in if total_unlev_in > 0 else None

    yr1 = annual_rows[0]
    going_in_cap = yr1["noi"] / pp
    noi_margin = yr1["noi"] / yr1["egi"] if yr1["egi"] else None
    coc = yr1["levered_cf"] / total_equity if total_equity > 0 else None

    metrics = {
        "purchase_price": pp,
        "loan_amount": round(loan_amount),
        "equity_required": round(total_equity),
        "closing_costs": round(closing_costs),
        "going_in_cap_rate": round(going_in_cap * 100, 2),
        "stabilized_cap_rate": round(annual_rows[min(2, hold - 1)]["noi"] / pp * 100, 2),
        "year1_noi": yr1["noi"],
        "year1_egi": yr1["egi"],
        "noi_margin": round(noi_margin * 100, 1) if noi_margin else None,
        "dscr_year1": yr1["dscr"],
        "levered_coc": round(coc * 100, 2) if coc else None,
        "unlevered_coc": round(yr1["noi"] / (pp + closing_costs) * 100, 2),
        "levered_irr": round(levered_irr * 100, 2) if levered_irr else None,
        "unlevered_irr": round(unlevered_irr * 100, 2) if unlevered_irr else None,
        "levered_em": round(levered_em, 2) if levered_em else None,
        "unlevered_em": round(unlevered_em, 2) if unlevered_em else None,
        "npv_10pct": round(calculate_npv(levered_cfs, 0.10)),
    }

    exit_summary = {
        "exit_year_noi": exit_noi,
        "exit_cap_rate": ex.exit_cap_rate,
        "gross_sale_price": round(gross_sale),
        "selling_costs": round(sell_costs),
        "net_sale_price": round(net_sale),
        "loan_payoff": round(loan_payoff),
        "net_sale_proceeds": round(net_proceeds),
    }

    return {
        "annual_rows": annual_rows,
        "metrics": metrics,
        "exit": exit_summary,
        "cash_flows": {
            "levered": [round(c) for c in levered_cfs],
            "unlevered": [round(c) for c in unlevered_cfs],
        },
        "loan_amount": round(loan_amount),
        "total_equity": round(total_equity),
    }


