"""
Sensitivity analysis: generates heatmap tables by varying two parameters.
"""

from copy import deepcopy
from typing import List, Dict, Any, Callable
from models.deal import DealState


def _set_nested(deal: DealState, param: str, value: float) -> DealState:
    """Set a nested deal parameter by dot-notation string."""
    parts = param.split(".")
    obj = deal
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return deal


def _get_metric(results: Dict, metric: str) -> float:
    """Extract a scalar metric from analysis results."""
    if metric in results.get("metrics", {}):
        return results["metrics"][metric] or 0
    if metric in results.get("waterfall", {}):
        return results["waterfall"][metric] or 0
    return 0


def build_sensitivity_table(
    deal: DealState,
    row_param: str,         # e.g. "property_info.purchase_price"
    col_param: str,         # e.g. "exit_assumptions.exit_cap_rate"
    row_values: List[float],
    col_values: List[float],
    output_metric: str,     # e.g. "lp_irr" or "levered_irr" or "levered_em"
    analyze_fn: Callable,   # function(deal) -> results dict
) -> Dict[str, Any]:
    table = []
    for rv in row_values:
        row = []
        for cv in col_values:
            d = deepcopy(deal)
            _set_nested(d, row_param, rv)
            _set_nested(d, col_param, cv)
            try:
                results = analyze_fn(d)
                val = _get_metric(results, output_metric)
                row.append(round(val, 2) if val is not None else None)
            except Exception:
                row.append(None)
        table.append(row)

    return {
        "row_param": row_param,
        "col_param": col_param,
        "row_values": row_values,
        "col_values": col_values,
        "output_metric": output_metric,
        "table": table,
    }


# Pre-defined sensitivity sets
SENSITIVITY_CONFIGS = {
    "lp_irr_vs_price_x_exit_cap": {
        "row_param": "property_info.purchase_price",
        "col_param": "exit_assumptions.exit_cap_rate",
        "row_label": "Purchase Price ($M)",
        "col_label": "Exit Cap Rate (%)",
        "output_metric": "lp_irr",
        "row_scale": 1_000_000,
        "col_scale": 1,
    },
    "lp_irr_vs_rent_growth_x_vacancy": {
        "row_param": "assumptions.rent_growth_rate",
        "col_param": "assumptions.vacancy_rate",
        "row_label": "Rent Growth (%)",
        "col_label": "Vacancy (%)",
        "output_metric": "lp_irr",
        "row_scale": 1,
        "col_scale": 1,
    },
    "coc_vs_ltv_x_rate": {
        "row_param": "financing.ltv_pct",
        "col_param": "financing.interest_rate",
        "row_label": "LTV (%)",
        "col_label": "Interest Rate (%)",
        "output_metric": "levered_coc",
        "row_scale": 1,
        "col_scale": 1,
    },
    "em_vs_hold_x_exit_cap": {
        "row_param": "exit_assumptions.hold_period_years",
        "col_param": "exit_assumptions.exit_cap_rate",
        "row_label": "Hold Period (yrs)",
        "col_label": "Exit Cap Rate (%)",
        "output_metric": "levered_em",
        "row_scale": 1,
        "col_scale": 1,
    },
}


def generate_all_sensitivities(deal: DealState, analyze_fn: Callable) -> Dict[str, Any]:
    pp = deal.property_info.purchase_price

    tables = {}

    # 1. LP IRR vs Purchase Price × Exit Cap
    tables["lp_irr_vs_price_x_exit_cap"] = build_sensitivity_table(
        deal=deal,
        row_param="property_info.purchase_price",
        col_param="exit_assumptions.exit_cap_rate",
        row_values=[pp * f for f in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]],
        col_values=[4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00],
        output_metric="lp_irr",
        analyze_fn=analyze_fn,
    )

    # 2. LP IRR vs Rent Growth × Vacancy
    tables["lp_irr_vs_rent_x_vacancy"] = build_sensitivity_table(
        deal=deal,
        row_param="assumptions.rent_growth_rate",
        col_param="assumptions.vacancy_rate",
        row_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        col_values=[3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        output_metric="lp_irr",
        analyze_fn=analyze_fn,
    )

    # 3. Cash-on-cash vs LTV × Interest Rate
    tables["coc_vs_ltv_x_rate"] = build_sensitivity_table(
        deal=deal,
        row_param="financing.ltv_pct",
        col_param="financing.interest_rate",
        row_values=[55.0, 60.0, 65.0, 70.0, 75.0],
        col_values=[5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
        output_metric="levered_coc",
        analyze_fn=analyze_fn,
    )

    # 4. Equity Multiple vs Hold Period × Exit Cap
    tables["em_vs_hold_x_exit_cap"] = build_sensitivity_table(
        deal=deal,
        row_param="exit_assumptions.hold_period_years",
        col_param="exit_assumptions.exit_cap_rate",
        row_values=[3, 4, 5, 7, 10],
        col_values=[4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00],
        output_metric="levered_em",
        analyze_fn=analyze_fn,
    )

    return tables
