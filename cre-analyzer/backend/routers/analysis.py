from fastapi import APIRouter, HTTPException
from models.deal import DealState, AnalysisRequest, SolveRequest
from services.underwriting_engine import build_proforma
from services.waterfall_engine import run_waterfall, solve_for_price, solve_for_exit_cap
from services.sensitivity_engine import generate_all_sensitivities

router = APIRouter()


def full_analysis(deal: DealState) -> dict:
    """Run proforma + waterfall and return combined results."""
    pf = build_proforma(deal)

    # Operating levered CFs (years 1..N-1, excluding year 0 and exit)
    lev_cfs = pf["cash_flows"]["levered"]
    annual_ops = [float(c) for c in lev_cfs[1:-1]]  # years 1..N-1
    # Final year operating CF (before exit)
    if len(lev_cfs) > 1:
        annual_ops.append(float(lev_cfs[-1]) - float(pf["exit"]["net_sale_proceeds"]))

    exit_proceeds = float(pf["exit"]["net_sale_proceeds"])

    wf = run_waterfall(
        equity_required=float(pf["total_equity"]),
        annual_levered_cfs=annual_ops,
        exit_proceeds=exit_proceeds,
        config=deal.waterfall_config,
    )

    return {
        "proforma": pf,
        "waterfall": wf,
        "metrics": pf["metrics"],
        "lp_irr": wf["lp_irr"],
        "levered_irr": pf["metrics"].get("levered_irr"),
        "levered_em": pf["metrics"].get("levered_em"),
        "levered_coc": pf["metrics"].get("levered_coc"),
    }


@router.post("/run")
def run_analysis(req: AnalysisRequest):
    try:
        results = full_analysis(req.deal)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sensitivity")
def run_sensitivity(req: AnalysisRequest):
    try:
        tables = generate_all_sensitivities(req.deal, full_analysis)
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solve")
def run_solve(req: SolveRequest):
    try:
        if req.solve_for == "purchase_price":
            result = solve_for_price(req.deal, req.target_lp_irr, build_proforma)
            return {"solve_for": "purchase_price", "value": result, "target_lp_irr": req.target_lp_irr}
        elif req.solve_for == "exit_cap_rate":
            result = solve_for_exit_cap(req.deal, req.target_lp_irr, build_proforma)
            return {"solve_for": "exit_cap_rate", "value": result, "target_lp_irr": req.target_lp_irr}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown solve_for: {req.solve_for}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
