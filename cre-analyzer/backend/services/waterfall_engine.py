"""
LP/GP Waterfall Engine.

Waterfall order per distribution event:
  1. Return of capital (pro-rata LP/GP)
  2. Preferred return to LP (cumulative, on unreturned capital)
  3. GP catch-up (to GP until GP holds target_promote_pct% of total profits)
  4. Remaining splits by IRR-based promote tiers (solved at exit)
"""

from typing import List, Dict, Any, Optional, Tuple
from models.deal import WaterfallConfig, WaterfallTier
from services.underwriting_engine import calculate_irr


def _solve_lp_amount_for_irr(
    lp_invested: float,
    prior_lp_dists: List[float],
    target_irr: float,
    max_available: float,
) -> float:
    """Binary search: how much LP needs to reach target_irr at exit."""
    if target_irr <= 0:
        return 0.0

    def lp_irr_at(lp_alloc: float) -> float:
        cfs = [-lp_invested] + prior_lp_dists[:-1] + [prior_lp_dists[-1] + lp_alloc]
        irr = calculate_irr(cfs)
        return irr if irr is not None else -1.0

    # Check if max_available gets LP past target
    if lp_irr_at(max_available) < target_irr:
        return max_available

    # Binary search
    lo, hi = 0.0, max_available
    for _ in range(60):
        mid = (lo + hi) / 2
        if lp_irr_at(mid) < target_irr:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1:
            break
    return (lo + hi) / 2


def run_waterfall(
    equity_required: float,
    annual_levered_cfs: List[float],
    exit_proceeds: float,
    config: WaterfallConfig,
) -> Dict[str, Any]:
    lp_pct = config.lp_equity_pct / 100
    gp_pct = config.gp_equity_pct / 100
    lp_invested = equity_required * lp_pct
    gp_invested = equity_required * gp_pct
    pref_rate = config.preferred_return / 100

    lp_cap_returned = 0.0
    gp_cap_returned = 0.0
    lp_pref_accrued = 0.0
    lp_pref_paid = 0.0
    gp_catchup_paid = 0.0
    lp_promote_paid = 0.0
    gp_promote_paid = 0.0

    lp_dists_by_year: List[float] = []
    gp_dists_by_year: List[float] = []
    yearly_results: List[Dict] = []

    all_events = list(annual_levered_cfs) + [exit_proceeds]
    n_operating = len(annual_levered_cfs)

    for idx, cf in enumerate(all_events):
        is_exit = idx == len(all_events) - 1
        year = idx + 1

        # Accrue preferred return on unreturned LP capital
        lp_unreturned = lp_invested - lp_cap_returned
        if config.pref_compounding:
            lp_pref_accrued = lp_pref_accrued * (1 + pref_rate) + lp_unreturned * pref_rate
        else:
            lp_pref_accrued += lp_unreturned * pref_rate

        distributable = max(0.0, cf)
        remaining = distributable
        lp_dist = 0.0
        gp_dist = 0.0

        tier_roc_lp = 0.0
        tier_roc_gp = 0.0
        tier_pref = 0.0
        tier_catchup = 0.0
        tier_promote_lp = 0.0
        tier_promote_gp = 0.0

        if remaining > 0:
            # --- Tier 1: Return of Capital ---
            lp_cap_owed = max(0.0, lp_invested - lp_cap_returned)
            gp_cap_owed = max(0.0, gp_invested - gp_cap_returned)
            total_cap_owed = lp_cap_owed + gp_cap_owed

            if total_cap_owed > 0:
                roc_payment = min(remaining, total_cap_owed)
                lp_roc = roc_payment * (lp_cap_owed / total_cap_owed)
                gp_roc = roc_payment * (gp_cap_owed / total_cap_owed)
                lp_dist += lp_roc
                gp_dist += gp_roc
                remaining -= roc_payment
                lp_cap_returned += lp_roc
                gp_cap_returned += gp_roc
                tier_roc_lp = lp_roc
                tier_roc_gp = gp_roc

            # --- Tier 2: Preferred Return to LP ---
            lp_pref_owed = max(0.0, lp_pref_accrued - lp_pref_paid)
            if lp_pref_owed > 0 and remaining > 0:
                pref_pmt = min(remaining, lp_pref_owed)
                lp_dist += pref_pmt
                remaining -= pref_pmt
                lp_pref_paid += pref_pmt
                tier_pref = pref_pmt

            # --- Tier 3: GP Catch-up ---
            if config.gp_catchup and remaining > 0:
                total_profits = lp_pref_paid + gp_catchup_paid + lp_promote_paid + gp_promote_paid
                # GP should receive gp_target_promote_pct% of cumulative profits (non-capital)
                gp_target = total_profits * (config.gp_target_promote_pct / 100) / (
                    1 - config.gp_target_promote_pct / 100
                )
                catchup_owed = max(0.0, gp_target - gp_catchup_paid)
                if catchup_owed > 0 and remaining > 0:
                    catchup_pmt = min(remaining, catchup_owed)
                    gp_dist += catchup_pmt
                    remaining -= catchup_pmt
                    gp_catchup_paid += catchup_pmt
                    tier_catchup = catchup_pmt

            # --- Tier 4: Promote splits ---
            if remaining > 0:
                if is_exit:
                    lp_p, gp_p = _allocate_by_irr_tiers(
                        lp_invested=lp_invested,
                        prior_lp_dists=lp_dists_by_year + [lp_dist],
                        remaining=remaining,
                        tiers=sorted(config.tiers, key=lambda t: t.irr_min),
                    )
                else:
                    # During hold: use first tier's split for operating promotes
                    tier0 = sorted(config.tiers, key=lambda t: t.irr_min)[0]
                    lp_p = remaining * tier0.lp_split / 100
                    gp_p = remaining * tier0.gp_split / 100

                lp_dist += lp_p
                gp_dist += gp_p
                remaining -= lp_p + gp_p
                lp_promote_paid += lp_p
                gp_promote_paid += gp_p
                tier_promote_lp = lp_p
                tier_promote_gp = gp_p

        lp_dists_by_year.append(lp_dist)
        gp_dists_by_year.append(gp_dist)

        yearly_results.append(
            {
                "year": year,
                "is_exit": is_exit,
                "distributable": round(distributable),
                "lp_distribution": round(lp_dist),
                "gp_distribution": round(gp_dist),
                "tier_roc_lp": round(tier_roc_lp),
                "tier_roc_gp": round(tier_roc_gp),
                "tier_preferred_return": round(tier_pref),
                "tier_gp_catchup": round(tier_catchup),
                "tier_lp_promote": round(tier_promote_lp),
                "tier_gp_promote": round(tier_promote_gp),
            }
        )

    # Final IRR / EM
    lp_cfs = [-lp_invested] + lp_dists_by_year
    gp_cfs = [-gp_invested] + gp_dists_by_year
    lp_irr = calculate_irr(lp_cfs)
    gp_irr = calculate_irr(gp_cfs)

    total_lp_out = sum(lp_dists_by_year)
    total_gp_out = sum(gp_dists_by_year)
    lp_em = total_lp_out / lp_invested if lp_invested > 0 else None
    gp_em = total_gp_out / gp_invested if gp_invested > 0 else None
    gp_promote = gp_catchup_paid + gp_promote_paid

    return {
        "yearly": yearly_results,
        "lp_total_distributions": round(total_lp_out),
        "gp_total_distributions": round(total_gp_out),
        "lp_irr": round(lp_irr * 100, 2) if lp_irr else None,
        "gp_irr": round(gp_irr * 100, 2) if gp_irr else None,
        "lp_em": round(lp_em, 2) if lp_em else None,
        "gp_em": round(gp_em, 2) if gp_em else None,
        "gp_promote_earned": round(gp_promote),
        "lp_invested": round(lp_invested),
        "gp_invested": round(gp_invested),
        "pref_paid": round(lp_pref_paid),
    }


def _allocate_by_irr_tiers(
    lp_invested: float,
    prior_lp_dists: List[float],
    remaining: float,
    tiers: List[WaterfallTier],
) -> Tuple[float, float]:
    """Split remaining exit proceeds across IRR-based tiers."""
    lp_alloc = 0.0
    gp_alloc = 0.0

    for i, tier in enumerate(tiers):
        if remaining <= 0:
            break

        irr_max = tier.irr_max / 100
        lp_split = tier.lp_split / 100
        gp_split = tier.gp_split / 100

        is_last = i == len(tiers) - 1 or tier.irr_max >= 999

        if is_last:
            lp_alloc += remaining * lp_split
            gp_alloc += remaining * gp_split
            remaining = 0
            break

        # How much does LP need to reach irr_max?
        lp_needed = _solve_lp_amount_for_irr(
            lp_invested=lp_invested,
            prior_lp_dists=prior_lp_dists,
            target_irr=irr_max,
            max_available=remaining * lp_split,
        )
        # Total to distribute at this tier
        total_tier = lp_needed / lp_split if lp_split > 0 else 0

        if total_tier >= remaining:
            # All remaining goes at this tier's split
            lp_alloc += remaining * lp_split
            gp_alloc += remaining * gp_split
            remaining = 0
        else:
            lp_alloc += total_tier * lp_split
            gp_alloc += total_tier * gp_split
            remaining -= total_tier
            # Update prior_lp_dists for next tier
            if prior_lp_dists:
                prior_lp_dists = prior_lp_dists[:-1] + [
                    prior_lp_dists[-1] + total_tier * lp_split
                ]

    return lp_alloc, gp_alloc


def solve_for_price(
    deal: "DealState",
    target_lp_irr: float,
    build_fn,
) -> Optional[float]:
    """Binary search for purchase price that yields target LP IRR."""
    from copy import deepcopy

    lo, hi = 1_000_000.0, 200_000_000.0

    def lp_irr_at_price(price: float) -> float:
        d = deepcopy(deal)
        d.property_info.purchase_price = price
        pf = build_fn(d)
        wf = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=pf["cash_flows"]["levered"][1:-1],
            exit_proceeds=pf["exit"]["net_sale_proceeds"],
            config=d.waterfall_config,
        )
        return wf["lp_irr"] or 0

    for _ in range(60):
        mid = (lo + hi) / 2
        irr = lp_irr_at_price(mid)
        if irr < target_lp_irr:
            hi = mid  # lower price needed to improve IRR
        else:
            lo = mid
        if hi - lo < 1000:
            break

    return round((lo + hi) / 2)


def solve_for_exit_cap(
    deal: "DealState",
    target_lp_irr: float,
    build_fn,
) -> Optional[float]:
    """Binary search for exit cap rate that yields target LP IRR."""
    from copy import deepcopy

    lo, hi = 3.0, 10.0

    def lp_irr_at_cap(cap: float) -> float:
        d = deepcopy(deal)
        d.exit_assumptions.exit_cap_rate = cap
        pf = build_fn(d)
        wf = run_waterfall(
            equity_required=pf["total_equity"],
            annual_levered_cfs=pf["cash_flows"]["levered"][1:-1],
            exit_proceeds=pf["exit"]["net_sale_proceeds"],
            config=d.waterfall_config,
        )
        return wf["lp_irr"] or 0

    for _ in range(60):
        mid = (lo + hi) / 2
        irr = lp_irr_at_cap(mid)
        if irr < target_lp_irr:
            hi = mid  # lower exit cap = higher proceeds
        else:
            lo = mid
        if hi - lo < 0.001:
            break

    return round((lo + hi) / 2, 2)
