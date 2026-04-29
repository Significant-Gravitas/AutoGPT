import { type Country } from "../../countries";
import { type PlanDef, YEARLY_DISCOUNT } from "../../helpers";

interface PriceComputationArgs {
  plan: PlanDef;
  country: Country;
  isYearly: boolean;
}

export function computePlanPricing({
  plan,
  country,
  isYearly,
}: PriceComputationArgs) {
  const monthlyLocal =
    plan.usdMonthly !== null ? plan.usdMonthly * country.rate : null;
  const displayPrice = monthlyLocal
    ? isYearly
      ? monthlyLocal * YEARLY_DISCOUNT * 12
      : monthlyLocal
    : null;
  const monthlyEquiv =
    isYearly && monthlyLocal ? monthlyLocal * YEARLY_DISCOUNT : null;
  const perLabel = isYearly ? "/ year" : "/ month";

  return { displayPrice, monthlyEquiv, perLabel };
}
