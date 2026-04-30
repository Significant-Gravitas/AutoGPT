import { type Country } from "../../countries";
import { type PlanDef, YEARLY_PRICE_FACTOR } from "../../helpers";

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
  const displayPrice =
    monthlyLocal !== null
      ? isYearly
        ? monthlyLocal * YEARLY_PRICE_FACTOR * 12
        : monthlyLocal
      : null;
  const monthlyEquiv =
    isYearly && monthlyLocal !== null
      ? monthlyLocal * YEARLY_PRICE_FACTOR
      : null;
  const perLabel = isYearly ? "/ year" : "/ month";

  return { displayPrice, monthlyEquiv, perLabel };
}
