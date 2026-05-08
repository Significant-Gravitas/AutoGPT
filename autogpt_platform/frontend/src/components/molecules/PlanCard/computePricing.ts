import { type Country } from "./countries";
import { type PlanDef, YEARLY_PRICE_FACTOR } from "./plans";

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
  // Prefer the explicit yearly price (API-driven) over the YEARLY_PRICE_FACTOR
  // estimate so the displayed amount matches what Stripe will actually charge.
  const yearlyLocal =
    plan.usdYearly != null
      ? plan.usdYearly * country.rate
      : monthlyLocal !== null
        ? monthlyLocal * YEARLY_PRICE_FACTOR * 12
        : null;
  const displayPrice = isYearly ? yearlyLocal : monthlyLocal;
  const monthlyEquiv =
    isYearly && yearlyLocal !== null ? yearlyLocal / 12 : null;
  const perLabel = isYearly ? "/ year" : "/ month";

  return { displayPrice, monthlyEquiv, perLabel };
}
