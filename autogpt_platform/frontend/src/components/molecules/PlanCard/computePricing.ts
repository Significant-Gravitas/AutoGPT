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
  const monthlyEquiv =
    isYearly && yearlyLocal !== null ? yearlyLocal / 12 : null;

  const primaryPrice = isYearly ? monthlyEquiv : monthlyLocal;
  const chargedToday = isYearly ? yearlyLocal : monthlyLocal;
  const monthlyOriginal =
    isYearly && monthlyLocal !== null && monthlyEquiv !== null
      ? monthlyLocal
      : null;
  const discountPercent =
    isYearly &&
    monthlyLocal !== null &&
    monthlyLocal > 0 &&
    monthlyEquiv !== null
      ? Math.round((1 - monthlyEquiv / monthlyLocal) * 100)
      : null;

  return {
    primaryPrice,
    monthlyOriginal,
    chargedToday,
    discountPercent,
  };
}
