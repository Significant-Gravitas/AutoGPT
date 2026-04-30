import { describe, expect, test } from "vitest";
import { type Country } from "../../countries";
import { type PlanDef, PLAN_KEYS, YEARLY_PRICE_FACTOR } from "../../helpers";
import { computePlanPricing } from "./helpers";

const usdCountry: Country = {
  name: "United States",
  flag: "🇺🇸",
  countryCode: "US",
  currencyCode: "USD",
  symbol: "$",
  rate: 1,
};

const brlCountry: Country = {
  name: "Brazil",
  flag: "🇧🇷",
  countryCode: "BR",
  currencyCode: "BRL",
  symbol: "R$",
  rate: 5,
};

function makePlan(overrides: Partial<PlanDef> = {}): PlanDef {
  return {
    key: PLAN_KEYS.PRO,
    name: "Pro",
    usage: "1x",
    usdMonthly: 50,
    description: "",
    features: [],
    cta: "Get Pro",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
    ...overrides,
  };
}

describe("computePlanPricing", () => {
  test("monthly USD passes through the country rate", () => {
    const { displayPrice, monthlyEquiv, perLabel } = computePlanPricing({
      plan: makePlan(),
      country: usdCountry,
      isYearly: false,
    });
    expect(displayPrice).toBe(50);
    expect(monthlyEquiv).toBeNull();
    expect(perLabel).toBe("/ month");
  });

  test("yearly applies the price factor and multiplies by 12", () => {
    const { displayPrice, monthlyEquiv, perLabel } = computePlanPricing({
      plan: makePlan(),
      country: usdCountry,
      isYearly: true,
    });
    expect(displayPrice).toBe(50 * YEARLY_PRICE_FACTOR * 12);
    expect(monthlyEquiv).toBe(50 * YEARLY_PRICE_FACTOR);
    expect(perLabel).toBe("/ year");
  });

  test("non-USD currency multiplies by the country rate", () => {
    const { displayPrice } = computePlanPricing({
      plan: makePlan({ usdMonthly: 50 }),
      country: brlCountry,
      isYearly: false,
    });
    expect(displayPrice).toBe(250);
  });

  test("usdMonthly === null returns null prices for both branches", () => {
    const monthly = computePlanPricing({
      plan: makePlan({ usdMonthly: null }),
      country: usdCountry,
      isYearly: false,
    });
    const yearly = computePlanPricing({
      plan: makePlan({ usdMonthly: null }),
      country: usdCountry,
      isYearly: true,
    });
    expect(monthly.displayPrice).toBeNull();
    expect(monthly.monthlyEquiv).toBeNull();
    expect(yearly.displayPrice).toBeNull();
    expect(yearly.monthlyEquiv).toBeNull();
  });

  test("usdMonthly === 0 renders 0, not 'Contact us'", () => {
    const { displayPrice, monthlyEquiv } = computePlanPricing({
      plan: makePlan({ usdMonthly: 0 }),
      country: usdCountry,
      isYearly: true,
    });
    expect(displayPrice).toBe(0);
    expect(monthlyEquiv).toBe(0);
  });
});
