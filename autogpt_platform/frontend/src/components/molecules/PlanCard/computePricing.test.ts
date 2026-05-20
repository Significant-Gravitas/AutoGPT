import { describe, expect, test } from "vitest";
import { type Country } from "./countries";
import { type PlanDef, PLAN_KEYS, YEARLY_PRICE_FACTOR } from "./plans";
import { computePlanPricing } from "./computePricing";

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
  test("monthly USD: primary is the monthly price, no discount, charged today equals monthly", () => {
    const result = computePlanPricing({
      plan: makePlan(),
      country: usdCountry,
      isYearly: false,
    });
    expect(result.primaryPrice).toBe(50);
    expect(result.chargedToday).toBe(50);
    expect(result.monthlyOriginal).toBeNull();
    expect(result.discountPercent).toBeNull();
  });

  test("yearly USD: primary is the monthly-equivalent, charged today is the annual total", () => {
    const result = computePlanPricing({
      plan: makePlan(),
      country: usdCountry,
      isYearly: true,
    });
    expect(result.primaryPrice).toBe(50 * YEARLY_PRICE_FACTOR);
    expect(result.chargedToday).toBe(50 * YEARLY_PRICE_FACTOR * 12);
    expect(result.monthlyOriginal).toBe(50);
    expect(result.discountPercent).toBe(15);
  });

  test("non-USD currency multiplies by the country rate", () => {
    const result = computePlanPricing({
      plan: makePlan({ usdMonthly: 50 }),
      country: brlCountry,
      isYearly: false,
    });
    expect(result.primaryPrice).toBe(250);
    expect(result.chargedToday).toBe(250);
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
    expect(monthly.primaryPrice).toBeNull();
    expect(monthly.chargedToday).toBeNull();
    expect(monthly.monthlyOriginal).toBeNull();
    expect(monthly.discountPercent).toBeNull();
    expect(yearly.primaryPrice).toBeNull();
    expect(yearly.chargedToday).toBeNull();
    expect(yearly.monthlyOriginal).toBeNull();
    expect(yearly.discountPercent).toBeNull();
  });

  test("usdMonthly === 0 renders 0, not 'Contact us'", () => {
    const result = computePlanPricing({
      plan: makePlan({ usdMonthly: 0 }),
      country: usdCountry,
      isYearly: true,
    });
    expect(result.primaryPrice).toBe(0);
    expect(result.chargedToday).toBe(0);
    expect(result.discountPercent).toBeNull();
  });

  test("explicit usdYearly overrides the YEARLY_PRICE_FACTOR estimate", () => {
    const result = computePlanPricing({
      plan: makePlan({ usdMonthly: 50, usdYearly: 480 }),
      country: usdCountry,
      isYearly: true,
    });
    expect(result.chargedToday).toBe(480);
    expect(result.primaryPrice).toBe(40);
    expect(result.monthlyOriginal).toBe(50);
    expect(result.discountPercent).toBe(20);
  });
});
