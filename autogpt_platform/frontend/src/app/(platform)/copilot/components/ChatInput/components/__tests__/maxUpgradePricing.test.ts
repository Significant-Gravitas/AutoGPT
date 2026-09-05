import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { describe, expect, it } from "vitest";

import {
  formatUpgradePrice,
  getMaxUpgradePricing,
  getMaxUpgradeUnavailableReason,
} from "../ConnectionPicker/maxUpgrade";

function subscription(
  overrides: Partial<SubscriptionStatusResponse> = {},
): SubscriptionStatusResponse {
  return {
    tier: "PRO",
    monthly_cost: 2700,
    tier_costs: { PRO: 2700, MAX: 18300 },
    tier_costs_yearly: { PRO: 27000, MAX: 183000 },
    billing_cycle: "monthly",
    proration_credit_cents: 900,
    has_active_stripe_subscription: true,
    ...overrides,
  };
}

describe("Max upgrade pricing", () => {
  it("formats whole dollar prices compactly and preserves two digits for cents", () => {
    expect(formatUpgradePrice(18300)).toBe("$183");
    expect(formatUpgradePrice(18345)).toBe("$183.45");
    expect(formatUpgradePrice(183000)).toBe("$1,830");
    expect(formatUpgradePrice(251940)).toBe("$2,519.40");
  });

  it("uses configured prices instead of plan defaults", () => {
    expect(getMaxUpgradePricing(subscription())).toEqual({
      cycle: "monthly",
      currentCents: 2700,
      maxCents: 18300,
    });
  });

  it("preserves annual totals rather than substituting monthly prices", () => {
    expect(
      getMaxUpgradePricing(subscription({ billing_cycle: "yearly" })),
    ).toEqual({
      cycle: "yearly",
      currentCents: 27000,
      maxCents: 183000,
    });
  });

  it("does not invent annual pricing when only monthly is configured", () => {
    const data = subscription({
      billing_cycle: "yearly",
      tier_costs_yearly: {},
    });
    expect(getMaxUpgradePricing(data)?.maxCents).toBeNull();
    expect(getMaxUpgradeUnavailableReason(data)).toMatch(
      /yearly.*unavailable/i,
    );
  });

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
    "treats invalid target price %s as unavailable, including missing-price zero",
    (price) => {
      const data = subscription({ tier_costs: { PRO: 2700, MAX: price } });
      expect(getMaxUpgradePricing(data)?.maxCents).toBeNull();
      expect(getMaxUpgradeUnavailableReason(data)).toMatch(/unavailable/i);
    },
  );

  it("requires account data, an active Pro subscription and no scheduled change", () => {
    expect(getMaxUpgradePricing(undefined)).toBeNull();
    expect(getMaxUpgradeUnavailableReason(undefined)).toBeTruthy();
    expect(getMaxUpgradeUnavailableReason(subscription())).toBeNull();
    expect(
      getMaxUpgradeUnavailableReason(subscription({ tier: "MAX" })),
    ).toBeTruthy();
    expect(
      getMaxUpgradeUnavailableReason(
        subscription({ has_active_stripe_subscription: false }),
      ),
    ).toMatch(/billing/i);
    expect(
      getMaxUpgradeUnavailableReason(subscription({ pending_tier: "NO_TIER" })),
    ).toMatch(/scheduled/i);
  });
});
