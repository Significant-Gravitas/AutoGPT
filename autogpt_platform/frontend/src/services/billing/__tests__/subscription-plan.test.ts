import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { describe, expect, it } from "vitest";
import { getSubscriptionPlan, PLAN_LABEL } from "../subscription-plan";

const EFFECTIVE_AT = new Date("2026-11-01T00:00:00Z");

function makeSubscription(
  overrides: Partial<SubscriptionStatusResponse> = {},
): SubscriptionStatusResponse {
  return {
    tier: "PRO",
    monthly_cost: 5000,
    tier_costs: { PRO: 5000, MAX: 32000 },
    proration_credit_cents: 0,
    ...overrides,
  };
}

const NOTHING_PENDING = {
  pendingTier: null,
  pendingEffectiveAt: null,
  pendingBillingCycle: null,
  isPendingCancel: false,
  isPendingDowngrade: false,
  isPendingCycleSwitch: false,
};

describe("getSubscriptionPlan", () => {
  it("reports nothing while the subscription is loading or failed", () => {
    expect(getSubscriptionPlan(undefined)).toEqual({
      tier: null,
      billingCycle: "monthly",
      isAdminManaged: false,
      ...NOTHING_PENDING,
    });
  });

  it("reports an active plan with nothing scheduled", () => {
    expect(
      getSubscriptionPlan(makeSubscription({ billing_cycle: "yearly" })),
    ).toEqual({
      tier: "PRO",
      billingCycle: "yearly",
      isAdminManaged: false,
      ...NOTHING_PENDING,
    });
  });

  it("treats a pending NO_TIER as a cancellation", () => {
    const plan = getSubscriptionPlan(
      makeSubscription({
        pending_tier: "NO_TIER",
        pending_tier_effective_at: EFFECTIVE_AT,
      }),
    );

    expect(plan).toMatchObject({
      pendingTier: "NO_TIER",
      pendingEffectiveAt: EFFECTIVE_AT,
      isPendingCancel: true,
      isPendingDowngrade: false,
      isPendingCycleSwitch: false,
    });
  });

  it("treats a pending lower paid tier as a downgrade", () => {
    const plan = getSubscriptionPlan(
      makeSubscription({
        tier: "MAX",
        pending_tier: "PRO",
        pending_tier_effective_at: EFFECTIVE_AT,
        pending_billing_cycle: "monthly",
      }),
    );

    expect(plan).toMatchObject({
      pendingTier: "PRO",
      pendingBillingCycle: "monthly",
      isPendingCancel: false,
      isPendingDowngrade: true,
      isPendingCycleSwitch: false,
    });
  });

  it("treats the same tier on another cycle as a cycle switch, not a downgrade", () => {
    const plan = getSubscriptionPlan(
      makeSubscription({
        billing_cycle: "yearly",
        pending_tier: "PRO",
        pending_billing_cycle: "monthly",
      }),
    );

    expect(plan).toMatchObject({
      isPendingCancel: false,
      isPendingDowngrade: false,
      isPendingCycleSwitch: true,
    });
  });

  it("ignores a same-tier schedule on the current cycle", () => {
    const plan = getSubscriptionPlan(
      makeSubscription({
        pending_tier: "PRO",
        pending_billing_cycle: "monthly",
      }),
    );

    expect(plan).toMatchObject({
      isPendingCancel: false,
      isPendingDowngrade: false,
      isPendingCycleSwitch: false,
    });
  });

  it("hides any schedule on an admin-managed (ENTERPRISE) account", () => {
    const plan = getSubscriptionPlan(
      makeSubscription({
        tier: "ENTERPRISE",
        pending_tier: "NO_TIER",
        pending_tier_effective_at: EFFECTIVE_AT,
      }),
    );

    expect(plan).toMatchObject({
      tier: "ENTERPRISE",
      isAdminManaged: true,
      pendingTier: null,
      isPendingCancel: false,
      isPendingDowngrade: false,
    });
  });
});

describe("PLAN_LABEL", () => {
  it("names every self-serve and managed tier", () => {
    expect(PLAN_LABEL).toEqual({
      NO_TIER: "No active subscription",
      BASIC: "Basic",
      PRO: "Pro",
      MAX: "Max",
      BUSINESS: "Team",
      ENTERPRISE: "Enterprise",
    });
  });
});
