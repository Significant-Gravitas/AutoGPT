import type { Subscription, TrialStatus } from "../../../lib/api/billing";
import { formatMoney, planSummary, usageMeters } from "../helpers";

// Midday UTC; `pnpm test` runs in UTC, so the day is fixed.
const DEC_19 = "2026-12-19T12:00:00Z";

function subscription(overrides: Partial<Subscription> = {}): Subscription {
  return {
    tier: "PRO",
    monthly_cost: 5000,
    billing_cycle: "monthly",
    current_period_end: Date.parse(DEC_19) / 1000,
    pending_tier: null,
    pending_tier_effective_at: null,
    pending_billing_cycle: null,
    ...overrides,
  };
}

function trial(overrides: Partial<TrialStatus> = {}): TrialStatus {
  return {
    active: true,
    ends_at: DEC_19,
    cancel_at_period_end: false,
    allowance_used_percent: 18,
    ...overrides,
  };
}

describe("planSummary", () => {
  it("names a paid plan, when it renews and what it costs", () => {
    expect(planSummary(subscription(), undefined)).toEqual({
      name: "Pro",
      badge: { label: "Active", tone: "active" },
      headline: "Renews Sat, Dec 19",
      detail: "$50.00 a month.",
    });
  });

  it("says a yearly plan's price by the year", () => {
    const yearly = subscription({ monthly_cost: 51000, billing_cycle: "yearly" });

    expect(planSummary(yearly, undefined).detail).toBe("$510.00 a year.");
  });

  it("says when a cancelled plan ends", () => {
    const ending = subscription({
      pending_tier: "NO_TIER",
      pending_tier_effective_at: DEC_19,
    });

    expect(planSummary(ending, undefined)).toMatchObject({
      badge: { label: "Ends Dec 19", tone: "ending" },
      headline: "Your plan ends Sat, Dec 19",
    });
  });

  it("says when a plan changes to another", () => {
    const changing = subscription({
      tier: "MAX",
      pending_tier: "PRO",
      pending_tier_effective_at: DEC_19,
    });

    expect(planSummary(changing, undefined).headline).toBe(
      "Changes to Pro on Sat, Dec 19",
    );
  });

  it("says when a yearly plan switches to monthly billing", () => {
    const switching = subscription({
      billing_cycle: "yearly",
      pending_tier: "PRO",
      pending_tier_effective_at: DEC_19,
      pending_billing_cycle: "monthly",
    });

    expect(planSummary(switching, undefined).headline).toBe(
      "Switches to monthly billing on Sat, Dec 19",
    );
  });

  it("never calls an account with no plan active", () => {
    expect(planSummary(subscription({ tier: "NO_TIER" }), undefined)).toEqual({
      name: "No plan",
      badge: { label: "Inactive", tone: "ending" },
      headline: null,
      detail: null,
    });
  });

  it("shows a hand-set plan with no price or date", () => {
    const enterprise = subscription({
      tier: "ENTERPRISE",
      monthly_cost: 0,
      current_period_end: null,
    });

    expect(planSummary(enterprise, undefined)).toMatchObject({
      name: "Enterprise",
      headline: null,
      detail: null,
    });
  });

  it("says when a trial ends", () => {
    expect(planSummary(subscription({ tier: "TRIAL" }), trial())).toMatchObject({
      name: "Free trial",
      badge: { label: "Trial", tone: "trial" },
      headline: "Your trial ends Sat, Dec 19",
    });
  });

  it("says a cancelled trial charges nothing", () => {
    const cancelled = trial({ cancel_at_period_end: true });

    expect(planSummary(subscription({ tier: "TRIAL" }), cancelled)).toMatchObject({
      headline: "Your trial is cancelled",
      detail: "You keep access until Sat, Dec 19. Nothing will be charged.",
    });
  });
});

describe("usageMeters", () => {
  it("shows the trial allowance, then the week, then the day", () => {
    const meters = usageMeters(
      {
        weekly: { percent_used: 36.4, resets_at: DEC_19 },
        daily: { percent_used: 42, resets_at: DEC_19 },
      },
      trial(),
    );

    expect(meters.map(({ label }) => label)).toEqual([
      "Trial allowance",
      "This week",
      "Today",
    ]);
    expect(meters[1].note).toBe("36% used");
    expect(meters[2].note).toMatch(/^Resets at /);
  });

  it("shows nothing it has no numbers for", () => {
    expect(usageMeters({ weekly: null, daily: null }, undefined)).toEqual([]);
  });
});

describe("formatMoney", () => {
  it("writes cents as dollars", () => {
    expect(formatMoney(2000)).toBe("$20.00");
    expect(formatMoney(5)).toBe("$0.05");
  });
});
