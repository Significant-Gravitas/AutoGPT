import { maxUpgradePreview } from "@/app/(platform)/settings/billing/__tests__/max-upgrade.fixture";
export type Tier =
  | "trial"
  | "pro"
  | "max"
  | "team"
  | "enterprise"
  | "basic"
  | "none";
export type State =
  | "active"
  | "daily"
  | "weekly"
  | "both"
  | "total"
  | "expired"
  | "canceled"
  | "cancel_pending"
  | "payment_failed"
  | "fresh"
  | "unavailable";
export interface Experience {
  tier: Tier;
  state: State;
  linkedProvider?: boolean;
  usage?: { daily: number; weekly: number };
}
export let fixture: Experience = { tier: "trial", state: "total" };
export const setFixture = (value: Experience) => {
  fixture = value;
};
export const tiers = {
  trial: "TRIAL",
  pro: "PRO",
  max: "MAX",
  team: "BUSINESS",
  enterprise: "ENTERPRISE",
  basic: "BASIC",
  none: "NO_TIER",
};
export const noop = () => {};
export const dailyReset = "2026-10-07T21:00:00Z";
export const weeklyReset = "2026-10-12T13:00:00Z";
export function usage() {
  if (fixture.state === "unavailable") return undefined;
  const exhaustedTrial = fixture.tier === "trial" && fixture.state === "total";
  return {
    tier: tiers[fixture.tier],
    daily: {
      percent_used:
        fixture.usage?.daily ??
        (fixture.state === "fresh"
          ? 0
          : exhaustedTrial || ["daily", "both"].includes(fixture.state)
            ? 100
            : 38),
      resets_at: dailyReset,
    },
    weekly: {
      percent_used:
        fixture.usage?.weekly ??
        (fixture.state === "fresh"
          ? 0
          : ["weekly", "both"].includes(fixture.state)
            ? 100
            : 62),
      resets_at: weeklyReset,
    },
  };
}
export function trial() {
  const ended = ["expired", "canceled", "payment_failed"].includes(
    fixture.state,
  );
  const hadTrial =
    fixture.tier === "trial" || fixture.state === "payment_failed";
  if (!hadTrial)
    return {
      eligible: false,
      converted: fixture.tier !== "none",
      active: false,
      status: "not_started",
      offer: null,
    };
  return {
    eligible: false,
    converted: false,
    active: !ended,
    usage_policy: "lifetime",
    status:
      fixture.state === "canceled"
        ? "canceled"
        : fixture.state === "payment_failed"
          ? "past_due"
          : ended
            ? "expired"
            : "trialing",
    ends_at: "2026-10-12T12:00:00Z",
    allowance_used_percent: fixture.state === "active" ? 64 : 100,
    cancel_at_period_end: fixture.state === "cancel_pending",
    payment_due:
      fixture.state === "payment_failed"
        ? {
            amount_due: 5000,
            currency: "usd",
            payment_url: "https://invoice.stripe.com/i/fixture",
          }
        : null,
    offer: {
      token: "fixture",
      version: "fixture",
      duration_days: 7,
      tier: "PRO",
      billing_cycle: "monthly",
      unit_amount: 5000,
      currency: "usd",
      onboarding_credit_amount: 300,
    },
  };
}

export const activationTerms = {
  plan: "PRO" as const,
  price_id: "fixture-pro-monthly",
  accepted_offer_token: "fixture-offer",
  amount_due: 5000,
  currency: "usd",
  billing_interval: "month" as const,
  billing_interval_count: 1,
  renewal_unit_amount: 5000,
  renewal_terms:
    "Renews automatically every month until canceled. Cancel in Billing before your next payment.",
  renewal_discounts: [],
  renewal_tax: {
    automatic: true,
    price_tax_behavior: "exclusive" as const,
    rates: [],
  },
  trial_ends_at: new Date("2026-10-12T12:00:00Z"),
  period_starts_at: new Date("2026-10-07T12:00:00Z"),
  next_payment_at: new Date("2026-11-07T12:00:00Z"),
  tax_amount: 0,
  expires_at: new Date("2026-10-07T12:15:00Z"),
};
const previewStart = new Date("2026-10-07T12:00:00Z").getTime() / 1000;
const previewEnd = new Date("2026-11-05T12:00:00Z").getTime() / 1000;
export const upgradeReview = {
  required: true,
  isLoading: false,
  error: "",
  retry: noop,
  invalidate: noop,
  preview: maxUpgradePreview({
    preview_id: "fixture-max-preview",
    proration_date: previewStart,
    period_end: previewEnd,
    expires_at: previewStart + 900,
  }),
};
