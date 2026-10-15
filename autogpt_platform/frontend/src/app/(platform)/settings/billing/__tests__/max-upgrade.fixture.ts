import type { SubscriptionUpgradePreview } from "@/app/api/__generated__/models/subscriptionUpgradePreview";

export function maxUpgradePreview(
  overrides: Partial<SubscriptionUpgradePreview> = {},
): SubscriptionUpgradePreview {
  const now = Math.floor(Date.now() / 1000);
  return {
    preview_id: "a".repeat(32),
    amount_due: 27278,
    tax_amount: 2020,
    tax_included: false,
    adjustment_amount: 0,
    currency: "usd",
    proration_date: now,
    period_end: now + 29 * 86400,
    expires_at: now + 900,
    recurring_amount: 32000,
    recurring_tax_included: false,
    billing_cycle: "monthly",
    lines: [
      {
        description: "Remaining Max",
        amount: 29935,
        period_start: now,
        period_end: now + 29 * 86400,
        proration: true,
      },
      {
        description: "Unused Pro",
        amount: -4677,
        period_start: now,
        period_end: now + 29 * 86400,
        proration: true,
      },
    ],
    ...overrides,
  };
}
