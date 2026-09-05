import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";

export function formatUpgradePrice(cents: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: cents % 100 === 0 ? 0 : 2,
    maximumFractionDigits: 2,
  }).format(cents / 100);
}

export function getMaxUpgradePricing(
  subscription: SubscriptionStatusResponse | undefined,
) {
  if (!subscription) return null;
  const cycle: "monthly" | "yearly" =
    subscription.billing_cycle === "yearly" ? "yearly" : "monthly";
  const prices =
    cycle === "yearly"
      ? subscription.tier_costs_yearly
      : subscription.tier_costs;
  return {
    cycle,
    currentCents: positiveCents(prices?.PRO),
    maxCents: positiveCents(prices?.MAX),
  };
}

export function getMaxUpgradeUnavailableReason(
  subscription: SubscriptionStatusResponse | undefined,
): string | null {
  if (!subscription) return "Your plan details are not available yet.";
  if (subscription.tier !== "PRO") {
    return "Your plan has changed. Review your available upgrades in billing.";
  }
  if (
    subscription.pending_tier != null ||
    subscription.pending_billing_cycle != null
  ) {
    return "Manage your scheduled plan change in billing before upgrading.";
  }
  if (!subscription.has_active_stripe_subscription) {
    return "Continue in billing to set up your Max subscription.";
  }
  const pricing = getMaxUpgradePricing(subscription);
  if (!pricing?.maxCents) {
    return `Max ${pricing?.cycle ?? "monthly"} pricing is unavailable. Please check billing or contact support.`;
  }
  return null;
}

function positiveCents(value: number | undefined): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : null;
}
