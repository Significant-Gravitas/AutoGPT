import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionStatusResponseBillingCycle } from "@/app/api/__generated__/models/subscriptionStatusResponseBillingCycle";

export const PLAN_LABEL: Record<string, string> = {
  NO_TIER: "No active subscription",
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
  ENTERPRISE: "Enterprise",
};

export function getSubscriptionPlan(
  subscription: SubscriptionStatusResponse | undefined,
) {
  const tier = subscription?.tier ?? null;
  const billingCycle: SubscriptionStatusResponseBillingCycle =
    subscription?.billing_cycle === "yearly" ? "yearly" : "monthly";
  // ENTERPRISE seats are provisioned by an administrator, so any Stripe
  // schedule attached to one is admin business; surfacing it would offer a
  // "Resume" the backend rejects.
  const isAdminManaged = tier === "ENTERPRISE";
  const pendingTier = isAdminManaged
    ? null
    : (subscription?.pending_tier ?? null);
  const pendingBillingCycle = subscription?.pending_billing_cycle ?? null;
  const isPendingChange = pendingTier !== null && pendingTier !== "NO_TIER";

  return {
    tier,
    billingCycle,
    isAdminManaged,
    pendingTier,
    pendingEffectiveAt: subscription?.pending_tier_effective_at ?? null,
    pendingBillingCycle,
    // Cancellation = pending change to the "no active subscription" state.
    isPendingCancel: pendingTier === "NO_TIER",
    // Paid→paid tier downgrade scheduled in Stripe (e.g. MAX → PRO at period
    // end), from the Downgrade button or the Stripe billing portal.
    isPendingDowngrade: isPendingChange && pendingTier !== tier,
    // Same-tier yearly→monthly schedule: a cycle-only switch, so the copy
    // describes the change rather than a "downgrade to <same tier>".
    isPendingCycleSwitch:
      isPendingChange &&
      pendingTier === tier &&
      pendingBillingCycle !== null &&
      pendingBillingCycle !== billingCycle,
  };
}
