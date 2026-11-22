import { SubscriptionTierRequestBillingCycle } from "@/app/api/__generated__/models/subscriptionTierRequestBillingCycle";
import { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";

export interface AbandonedPlan {
  tier: SubscriptionTierRequestTier;
  cycle: SubscriptionTierRequestBillingCycle;
}

function isOneOf<T extends string>(
  values: Record<string, T>,
  value: string | null,
): value is T {
  return value !== null && (Object.values(values) as string[]).includes(value);
}

// The plan the person left checkout for, from the cancel URL, so "Try another
// card" can open the same checkout again.
export function readAbandonedPlan(
  searchParams: URLSearchParams,
): AbandonedPlan | null {
  const tier = searchParams.get("plan");
  const cycle = searchParams.get("cycle");
  if (!isOneOf(SubscriptionTierRequestTier, tier) || tier === "NO_TIER")
    return null;
  return {
    tier,
    cycle: isOneOf(SubscriptionTierRequestBillingCycle, cycle)
      ? cycle
      : "monthly",
  };
}
