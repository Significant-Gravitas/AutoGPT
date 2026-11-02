import type { getSubscriptionStatusResponse } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";
import type { SubscriptionTierRequestBillingCycle } from "@/app/api/__generated__/models/subscriptionTierRequestBillingCycle";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";

interface CheckoutReturnArgs {
  tier: SubscriptionTierRequestTier;
  cycle: SubscriptionTierRequestBillingCycle;
}

// Stripe fills {CHECKOUT_SESSION_ID}; plan and cycle let the return page
// report the subscription to Google Ads.
export function getCheckoutReturnURLs({ tier, cycle }: CheckoutReturnArgs) {
  const page = `${window.location.origin}${window.location.pathname}`;
  return {
    successURL: `${page}?subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=${tier}&cycle=${cycle}`,
    cancelURL: `${page}?subscription=cancelled`,
  };
}

const PAID_TIERS: readonly SubscriptionStatusResponseTier[] = [
  "BASIC",
  "PRO",
  "MAX",
  "BUSINESS",
  "ENTERPRISE",
];

export function isPaidTier(tier: SubscriptionStatusResponseTier | undefined) {
  return tier !== undefined && PAID_TIERS.includes(tier);
}

export function readTier(response: getSubscriptionStatusResponse | undefined) {
  return response?.status === 200 ? response.data.tier : undefined;
}
