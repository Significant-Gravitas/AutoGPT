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
