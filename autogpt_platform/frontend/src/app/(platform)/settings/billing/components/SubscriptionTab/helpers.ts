import type { getSubscriptionStatusResponse } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";
import type { SubscriptionTierRequestBillingCycle } from "@/app/api/__generated__/models/subscriptionTierRequestBillingCycle";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import type { FetchStatus } from "@tanstack/react-query";

interface CheckoutReturnArgs {
  tier: SubscriptionTierRequestTier;
  cycle: SubscriptionTierRequestBillingCycle;
}

// Stripe fills {CHECKOUT_SESSION_ID}; plan and cycle let the return page
// report the subscription to Google Ads, or open the same checkout again.
export function getCheckoutReturnURLs({ tier, cycle }: CheckoutReturnArgs) {
  const page = `${window.location.origin}${window.location.pathname}`;
  return {
    successURL: `${page}?subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=${tier}&cycle=${cycle}`,
    cancelURL: `${page}?subscription=cancelled&plan=${tier}&cycle=${cycle}`,
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

interface Read {
  isPending: boolean;
  isFetched: boolean;
  fetchStatus: FetchStatus;
}

// Whether a read has been asked for and has yet to answer at all: it is
// fetching, or paused while offline. A disabled read hasn't been asked for,
// so it doesn't count. React Query puts a read that holds no data back to
// pending on each retry, so isPending alone also hides what a failed read
// decides. A card that reads it retries it as it mounts, and would then mount
// and unmount in a loop, one read per round trip.
export function isAwaitingFirstAnswer(read: Read) {
  return read.isPending && read.fetchStatus !== "idle" && !read.isFetched;
}
