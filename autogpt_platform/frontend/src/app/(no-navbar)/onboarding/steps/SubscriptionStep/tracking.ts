import {
  trackCheckoutAbandoned,
  trackPaywallViewed,
} from "@/services/analytics/monetization-analytics";

/**
 * Reports that the user was actually shown the paywall.
 *
 * Onboarding's steps all share the `/onboarding` URL, so without this event
 * the funnel can't tell "never reached the subscription screen" from "saw the
 * price and declined" — which is the only question that matters between
 * signup and payment.
 *
 * Deliberately untagged by pricing arm. `posthog.init` runs in an effect with
 * no bootstrap, so the experiment variant is still unresolved when this step
 * mounts and would record "control" for everyone; PostHog records arm
 * assignment itself, so segmentation belongs there rather than here.
 *
 * PostHog keeps its own once-per-tab guard, marked only once the event is sent
 * or dropped for lack of consent, not while it waits for the answer.
 */
export function trackPaywallView() {
  trackPaywallViewed("onboarding");
}

const POSTHOG_CHECKOUT_CANCELLED_KEY =
  "posthog_checkout_abandoned_onboarding_subscription";

/**
 * Re-arms the PostHog abandonment guard when a new Checkout starts, so a second
 * real abandonment in the same tab still counts; a refresh after the return
 * stays guarded.
 */
export function markPaywallCheckoutStarted() {
  try {
    sessionStorage.removeItem(POSTHOG_CHECKOUT_CANCELLED_KEY);
  } catch {
    // In-app browsers may block sessionStorage; the guard is then unset anyway.
  }
}

/**
 * Reports that the user reached Stripe Checkout and came back without paying.
 *
 * Distinguishes "looked at the price and never clicked" from "picked a plan,
 * saw the card form, and backed out" — two very different problems that
 * `paywall_viewed` alone cannot separate, since Stripe is off-domain.
 *
 * Guarded per tab because the `subscription=cancelled` param persists after the
 * return: the wizard's URL-sync effect only rewrites the query when the step
 * changes, and on a cancel return the step is already correct, so a refresh
 * would otherwise report the same abandonment again.
 */
export function trackPaywallCheckoutCancelled() {
  trackCheckoutAbandoned(
    { checkout_kind: "subscription", surface: "onboarding" },
    POSTHOG_CHECKOUT_CANCELLED_KEY,
  );
}
