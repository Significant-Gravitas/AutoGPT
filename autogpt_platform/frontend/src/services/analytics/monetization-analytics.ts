// The paywall funnel in PostHog: a plan picker is seen, a plan is picked,
// Checkout opens (`checkout_started`, sent by the backend when it creates the
// Stripe session) and is either paid or abandoned. The billing portal is the
// way out to cancel. The pricing arm rides along as PostHog's own
// `$feature/<flag>` properties.

import {
  MonetizationEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import { capturePostHogEvent } from "./posthog-capture";

export type PaywallSurface = "onboarding" | "paywall_gate" | "billing";
export type CheckoutKind = "subscription" | "top_up" | "trial";
export type BillingPortalSurface = "billing" | "billing_payment_method";

type MonetizationEventName = EventName<typeof MonetizationEvent>;

const PAYWALL_VIEWED_KEY_PREFIX = "posthog_paywall_viewed_";

// Once per tab per surface: the onboarding paywall remounts on the return
// from Stripe, and the billing page re-renders on every refetch.
export function trackPaywallViewed(surface: PaywallSurface) {
  const sentKey = `${PAYWALL_VIEWED_KEY_PREFIX}${surface}`;
  try {
    if (sessionStorage.getItem(sentKey)) return;
    sessionStorage.setItem(sentKey, "1");
  } catch {
    // In-app browsers may block sessionStorage — double-counting beats dropping.
  }
  capture(MonetizationEvent.PAYWALL_VIEWED, { surface });
}

type PlanSelectedProperties = {
  subscription_tier: string;
  billing_cycle: "monthly" | "yearly";
  surface: PaywallSurface;
  pricing_variant?: string;
};

export function trackPlanSelected(properties: PlanSelectedProperties) {
  capture(MonetizationEvent.PLAN_SELECTED, properties);
}

export function trackBillingPortalOpened(surface: BillingPortalSurface) {
  capture(MonetizationEvent.BILLING_PORTAL_OPENED, { surface });
}

export function trackCheckoutAbandoned(properties: {
  checkout_kind: CheckoutKind;
  surface: PaywallSurface;
}) {
  capture(MonetizationEvent.CHECKOUT_ABANDONED, properties);
}

const TRIAL_ABANDONED_KEY_PREFIX = "posthog_trial_checkout_abandoned_";

// The trial's cancel URL (`?trial=cancelled`) is not cleaned up after the
// return, so a refresh must not report the same abandonment twice. Starting
// another trial checkout clears the guard (`markTrialCheckoutStarted`), so a
// second real abandonment in the same tab still counts.
export function trackTrialCheckoutAbandoned(surface: PaywallSurface) {
  const sentKey = `${TRIAL_ABANDONED_KEY_PREFIX}${surface}`;
  try {
    if (sessionStorage.getItem(sentKey)) return;
    sessionStorage.setItem(sentKey, "1");
  } catch {
    // In-app browsers may block sessionStorage — double-counting beats dropping.
  }
  trackCheckoutAbandoned({ checkout_kind: "trial", surface });
}

export function markTrialCheckoutStarted(surface: PaywallSurface) {
  try {
    sessionStorage.removeItem(`${TRIAL_ABANDONED_KEY_PREFIX}${surface}`);
  } catch {
    // In-app browsers may block sessionStorage; the guard is then unset anyway.
  }
}

function capture(
  event: MonetizationEventName,
  properties: Record<string, unknown>,
) {
  capturePostHogEvent(event, properties);
}
