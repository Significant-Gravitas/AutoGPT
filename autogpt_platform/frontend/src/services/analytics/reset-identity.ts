import { syncPostHogConsent } from "@/providers/posthog/posthog-consent";
import posthog from "posthog-js";
import { resetAnonymousID } from "./anonymous-id";
import { getPostHogBaseProperties } from "./posthog-base-properties";

export function resetAnalyticsIdentity(): void {
  let anonymousID: string | undefined;
  let didReset = false;
  try {
    if (posthog.get_distinct_id()) {
      posthog.reset(true);
      didReset = true;
      anonymousID = posthog.get_distinct_id();
    }
  } catch {
    anonymousID = undefined;
  }
  if (didReset) restoreBaseProperties();
  resetAnonymousID(anonymousID);
  // reset() also drops PostHog's own opt-in, which opts a consenting visitor
  // out until the next page load unless it is restored here.
  if (didReset) {
    try {
      syncPostHogConsent();
    } catch {
      // Analytics must never block an account change.
    }
  }
}

// reset() wipes the super properties along with the identity.
function restoreBaseProperties() {
  try {
    posthog.register(getPostHogBaseProperties());
  } catch {
    // Analytics is never worth a broken logout.
  }
}
