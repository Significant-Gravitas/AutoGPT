import {
  forgetPostHogStorage,
  syncPostHogConsent,
} from "@/providers/posthog/posthog-consent";
import { hasConsentFor } from "@/services/consent/consent";
import posthog from "posthog-js";
import { resetAnonymousID } from "./anonymous-id";
import { getPostHogBaseProperties } from "./posthog-base-properties";

// keepLanding: the new identity is for the visitor who loaded this page (a
// spent id rotated once the session is known), not the next account after a
// sign-out or switch, so this page's landing stays as its first.
export function resetAnalyticsIdentity({
  keepLanding = false,
}: { keepLanding?: boolean } = {}): void {
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
  resetAnonymousID({ nextID: anonymousID, keepLanding });
  // With consent reset() rewrote PostHog's storage with the new identity;
  // without it (declined or still pending) the old one would stay stored.
  if (!hasConsentFor("analytics")) {
    try {
      forgetPostHogStorage();
    } catch {
      // Analytics must never block an account change.
    }
  }
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
