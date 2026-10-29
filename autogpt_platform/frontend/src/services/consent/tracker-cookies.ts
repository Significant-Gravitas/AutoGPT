import { deleteCookie, readCookieNames } from "./browser-cookies";
import {
  getConsentAnswer,
  subscribeToConsent,
  type ConsentCategory,
} from "./consent";

interface TrackerCookies {
  category: ConsentCategory;
  matches: (name: string) => boolean;
}

// What the third-party scripts behind each category leave in the browser: the
// Google tag's GA4 and Google Ads cookies, and DataFast's visitor and session
// ids. PostHog's storage is handled in providers/posthog/posthog-consent.ts.
const TRACKER_COOKIES: readonly TrackerCookies[] = [
  {
    category: "analytics",
    matches: (name) =>
      name === "_ga" || name.startsWith("_ga_") || name.startsWith("datafast_"),
  },
  {
    category: "advertising",
    matches: (name) => name.startsWith("_gcl_"),
  },
];

/**
 * Once the visitor has answered without a category, delete the cookies its
 * scripts set while it was allowed. Nothing reads them without consent, but
 * they would otherwise stay until they expire. Nothing goes while the answer
 * is pending, or without a consent manager to answer: the visitor may still
 * allow it, and nothing could have set them.
 */
function forgetDeclinedTrackerCookies() {
  const answer = getConsentAnswer();
  if (!answer) return;
  const declined = TRACKER_COOKIES.filter(({ category }) => !answer[category]);
  if (declined.length === 0) return;
  readCookieNames()
    .filter((name) => declined.some(({ matches }) => matches(name)))
    .forEach(deleteCookie);
}

/** Runs forgetDeclinedTrackerCookies now and on every answer; returns the unsubscribe. */
export function followConsentForTrackerCookies(): () => void {
  forgetDeclinedTrackerCookies();
  return subscribeToConsent(forgetDeclinedTrackerCookies);
}
