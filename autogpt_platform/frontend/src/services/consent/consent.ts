import { environment } from "@/services/environment";
import {
  COOKIEBOT_CONSENT_EVENTS,
  parseCookieConsentHeader,
  type CookiebotConsent,
} from "./cookiebot";

export type ConsentCategory = "analytics" | "monitoring" | "advertising";
export type ConsentState = Readonly<Record<ConsentCategory, boolean>>;

export const NO_CONSENT: ConsentState = Object.freeze({
  analytics: false,
  monitoring: false,
  advertising: false,
});

const COOKIEBOT_CATEGORY: Record<
  ConsentCategory,
  Exclude<keyof CookiebotConsent, "necessary">
> = {
  analytics: "statistics",
  monitoring: "statistics",
  advertising: "marketing",
};

/**
 * Without a Cookiebot domain group (self-hosted and local installs) there is
 * no banner, and every optional category stays denied.
 */
export function isConsentManagerConfigured(): boolean {
  return Boolean(environment.getCookiebotCBID());
}

export function toConsentState(
  cookiebot: CookiebotConsent | null,
): ConsentState {
  if (!cookiebot) return NO_CONSENT;
  return {
    analytics: cookiebot[COOKIEBOT_CATEGORY.analytics],
    monitoring: cookiebot[COOKIEBOT_CATEGORY.monitoring],
    advertising: cookiebot[COOKIEBOT_CATEGORY.advertising],
  };
}

export function getConsent(): ConsentState {
  return getConsentAnswer() ?? NO_CONSENT;
}

/** The visitor's answer, or null while there is none to act on. */
export function getConsentAnswer(): ConsentState | null {
  if (typeof window === "undefined") return null;
  if (!isConsentManagerConfigured()) return null;
  const answer = readBrowserConsent();
  return answer ? toConsentState(answer) : null;
}

export function hasConsentFor(category: ConsentCategory): boolean {
  return getConsent()[category];
}

export function isSameConsent(a: ConsentState, b: ConsentState): boolean {
  return (
    a.analytics === b.analytics &&
    a.monitoring === b.monitoring &&
    a.advertising === b.advertising
  );
}

export function subscribeToConsent(listener: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  COOKIEBOT_CONSENT_EVENTS.forEach((event) =>
    window.addEventListener(event, listener),
  );
  return () =>
    COOKIEBOT_CONSENT_EVENTS.forEach((event) =>
      window.removeEventListener(event, listener),
    );
}

export function openConsentSettings(): void {
  if (typeof window === "undefined") return;
  window.Cookiebot?.renew();
}

// Once loaded, the script is authoritative: it also withdraws a stored answer
// it no longer accepts (a new banner version, an expired answer, a region
// change) and asks again, while the old cookie stays behind until the visitor
// answers. Only before it loads, or when a blocker stops it, does the stored
// cookie speak for the visitor.
function readBrowserConsent(): CookiebotConsent | null {
  const cookiebot = window.Cookiebot;
  if (cookiebot) return cookiebot.hasResponse ? cookiebot.consent : null;
  return parseCookieConsentHeader(document.cookie);
}
