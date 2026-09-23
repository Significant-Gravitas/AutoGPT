import { environment } from "@/services/environment";
import {
  COOKIEBOT_CONSENT_COOKIE,
  COOKIEBOT_CONSENT_EVENTS,
  parseCookieConsent,
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
  if (typeof window === "undefined") return NO_CONSENT;
  if (!isConsentManagerConfigured()) return NO_CONSENT;
  return toConsentState(readBrowserConsent());
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

// The loaded script is authoritative once the visitor has answered; before it
// loads (or when an ad blocker stops it) the stored cookie still carries the
// answer from an earlier visit.
function readBrowserConsent(): CookiebotConsent | null {
  const cookiebot = window.Cookiebot;
  if (cookiebot?.hasResponse) return cookiebot.consent;
  return parseCookieConsent(readCookie(COOKIEBOT_CONSENT_COOKIE));
}

function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  const entry = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  return entry ? entry.slice(prefix.length) : null;
}
