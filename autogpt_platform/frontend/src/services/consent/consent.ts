import { environment } from "@/services/environment";
import {
  COOKIEBOT_CONSENT_EVENTS,
  COOKIEBOT_DIALOG_EVENTS,
  COOKIEBOT_SCRIPT_ID,
  getCookiebotAPI,
  parseCookieConsentHeader,
  type CookiebotConsent,
} from "./cookiebot";

// "monitoring" gates Sentry session replay only. Error monitoring is part of
// running the service (legitimate interest), so it needs no consent.
// "preferences" gates nothing in the app; it is only passed on to the Google
// tag as Consent Mode's functionality and personalization signals.
export type ConsentCategory =
  | "analytics"
  | "monitoring"
  | "advertising"
  | "preferences";
export type ConsentState = Readonly<Record<ConsentCategory, boolean>>;

export const NO_CONSENT: ConsentState = Object.freeze({
  analytics: false,
  monitoring: false,
  advertising: false,
  preferences: false,
});

const COOKIEBOT_CATEGORY: Record<
  ConsentCategory,
  Exclude<keyof CookiebotConsent, "necessary">
> = {
  analytics: "statistics",
  monitoring: "statistics",
  advertising: "marketing",
  preferences: "preferences",
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
    preferences: cookiebot[COOKIEBOT_CATEGORY.preferences],
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

/**
 * Whether the visitor has answered without allowing `category`. False while
 * the answer is still pending (uc.js not loaded yet and no stored answer, or
 * the banner asking): nothing for the category is read or written then, but
 * what an earlier visit stored stays until the visitor actually declines.
 * Cookiebot answers per domain, so the visitor may have allowed it on agpt.co,
 * whose PostHog cookie on `.agpt.co` this app shares. Without a consent
 * manager nothing can ever be allowed, so that counts as declined.
 */
export function hasDeclinedConsentFor(category: ConsentCategory): boolean {
  if (!isConsentManagerConfigured()) return true;
  const answer = getConsentAnswer();
  return answer !== null && !answer[category];
}

export function isSameConsent(a: ConsentState, b: ConsentState): boolean {
  return (
    a.analytics === b.analytics &&
    a.monitoring === b.monitoring &&
    a.advertising === b.advertising &&
    a.preferences === b.preferences
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

/** Calls `listener` when the banner is about to ask the visitor. */
export function subscribeToConsentPrompt(listener: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  COOKIEBOT_DIALOG_EVENTS.forEach((event) =>
    window.addEventListener(event, listener),
  );
  return () =>
    COOKIEBOT_DIALOG_EVENTS.forEach((event) =>
      window.removeEventListener(event, listener),
    );
}

/**
 * Whether the consent manager has loaded without an answer from the visitor,
 * so the banner is asking (or about to). An answer given after this is a
 * fresh decision, not one that covers what happened before it.
 */
export function isAwaitingConsentAnswer(): boolean {
  return getCookiebotAPI() !== null && getConsentAnswer() === null;
}

export function openConsentSettings(): void {
  if (typeof window === "undefined") return;
  getCookiebotAPI()?.renew();
}

// uc.js is injected after hydration, so it can arrive after the page has
// finished loading. Only once it is this late is it taken as blocked.
export const COOKIEBOT_LOAD_GRACE_MS = 10_000;

/**
 * Whether the visitor's answer is still to come: a consent manager is
 * configured and has no answer yet, and it is asking, or uc.js may still
 * arrive. A uc.js that hasn't arrived COOKIEBOT_LOAD_GRACE_MS after the page
 * loaded counts as blocked: it never answers.
 */
export function isConsentAnswerPending(): boolean {
  if (typeof window === "undefined") return true;
  if (!isConsentManagerConfigured() || getConsentAnswer() !== null) {
    return false;
  }
  if (getCookiebotAPI()) return true;
  return msSincePageLoaded() < COOKIEBOT_LOAD_GRACE_MS;
}

/** Like subscribeToConsentManagerStatus, plus the end of the grace. */
export function subscribeToConsentAnswerPending(
  listener: () => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  let timer: ReturnType<typeof setTimeout> | undefined;
  function scheduleGraceEnd() {
    clearTimeout(timer);
    if (document.readyState !== "complete") return;
    const remaining = COOKIEBOT_LOAD_GRACE_MS - msSincePageLoaded();
    if (remaining >= 0) timer = setTimeout(listener, remaining + 1);
  }
  function onChange() {
    scheduleGraceEnd();
    listener();
  }
  scheduleGraceEnd();
  const unsubscribe = subscribeToConsentManagerStatus(onChange);
  return () => {
    clearTimeout(timer);
    unsubscribe();
  };
}

// 0 until the page has loaded; unbounded where the browser has no
// navigation timing to tell when that was.
function msSincePageLoaded(): number {
  if (document.readyState !== "complete") return 0;
  const [navigation] = performance.getEntriesByType("navigation") as
    | PerformanceNavigationTiming[]
    | [];
  if (!navigation) return Number.POSITIVE_INFINITY;
  const loadedAt = navigation.loadEventEnd || navigation.loadEventStart;
  return loadedAt ? performance.now() - loadedAt : 0;
}

export type ConsentManagerStatus = "loading" | "ready" | "unavailable";

/**
 * Whether the consent dialog can be opened. The script loads after the page
 * becomes interactive; once the page has finished loading without it, a
 * content blocker (or a network failure) stopped it.
 */
export function getConsentManagerStatus(): ConsentManagerStatus {
  if (typeof window === "undefined") return "loading";
  if (getCookiebotAPI()) return "ready";
  return document.readyState === "complete" ? "unavailable" : "loading";
}

// The script is injected after hydration, so the page can finish loading
// before it arrives (and before its tag exists to listen to), and a visitor
// with no answer yet gets no consent event until they reply: the dialog
// events are what say Cookiebot arrived for them.
export function subscribeToConsentManagerStatus(
  listener: () => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const script = document.getElementById(COOKIEBOT_SCRIPT_ID);
  const windowEvents = ["load", ...COOKIEBOT_DIALOG_EVENTS];
  windowEvents.forEach((event) => window.addEventListener(event, listener));
  script?.addEventListener("load", listener);
  script?.addEventListener("error", listener);
  const unsubscribe = subscribeToConsent(listener);
  return () => {
    windowEvents.forEach((event) =>
      window.removeEventListener(event, listener),
    );
    script?.removeEventListener("load", listener);
    script?.removeEventListener("error", listener);
    unsubscribe();
  };
}

// Once loaded, the script is authoritative: it also withdraws a stored answer
// it no longer accepts (a new banner version, an expired answer, a region
// change) and asks again, while the old cookie stays behind until the visitor
// answers. Only before it loads, or when a blocker stops it, does the stored
// cookie speak for the visitor.
function readBrowserConsent(): CookiebotConsent | null {
  const cookiebot = getCookiebotAPI();
  if (cookiebot) return cookiebot.hasResponse ? cookiebot.consent : null;
  return parseCookieConsentHeader(document.cookie);
}
