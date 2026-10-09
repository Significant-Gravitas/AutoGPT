/**
 * One first-party anonymous identity for the whole analytics stack.
 *
 * PostHog and LaunchDarkly each mint their own visitor id, which is why
 * LaunchDarkly could only bucket by user id (every logged-out visitor shared
 * the literal key "anonymous") and nothing could tie a pre-signup flag arm
 * to a post-signup PostHog person. This module owns that id instead:
 *
 * - PostHog is bootstrapped with it as the anonymous distinct id, so identify()
 *   merges it into the user as usual.
 * - LaunchDarkly gets it as the anonymous user key before login and as a
 *   `device` context after login, so rules bucketed by device stay stable
 *   across signup.
 * - The backend stores it on the user at signup (`UserAttribution`), which is
 *   the join key for everything else.
 *
 * It only outlives the page with analytics consent. Without it the id is
 * minted fresh for each page load and kept in memory, nothing an earlier
 * visit stored is read, and the first landing stays in memory too. Granting
 * persists both without a reload; withdrawing deletes them. With consent an
 * existing PostHog device id is adopted rather than replaced, so returning
 * visitors keep their history.
 */

import { hasConsentFor, subscribeToConsent } from "@/services/consent/consent";

const ANONYMOUS_ID_KEY = "agpt_anonymous_id";
const FIRST_LANDING_KEY = "agpt_first_landing";

// Routes that carry a one-time secret in the path itself. Landing on one
// records the route and drops the rest, so an invite token never lands at rest.
const REDACTED_PREFIXES = ["/link", "/share", "/auth", "/reset-password"];
const SENSITIVE_PARAMS = [
  "token",
  "code",
  "access_token",
  "refresh_token",
  "state",
  "email",
];

export interface FirstLanding {
  path: string;
  referrer: string | null;
  utm_source: string | null;
  utm_medium: string | null;
  utm_campaign: string | null;
  at: string;
}

let memoryID: string | null = null;
let pageLanding: FirstLanding | null = null;

export function getAnonymousID(): string | null {
  if (typeof window === "undefined") return null;
  if (memoryID) return memoryID;

  if (!hasConsentFor("analytics")) {
    memoryID = newID();
    return memoryID;
  }
  const stored = readStorage(ANONYMOUS_ID_KEY);
  const id = stored ?? readPostHogDeviceID() ?? newID();
  if (!stored) writeStorage(ANONYMOUS_ID_KEY, id);
  memoryID = id;
  return id;
}

/**
 * Remember the page this visit landed on. It is stored as the browser's first
 * landing only with analytics consent (once, on the first visit that has it);
 * otherwise it stays in memory for this page load.
 */
export function captureFirstLanding(): void {
  if (typeof window === "undefined") return;
  pageLanding ??= describeLanding();
  if (hasConsentFor("analytics")) persistFirstLanding();
}

export function readFirstLanding(): FirstLanding | null {
  if (typeof window === "undefined") return null;
  if (!hasConsentFor("analytics")) return pageLanding;
  const raw = readStorage(FIRST_LANDING_KEY);
  if (!raw) return pageLanding;
  try {
    return JSON.parse(raw) as FirstLanding;
  } catch {
    return pageLanding;
  }
}

/**
 * Keep the stored identity in step with analytics consent: persist this page's
 * id and landing on a grant, delete them otherwise (also clearing whatever a
 * visit with consent left behind). Returns the unsubscribe.
 */
export function followAnalyticsConsentForIdentity(): () => void {
  if (typeof window === "undefined") return () => {};
  syncStoredIdentity();
  return subscribeToConsent(syncStoredIdentity);
}

function syncStoredIdentity(): void {
  if (!hasConsentFor("analytics")) {
    forgetStoredIdentity();
    return;
  }
  const id = getAnonymousID();
  if (id && readStorage(ANONYMOUS_ID_KEY) !== id) {
    writeStorage(ANONYMOUS_ID_KEY, id);
  }
  persistFirstLanding();
}

function persistFirstLanding(): void {
  if (!pageLanding || readStorage(FIRST_LANDING_KEY)) return;
  writeStorage(FIRST_LANDING_KEY, JSON.stringify(pageLanding));
}

function forgetStoredIdentity(): void {
  try {
    window.localStorage.removeItem(ANONYMOUS_ID_KEY);
    window.localStorage.removeItem(FIRST_LANDING_KEY);
  } catch {
    // Storage blocked: nothing persisted to clear.
  }
}

/**
 * Rotate the browser identity and clear its first landing. With consent the
 * new identity is persisted immediately so old PostHog storage cannot restore
 * the last visitor.
 */
export function resetAnonymousID(nextID?: string): void {
  if (typeof window === "undefined") {
    memoryID = null;
    return;
  }
  memoryID = nextID || newID();
  pageLanding = null;
  forgetStoredIdentity();
  if (hasConsentFor("analytics")) writeStorage(ANONYMOUS_ID_KEY, memoryID);
}

export function resetAnonymousIDForTests(): void {
  memoryID = null;
  pageLanding = null;
}

/** PostHog's own device id, when it may be read and exists in this browser. */
export function getPostHogDeviceID(): string | null {
  if (typeof window === "undefined") return null;
  if (!hasConsentFor("analytics")) return null;
  return readPostHogDeviceID();
}

function readPostHogDeviceID(): string | null {
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  if (!key) return null;
  const persistenceKey = `ph_${key}_posthog`;
  const raw = readStorage(persistenceKey) ?? readCookie(persistenceKey);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as { $device_id?: unknown };
    return typeof parsed.$device_id === "string" ? parsed.$device_id : null;
  } catch {
    return null;
  }
}

function describeLanding(): FirstLanding {
  const params = new URLSearchParams(window.location.search);
  return {
    path: redactPath(window.location.pathname, window.location.search),
    referrer: redactReferrer(document.referrer),
    utm_source: params.get("utm_source"),
    utm_medium: params.get("utm_medium"),
    utm_campaign: params.get("utm_campaign"),
    at: new Date().toISOString(),
  };
}

function redactPath(pathname: string, search: string): string {
  const prefix = REDACTED_PREFIXES.find(
    (candidate) =>
      pathname === candidate || pathname.startsWith(`${candidate}/`),
  );
  if (prefix) return prefix;

  const params = new URLSearchParams(search);
  SENSITIVE_PARAMS.forEach((name) => params.delete(name));
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
}

/** Same-origin referrers reach the same secret-bearing routes, so redact both. */
function redactReferrer(referrer: string): string | null {
  if (!referrer) return null;
  try {
    const url = new URL(referrer);
    if (url.origin !== window.location.origin) return url.origin + url.pathname;
    return url.origin + redactPath(url.pathname, url.search);
  } catch {
    return null;
  }
}

function newID(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private mode or blocked storage: the in-memory id still holds for
    // this page load.
  }
}

function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  const entry = document.cookie
    .split("; ")
    .find((part) => part.startsWith(prefix));
  if (!entry) return null;
  try {
    return decodeURIComponent(entry.slice(prefix.length));
  } catch {
    return null;
  }
}
