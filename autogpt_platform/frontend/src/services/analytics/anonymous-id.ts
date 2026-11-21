/**
 * One first-party anonymous identity for the whole analytics stack.
 *
 * Analytics tools each mint their own visitor id, so nothing could tie a
 * pre-signup visitor to the user they became. This module owns that id instead:
 *
 * - PostHog is bootstrapped with it as the anonymous distinct id, so identify()
 *   merges it into the user as usual.
 * - The backend stores it on the user at signup (`UserAttribution`), which is
 *   the join key for everything else.
 *
 * It only outlives the page with analytics consent. Without it the id is
 * minted fresh for each page load and kept in memory, nothing an earlier
 * visit stored is read, and the first landing stays in memory too. Granting
 * persists both without a reload; declining deletes them. While the answer is
 * still pending, what is stored is left alone.
 *
 * With consent the id is, in order: the one this browser stored on an
 * earlier visit, so returning visitors keep theirs; else PostHog's device id,
 * which the marketing site (agpt.co) leaves in the shared `.agpt.co` cookie,
 * so the website visits join this visitor at signup; else this page's id.
 * The website's id is only taken while nobody is signed in, or for an account
 * created in this browser: an older account must not take over the website
 * history of whoever used this browser on agpt.co. Until the session is known
 * that choice waits: the id stays this page's, unsaved, and settles (telling
 * subscribeToAnonymousIDSettled) once the auth store reports who is signed in.
 *
 * When PostHog's saved state shows an account identified in this browser, an
 * earlier page merged all of those ids into it. Another visitor gets a new id
 * instead, and this page's landing as their first, or PostHog refuses the
 * merge at their signup and their history stays with the older account. If the
 * session is not known yet, the stored id settles as usual and the auth store
 * rotates it (isAnonymousIDSpentFor); events captured before that still go to
 * the older account.
 */

import { readAccountCreatedFlag } from "@/services/analytics/account-created-cookie";
import {
  hasConsentFor,
  hasDeclinedConsentFor,
  subscribeToConsent,
} from "@/services/consent/consent";

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

type Session = "unknown" | "signed-in" | "signed-out";

let memoryID: string | null = null;
// Whether memoryID is settled for this page: resolved with consent, or set by
// a reset. Before that it only stands in.
let settled = false;
let session: Session = "unknown";
// Seen while signed in on this page. The flag itself goes as soon as the ads
// signup conversion is out, which can be before the consent answer.
let accountCreatedHere = false;
// PostHog's device ids as this browser held them before this page wrote any
// (its own localStorage, and the shared cookie the website may have written),
// read once with consent: PostHog rewrites both as soon as its storage turns
// persistent.
let ownDeviceID: string | null | undefined;
let websiteID: string | null | undefined;
// The account PostHog's saved state shows identified, read once with them.
let savedAccount: string | null | undefined;
// The account the settled id was merged into on an earlier page.
let mergedIntoAccount: string | null = null;
let sessionUserID: string | null = null;
let pageLanding: FirstLanding | null = null;
const settledListeners = new Set<() => void>();

/** Who is signed in, once the session is known; kept by the auth store. */
export function setAnalyticsIdentitySignedIn(userID: string | null): void {
  session = userID ? "signed-in" : "signed-out";
  sessionUserID = userID;
  if (userID && readAccountCreatedFlag()) accountCreatedHere = true;
  if (settled || typeof window === "undefined") return;
  if (!hasConsentFor("analytics")) return;
  syncStoredIdentity();
  if (settled) settledListeners.forEach((listener) => listener());
}

/**
 * Whether the consented id is decided. False while it waits for the session
 * (see above); PostHog stays opted out and in memory until then.
 */
export function isAnonymousIDSettled(): boolean {
  getAnonymousID();
  return settled;
}

/**
 * Whether an earlier page merged the settled id into an account other than
 * `userID` (null: nobody signed in), so it must be rotated.
 */
export function isAnonymousIDSpentFor(userID: string | null): boolean {
  return settled && mergedIntoAccount !== null && mergedIntoAccount !== userID;
}

/** Calls `listener` when a consented id settles after the session is known. */
export function subscribeToAnonymousIDSettled(
  listener: () => void,
): () => void {
  settledListeners.add(listener);
  return () => settledListeners.delete(listener);
}

export function getAnonymousID(): string | null {
  if (typeof window === "undefined") return null;
  if (hasConsentFor("analytics") && !settled) {
    const id = resolveConsentedID();
    if (id) {
      memoryID = id;
      settled = true;
    }
  }
  memoryID ??= newID();
  return memoryID;
}

// Null while the choice waits for the session.
function resolveConsentedID(): string | null {
  if (savedAccount === undefined) savedAccount = readSavedPostHogAccount();
  if (savedAccount && session !== "unknown" && savedAccount !== sessionUserID) {
    replaceFirstLanding();
    return saveID(memoryID ?? newID());
  }
  const stored = readStorage(ANONYMOUS_ID_KEY);
  if (ownDeviceID === undefined) ownDeviceID = readOwnPostHogDeviceID();
  if (stored || ownDeviceID) mergedIntoAccount = savedAccount;
  if (stored) return stored;
  const id = ownDeviceID ?? chooseWebsiteID();
  if (id === undefined) return null;
  return saveID(id ?? memoryID ?? newID());
}

function saveID(id: string): string {
  writeStorage(ANONYMOUS_ID_KEY, id);
  return id;
}

// The website's id, null when there is none or it may not be taken, and
// undefined while that depends on a session not known yet.
function chooseWebsiteID(): string | null | undefined {
  if (websiteID === undefined) websiteID = readSharedPostHogDeviceID();
  if (!websiteID) return null;
  if (session === "unknown") return undefined;
  if (session === "signed-out") return websiteID;
  if (accountCreatedHere || readAccountCreatedFlag()) return websiteID;
  return null;
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
 * Keep the stored identity in step with analytics consent: persist the id and
 * landing on a grant, delete them on an answer without it (also clearing
 * whatever a visit with consent left behind). Returns the unsubscribe.
 */
export function followAnalyticsConsentForIdentity(): () => void {
  if (typeof window === "undefined") return () => {};
  syncStoredIdentity();
  return subscribeToConsent(syncStoredIdentity);
}

function syncStoredIdentity(): void {
  if (!hasConsentFor("analytics")) {
    if (hasDeclinedConsentFor("analytics")) forgetStoredIdentity();
    return;
  }
  const id = getAnonymousID();
  if (settled && id && readStorage(ANONYMOUS_ID_KEY) !== id) {
    writeStorage(ANONYMOUS_ID_KEY, id);
  }
  persistFirstLanding();
}

function persistFirstLanding(): void {
  if (!pageLanding || readStorage(FIRST_LANDING_KEY)) return;
  writeStorage(FIRST_LANDING_KEY, JSON.stringify(pageLanding));
}

// The stored landing is an earlier visitor's, and persistFirstLanding never
// overwrites one, so this page's landing takes its place.
function replaceFirstLanding(): void {
  removeStorage(FIRST_LANDING_KEY);
  persistFirstLanding();
}

function forgetStoredIdentity(): void {
  removeStorage(ANONYMOUS_ID_KEY);
  removeStorage(FIRST_LANDING_KEY);
}

interface ResetArgs {
  nextID?: string;
  // The new id is for the visitor who loaded this page (a spent id rotated
  // once the session is known), so this page's landing becomes its first.
  keepLanding?: boolean;
}

/**
 * Rotate the browser identity and clear its first landing. With consent the
 * new identity is persisted immediately so old PostHog storage cannot restore
 * the last visitor.
 */
export function resetAnonymousID({
  nextID,
  keepLanding = false,
}: ResetArgs = {}): void {
  if (typeof window === "undefined") {
    memoryID = null;
    return;
  }
  memoryID = nextID || newID();
  settled = true;
  ownDeviceID = null;
  websiteID = null;
  mergedIntoAccount = null;
  if (!keepLanding) pageLanding = null;
  forgetStoredIdentity();
  if (!hasConsentFor("analytics")) return;
  writeStorage(ANONYMOUS_ID_KEY, memoryID);
  persistFirstLanding();
}

export function resetAnonymousIDForTests(): void {
  memoryID = null;
  settled = false;
  session = "unknown";
  sessionUserID = null;
  accountCreatedHere = false;
  ownDeviceID = undefined;
  websiteID = undefined;
  savedAccount = undefined;
  mergedIntoAccount = null;
  settledListeners.clear();
  pageLanding = null;
}

/** PostHog's own device id, when it may be read and exists in this browser. */
export function getPostHogDeviceID(): string | null {
  if (typeof window === "undefined") return null;
  if (!hasConsentFor("analytics")) return null;
  return readPostHogDeviceID();
}

function readPostHogDeviceID(): string | null {
  return readOwnPostHogDeviceID() ?? readSharedPostHogDeviceID();
}

// localStorage is this app's own (per origin).
function readOwnPostHogDeviceID(): string | null {
  const name = getPostHogPersistenceName();
  return name ? parseDeviceID(readStorage(name)) : null;
}

// The cookie sits on `.agpt.co`, so the marketing site may have written it.
function readSharedPostHogDeviceID(): string | null {
  const name = getPostHogPersistenceName();
  return name ? parseDeviceID(readCookie(name)) : null;
}

// localStorage is this app's own. The cookie holds the same state and may
// outlive it, so it counts only when localStorage has none: a stale cookie
// must not mark a fresh id as merged.
function readSavedPostHogAccount(): string | null {
  const name = getPostHogPersistenceName();
  if (!name) return null;
  return parseIdentifiedAccount(readStorage(name) ?? readCookie(name));
}

function getPostHogPersistenceName(): string | null {
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  return key ? `ph_${key}_posthog` : null;
}

function parseDeviceID(raw: string | null): string | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as { $device_id?: unknown };
    return typeof parsed.$device_id === "string" ? parsed.$device_id : null;
  } catch {
    return null;
  }
}

function parseIdentifiedAccount(raw: string | null): string | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as {
      distinct_id?: unknown;
      $user_state?: unknown;
    };
    if (parsed.$user_state !== "identified") return null;
    return typeof parsed.distinct_id === "string" ? parsed.distinct_id : null;
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

function removeStorage(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Storage blocked: nothing persisted to clear.
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
