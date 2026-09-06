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
 * An existing PostHog device id is adopted rather than replaced, so returning
 * visitors keep their history.
 */

const ANONYMOUS_ID_KEY = "agpt_anonymous_id";
const FIRST_LANDING_KEY = "agpt_first_landing";

export interface FirstLanding {
  path: string;
  referrer: string | null;
  utm_source: string | null;
  utm_medium: string | null;
  utm_campaign: string | null;
  at: string;
}

let memoryID: string | null = null;

export function getAnonymousID(): string | null {
  if (typeof window === "undefined") return null;
  if (memoryID) return memoryID;

  const stored = readStorage(ANONYMOUS_ID_KEY);
  const id = stored ?? readPostHogDeviceID() ?? newID();
  if (!stored) writeStorage(ANONYMOUS_ID_KEY, id);
  memoryID = id;
  return id;
}

/** Remember the first page this browser landed on, once. */
export function captureFirstLanding(): void {
  if (typeof window === "undefined") return;
  if (readStorage(FIRST_LANDING_KEY)) return;

  const params = new URLSearchParams(window.location.search);
  const landing: FirstLanding = {
    path: window.location.pathname + window.location.search,
    referrer: document.referrer || null,
    utm_source: params.get("utm_source"),
    utm_medium: params.get("utm_medium"),
    utm_campaign: params.get("utm_campaign"),
    at: new Date().toISOString(),
  };
  writeStorage(FIRST_LANDING_KEY, JSON.stringify(landing));
}

export function readFirstLanding(): FirstLanding | null {
  const raw = readStorage(FIRST_LANDING_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as FirstLanding;
  } catch {
    return null;
  }
}

/**
 * Forget this browser's anonymous identity and first landing. Called on
 * logout so the next visitor on a shared machine starts as a new person
 * instead of being bootstrapped onto the previous user's PostHog and
 * LaunchDarkly identity, or reporting their landing page and UTMs.
 */
export function resetAnonymousID(): void {
  memoryID = null;
  try {
    window.localStorage.removeItem(ANONYMOUS_ID_KEY);
    window.localStorage.removeItem(FIRST_LANDING_KEY);
  } catch {
    // Storage blocked: nothing persisted to clear.
  }
}

export function resetAnonymousIDForTests(): void {
  memoryID = null;
}

/** PostHog's own device id, when its persistence exists in this browser. */
export function getPostHogDeviceID(): string | null {
  if (typeof window === "undefined") return null;
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
