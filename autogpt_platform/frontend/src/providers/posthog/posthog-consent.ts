import { hasConsentFor, subscribeToConsent } from "@/services/consent/consent";
import posthog, { type PostHogConfig } from "posthog-js";

const POSTHOG_CONSENT_KEY = "agpt_posthog_consent";
const PERSISTENT_STORAGE = "localStorage+cookie";

/**
 * PostHog starts opted out: capture() drops events instead of queueing them
 * and session replay stays off. Without analytics consent it also keeps its
 * state in memory, and forgetPostHogStorageWithoutConsent has deleted
 * anything an earlier visit stored. Feature flags still evaluate, so
 * experiments keep working for visitors who decline.
 */
export function getConsentGatedConfig() {
  return {
    persistence: hasConsentFor("analytics") ? PERSISTENT_STORAGE : "memory",
    opt_out_capturing_by_default: true,
    opt_out_persistence_by_default: true,
    opt_out_capturing_persistence_type: "localStorage",
    consent_persistence_name: POSTHOG_CONSENT_KEY,
    disable_session_recording: true,
  } satisfies Partial<PostHogConfig>;
}

/**
 * Cookiebot is the source of truth. Run before init: without analytics
 * consent, delete what PostHog stored on an earlier visit (its identity,
 * which may belong to an identified user, and its own opt-in), since memory
 * persistence only stops it being read.
 */
export function forgetPostHogStorageWithoutConsent(token: string) {
  if (hasConsentFor("analytics")) return;
  clearPostHogStorage(token);
}

/** Call once posthog.init has run; returns the unsubscribe. */
export function followAnalyticsConsent() {
  syncPostHogConsent();
  return subscribeToConsent(syncPostHogConsent);
}

/**
 * Bring PostHog in line with the current answer. Idempotent, so it is also
 * safe to re-run after posthog.reset(), which drops PostHog's own opt-in.
 */
export function syncPostHogConsent() {
  if (!hasConsentFor("analytics")) {
    // Withdrawal also reloads the page; stop and drop PostHog's storage now,
    // whatever PostHog itself believes its opt-in to be.
    posthog.stopSessionRecording();
    posthog.clear_opt_in_out_capturing();
    clearPostHogStorage(posthog.config.token);
    return;
  }

  if (!posthog.has_opted_in_capturing()) {
    posthog.opt_in_capturing({ captureEventName: false });
  }
  if (posthog.config.persistence !== PERSISTENT_STORAGE) {
    posthog.set_config({ persistence: PERSISTENT_STORAGE });
  }
  if (posthog.config.disable_session_recording) {
    posthog.startSessionRecording();
  }
}

// PostHog keeps ph_<token>_posthog in a cookie and localStorage, and
// ph_<token>_window_id / _primary_window_exists in sessionStorage.
function clearPostHogStorage(token: string) {
  if (typeof window === "undefined" || !token) return;
  const prefix = `ph_${token}_`;
  clearStorage(() => window.localStorage, prefix);
  clearStorage(() => window.sessionStorage, prefix);
  try {
    window.localStorage.removeItem(POSTHOG_CONSENT_KEY);
  } catch {
    // Storage blocked: PostHog cannot have stored an opt-in either.
  }
  document.cookie
    .split(";")
    .map((part) => part.trim().split("=")[0])
    .filter((name) => name.startsWith(prefix))
    .forEach(deleteCookie);
}

function clearStorage(getStorage: () => Storage, prefix: string) {
  try {
    const storage = getStorage();
    const keys = Array.from({ length: storage.length }, (_, index) =>
      storage.key(index),
    );
    keys
      .filter((key): key is string => Boolean(key?.startsWith(prefix)))
      .forEach((key) => storage.removeItem(key));
  } catch {
    // Storage blocked: nothing was stored there.
  }
}

// PostHog sets its cookie on the widest domain it can (.agpt.co from
// platform.agpt.co), so delete the host-only cookie and every parent
// domain's.
function deleteCookie(name: string) {
  const expired = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
  document.cookie = expired;
  const labels = window.location.hostname.split(".");
  for (let index = 0; index < labels.length - 1; index++) {
    document.cookie = `${expired}; Domain=.${labels.slice(index).join(".")}`;
  }
}
