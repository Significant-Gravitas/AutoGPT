import {
  getAnonymousID,
  isAnonymousIDSettled,
  subscribeToAnonymousIDSettled,
} from "@/services/analytics/anonymous-id";
import { environment } from "@/services/environment";
import {
  hasConsentFor,
  hasDeclinedConsentFor,
  subscribeToConsent,
} from "@/services/consent/consent";
import posthog, { type CaptureResult, type PostHogConfig } from "posthog-js";

const POSTHOG_CONSENT_KEY = "agpt_posthog_consent";
const PERSISTENT_STORAGE = "localStorage+cookie";

/**
 * PostHog starts opted out: capture() drops events instead of queueing them
 * and session replay stays off. Without analytics consent it also keeps its
 * state in memory, so nothing an earlier visit stored is read, and
 * forgetPostHogStorageWithoutConsent has deleted it if the visitor declined.
 * Feature flags still evaluate, so experiments keep working for visitors who
 * decline.
 */
export function getConsentGatedConfig() {
  return {
    persistence: canPersist() ? PERSISTENT_STORAGE : "memory",
    opt_out_capturing_by_default: true,
    opt_out_persistence_by_default: true,
    opt_out_capturing_persistence_type: "localStorage",
    consent_persistence_name: POSTHOG_CONSENT_KEY,
    disable_session_recording: true,
  } satisfies Partial<PostHogConfig>;
}

/**
 * Cookiebot is the source of truth. Run before init. Once the visitor has
 * answered without analytics, delete what PostHog stored (its identity, which
 * may belong to an identified user, and its own opt-in), since memory
 * persistence only stops it being read. While the answer is pending only
 * PostHog's own opt-in goes, so it starts opted out; its identity stays,
 * including the `.agpt.co` cookie the marketing site writes, which a grant
 * adopts (services/analytics/anonymous-id.ts).
 */
export function forgetPostHogStorageWithoutConsent(token: string) {
  if (canPersist()) return;
  if (hasDeclinedConsentFor("analytics")) clearPostHogStorage(token);
  else clearPostHogOptIn();
}

// With consent, once the anonymous id is decided: until the session is known
// it may still move to the website's, and PostHog must not write (or send
// under) the page's id meanwhile.
function canPersist() {
  return hasConsentFor("analytics") && isAnonymousIDSettled();
}

/**
 * PostHog `before_send` hook: drops every event (autocapture, page leave and
 * replay included) while analytics is not allowed, for anything sent between
 * the answer changing and syncPostHogConsent catching up.
 */
export function dropWithoutConsent(
  event: CaptureResult | null,
): CaptureResult | null {
  return hasConsentFor("analytics") ? event : null;
}

/**
 * Delete what PostHog stored, the shared `.agpt.co` cookie included. For an
 * account change without analytics consent: PostHog then runs in memory, so
 * its reset() leaves the stored identity behind for the next account to
 * adopt.
 */
export function forgetPostHogStorage() {
  clearPostHogStorage(
    posthog.config?.token || environment.getPostHogCredentials().key || "",
  );
}

/** Call once posthog.init has run; returns the unsubscribe. */
export function followAnalyticsConsent() {
  syncPostHogConsent();
  const unfollowConsent = subscribeToConsent(syncPostHogConsent);
  // The anonymous id can settle later than the grant, once the session is
  // known (anonymous-id.ts).
  const unfollowSettled = subscribeToAnonymousIDSettled(syncPostHogConsent);
  return () => {
    unfollowConsent();
    unfollowSettled();
  };
}

/**
 * Bring PostHog in line with the current answer. Idempotent, so it is also
 * safe to re-run after posthog.reset(), which drops PostHog's own opt-in.
 */
export function syncPostHogConsent() {
  if (!hasConsentFor("analytics")) {
    posthog.stopSessionRecording();
    if (!hasDeclinedConsentFor("analytics")) {
      pauseCapture();
      return;
    }
    // Withdrawal also reloads the page; stop and drop PostHog's storage now,
    // whatever PostHog itself believes its opt-in to be.
    keepPersistenceWithCapture();
    posthog.clear_opt_in_out_capturing();
    clearPostHogStorage(posthog.config.token);
    return;
  }

  // The settled listener comes back once the id is decided.
  if (!isAnonymousIDSettled()) return;
  // Before the storage turns persistent, which writes the shared cookie.
  adoptAnonymousID();
  if (!posthog.has_opted_in_capturing()) {
    posthog.opt_in_capturing({ captureEventName: false });
  }
  keepPersistenceWithCapture();
  if (posthog.config.persistence !== PERSISTENT_STORAGE) {
    posthog.set_config({ persistence: PERSISTENT_STORAGE });
  }
  if (posthog.config.disable_session_recording) {
    posthog.startSessionRecording();
  }
}

/**
 * Pending after PostHog started with consent (Cookiebot no longer accepts the
 * stored answer): stop capturing, which is what stops PostHog writing, since
 * capture() updates the session and person info before before_send drops the
 * event. Its storage stays, the shared `.agpt.co` cookie included: with
 * opt_out_persistence_by_default on, posthog-js would delete it as capture
 * stops (persistence.set_disabled -> remove), as it would on a switch to
 * memory persistence (update_config -> clear), so that is turned off first.
 */
function pauseCapture() {
  if (posthog.config.persistence === "memory") return;
  if (!posthog.has_opted_in_capturing()) return;
  posthog.set_config({ opt_out_persistence_by_default: false });
  posthog.clear_opt_in_out_capturing();
}

// Undoes pauseCapture's setting. Only once PostHog captures again, or when
// its storage is to go anyway: with it on, an opted-out PostHog deletes its
// storage.
function keepPersistenceWithCapture() {
  if (posthog.config.opt_out_persistence_by_default) return;
  posthog.set_config({ opt_out_persistence_by_default: true });
}

/**
 * PostHog started on this page's in-memory id while the answer was pending. On
 * a grant the visitor's anonymous id can turn out to be another one: the
 * website's PostHog device id from the shared `.agpt.co` cookie
 * (anonymous-id.ts). Move PostHog onto it, so its storage keeps that id
 * rather than overwriting the cookie with this page's. Nothing was captured
 * under the in-memory id, since PostHog was opted out, so there is nothing to
 * merge; flags are read again for the adopted id. This is the same call
 * posthog-js makes for `bootstrap`, and it spares the visitor a reload.
 */
function adoptAnonymousID() {
  if (posthog.get_property("$user_state") === "identified") return;
  const id = getAnonymousID();
  if (!id || posthog.get_distinct_id() === id) return;
  posthog.register({ distinct_id: id, $device_id: id });
  posthog.reloadFeatureFlags();
}

function clearPostHogOptIn() {
  try {
    window.localStorage.removeItem(POSTHOG_CONSENT_KEY);
  } catch {
    // Storage blocked: PostHog cannot have stored an opt-in either.
  }
}

// PostHog keeps ph_<token>_posthog in a cookie and localStorage, and
// ph_<token>_window_id / _primary_window_exists in sessionStorage.
function clearPostHogStorage(token: string) {
  if (typeof window === "undefined" || !token) return;
  const prefix = `ph_${token}_`;
  clearStorage(() => window.localStorage, prefix);
  clearStorage(() => window.sessionStorage, prefix);
  clearPostHogOptIn();
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
