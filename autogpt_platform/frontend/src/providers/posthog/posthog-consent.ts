import { hasConsentFor, subscribeToConsent } from "@/services/consent/consent";
import posthog, { type PostHogConfig } from "posthog-js";

const POSTHOG_CONSENT_KEY = "agpt_posthog_consent";
const PERSISTENT_STORAGE = "localStorage+cookie";

/**
 * PostHog starts opted out: capture() drops events instead of queueing them
 * and session replay stays off. Without analytics consent it also keeps its
 * state in memory, ignoring anything an earlier visit stored. Feature flags
 * still evaluate, so experiments keep working for visitors who decline.
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
 * Cookiebot is the source of truth. PostHog remembers its own opt-in, so a
 * grant taken back since (in another tab, or by an expired Cookiebot answer)
 * has to be forgotten before init reads it.
 */
export function forgetWithdrawnPostHogConsent() {
  if (hasConsentFor("analytics")) return;
  try {
    window.localStorage.removeItem(POSTHOG_CONSENT_KEY);
  } catch {
    // Storage blocked: PostHog cannot have stored an opt-in either.
  }
}

/** Call once posthog.init has run; returns the unsubscribe. */
export function followAnalyticsConsent() {
  syncPostHogConsent();
  return subscribeToConsent(syncPostHogConsent);
}

function syncPostHogConsent() {
  if (!hasConsentFor("analytics")) {
    // Withdrawal also reloads the page; stop and drop PostHog's storage now.
    if (posthog.has_opted_in_capturing()) {
      posthog.stopSessionRecording();
      posthog.clear_opt_in_out_capturing();
    }
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
