import {
  getConsentAnswer,
  isConsentManagerConfigured,
  NO_CONSENT,
  subscribeToConsent,
  type ConsentState,
} from "@/services/consent/consent";
import { DATA_LAYER_NAME } from "./gtag";

// Every Google signal starts denied until the visitor answers the banner,
// except in the US, whose Cookiebot domain group is opt-out: there the tag
// runs with consent granted until the visitor opts out. Keep agpt.co in step
// so a click ID collected there is handled the same way here.
export const CONSENT_GRANTED_BY_DEFAULT_REGIONS = ["US"];

// How long the tag holds its first hit for the banner's stored answer, so a
// returning visitor's first page view already carries it.
const WAIT_FOR_UPDATE_MS = 500;

// Consent Mode v2 defaults, rendered as a beforeInteractive script ahead of
// Cookiebot and the Google tag. The updates come from followConsentForGoogleTag
// below. The shim stays local so it doesn't define window.gtag: that global is
// how the rest of the app knows the tag itself loaded.
export function buildConsentDefaultsScript(): string {
  return [
    `window['${DATA_LAYER_NAME}'] = window['${DATA_LAYER_NAME}'] || [];`,
    `(function(){`,
    `function gtag(){window['${DATA_LAYER_NAME}'].push(arguments);}`,
    // Google applies the default whose region most specifically matches the
    // visitor and falls back to the one without a region. The regional one
    // goes first, as in Google's own example.
    `gtag('consent','default',${JSON.stringify({
      ad_storage: "granted",
      ad_user_data: "granted",
      ad_personalization: "granted",
      analytics_storage: "granted",
      region: CONSENT_GRANTED_BY_DEFAULT_REGIONS,
      wait_for_update: WAIT_FOR_UPDATE_MS,
    })});`,
    `gtag('consent','default',${JSON.stringify({
      ad_storage: "denied",
      ad_user_data: "denied",
      ad_personalization: "denied",
      analytics_storage: "denied",
      wait_for_update: WAIT_FOR_UPDATE_MS,
    })});`,
    // Carries the ad click ID across pages in the URL while cookies are denied.
    `gtag('set','url_passthrough',true);`,
    `})();`,
  ].join("\n");
}

type Signal = "granted" | "denied";

function signal(granted: boolean): Signal {
  return granted ? "granted" : "denied";
}

export function buildConsentUpdate(consent: ConsentState) {
  const ads = signal(consent.advertising);
  return {
    analytics_storage: signal(consent.analytics),
    ad_storage: ads,
    ad_user_data: ads,
    ad_personalization: ads,
  };
}

/**
 * Sends the visitor's answer to the Google tag as a Consent Mode update, now
 * and whenever it changes. Cookiebot's own Consent Mode integration sends the
 * same signals; this keeps a denial reaching the tag even if that integration
 * is switched off in the Cookiebot admin. Until there is an answer the region
 * defaults stand; an answer Cookiebot later withdraws is sent as a denial.
 * Returns the unsubscribe.
 */
export function followConsentForGoogleTag(): () => void {
  if (typeof window === "undefined" || !isConsentManagerConfigured()) {
    return () => {};
  }
  let sent: string | null = null;

  function sync() {
    const answer = getConsentAnswer() ?? (sent ? NO_CONSENT : null);
    if (!answer) return;
    const update = buildConsentUpdate(answer);
    const key = JSON.stringify(update);
    if (key === sent) return;
    sent = key;
    queueGtagCommand("consent", "update", update);
  }

  sync();
  return subscribeToConsent(sync);
}

// Queued straight onto the dataLayer so the update lands whether or not the
// tag has loaded yet. gtag.js only runs entries that are real `arguments`
// objects, hence the rest parameter goes unused.
function queueGtagCommand(..._command: unknown[]) {
  const scope = window as unknown as Record<string, unknown[] | undefined>;
  const dataLayer = (scope[DATA_LAYER_NAME] ??= []);
  // eslint-disable-next-line prefer-rest-params
  dataLayer.push(arguments);
}
