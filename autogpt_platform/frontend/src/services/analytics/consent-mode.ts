import { DATA_LAYER_NAME } from "./gtag";

const EU_MEMBER_STATES = [
  "AT",
  "BE",
  "BG",
  "HR",
  "CY",
  "CZ",
  "DK",
  "EE",
  "FI",
  "FR",
  "DE",
  "GR",
  "HU",
  "IE",
  "IT",
  "LV",
  "LT",
  "LU",
  "MT",
  "NL",
  "PL",
  "PT",
  "RO",
  "SK",
  "SI",
  "ES",
  "SE",
];

// EEA (EU + IS, LI, NO), the UK and Switzerland start with every Google
// signal denied until the visitor answers the banner; everywhere else the tag
// runs with consent granted by default. Mirrored on agpt.co so a click ID
// collected there is handled the same way here.
export const CONSENT_DENIED_BY_DEFAULT_REGIONS = [
  ...EU_MEMBER_STATES,
  "IS",
  "LI",
  "NO",
  "GB",
  "CH",
];

// How long the tag holds its first hit for the banner's stored answer, so a
// returning visitor's first page view already carries it.
const WAIT_FOR_UPDATE_MS = 500;

// Consent Mode v2 defaults, rendered as a beforeInteractive script ahead of
// Cookiebot and the Google tag. Cookiebot sends every `consent update` itself
// (statistics → analytics_storage; marketing → ad_storage, ad_user_data and
// ad_personalization). The shim stays local so it doesn't define window.gtag:
// that global is how the rest of the app knows the tag itself loaded.
export function buildConsentDefaultsScript(): string {
  return [
    `window['${DATA_LAYER_NAME}'] = window['${DATA_LAYER_NAME}'] || [];`,
    `(function(){`,
    `function gtag(){window['${DATA_LAYER_NAME}'].push(arguments);}`,
    `gtag('consent','default',${JSON.stringify({
      ad_storage: "granted",
      ad_user_data: "granted",
      ad_personalization: "granted",
      analytics_storage: "granted",
    })});`,
    `gtag('consent','default',${JSON.stringify({
      ad_storage: "denied",
      ad_user_data: "denied",
      ad_personalization: "denied",
      analytics_storage: "denied",
      region: CONSENT_DENIED_BY_DEFAULT_REGIONS,
      wait_for_update: WAIT_FOR_UPDATE_MS,
    })});`,
    // Carries the ad click ID across pages in the URL while cookies are denied.
    `gtag('set','url_passthrough',true);`,
    `})();`,
  ].join("\n");
}
