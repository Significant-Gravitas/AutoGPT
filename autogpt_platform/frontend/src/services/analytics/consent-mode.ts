import type { ConsentPreferences } from "@/services/consent/cookies";

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

type Signal = "granted" | "denied";

function signal(granted: boolean): Signal {
  return granted ? "granted" : "denied";
}

// Consent Mode v2 commands for the Google tag init script. Must run before
// gtag('config', …) so the first hit already carries the right signals.
export function buildConsentModeScript(
  preferences: ConsentPreferences | null,
): string {
  const lines = [
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
    })});`,
    // Carries the ad click ID across pages in the URL while cookies are denied.
    `gtag('set','url_passthrough',true);`,
  ];

  if (preferences?.hasConsented) {
    const ads = signal(preferences.advertising);
    lines.push(
      `gtag('consent','update',${JSON.stringify({
        analytics_storage: signal(preferences.analytics),
        ad_storage: ads,
        ad_user_data: ads,
        ad_personalization: ads,
      })});`,
    );
  }

  return lines.join("\n");
}
