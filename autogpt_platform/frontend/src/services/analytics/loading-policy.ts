import type { ConsentPreferences } from "@/services/consent/cookies";

interface LoadingArgs {
  host: string;
  pathname: string | null;
  isLocal: boolean;
  preferences: ConsentPreferences | null;
}

export function resolveAnalyticsLoading({
  host,
  pathname,
  isLocal,
  preferences,
}: LoadingArgs) {
  // Stored consent is only readable in the browser; nothing loads until it is.
  if (!preferences) return { googleTag: false, dataFast: false };

  const isProductionDomain = isProductionHost(host);
  const hasAnalyticsConsent = preferences.hasConsented && preferences.analytics;

  return {
    // Production loads the Google tag before the visitor answers: Consent Mode
    // keeps it cookieless where consent is required (see consent-mode.ts).
    // Open-source developers running locally only send analytics after opting in.
    googleTag: isProductionDomain || (isLocal && hasAnalyticsConsent),
    // The public tour hides the cookie banner, so DataFast loads there without
    // the consent gate — otherwise tour funnel events would never fire for
    // first-touch visitors.
    dataFast:
      isProductionDomain && (hasAnalyticsConsent || isTourPath(pathname)),
  };
}

// Segment-boundary match: /tourism must not inherit the tour's consent
// exemption.
export function isTourPath(pathname: string | null): boolean {
  if (!pathname) return false;
  return pathname === "/tour" || pathname.startsWith("/tour/");
}

const PRODUCTION_HOST = "platform.agpt.co";

// The Host header is client-controlled, so a substring match would hand the
// production tag to `platform.agpt.co.example.com` (and to
// `notplatform.agpt.co`). Compare the hostname exactly, minus the port, the
// casing and the root-label dot a client can legally send.
function isProductionHost(host: string): boolean {
  const hostname = host.toLowerCase().split(":")[0].replace(/\.$/, "");
  return hostname === PRODUCTION_HOST;
}
