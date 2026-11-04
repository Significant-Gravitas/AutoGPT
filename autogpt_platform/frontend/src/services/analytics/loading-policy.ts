import type { ConsentState } from "@/services/consent/consent";

export const DATAFAST_SCRIPT_SRC = "https://datafa.st/js/script.js";

interface LoadingArgs {
  host: string;
  isLocal: boolean;
  isConsentManaged: boolean;
  consent: ConsentState;
}

export function resolveAnalyticsLoading({
  host,
  isLocal,
  isConsentManaged,
  consent,
}: LoadingArgs) {
  const isProductionDomain = isProductionHost(host);

  return {
    // Production loads the Google tag before the visitor answers: Consent Mode
    // keeps it cookieless where consent is required (see consent-mode.ts) and
    // Cookiebot sends the answer once there is one. Without a banner nothing
    // could ever answer, so the tag stays off. Open-source developers running
    // locally only send analytics after opting in. DataFast waits for consent
    // everywhere, the public tour included.
    googleTag:
      (isProductionDomain && isConsentManaged) ||
      (isLocal && consent.analytics),
    dataFast: isProductionDomain && consent.analytics,
  };
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
