import type { ConsentState } from "@/services/consent/consent";

interface LoadingArgs {
  host: string;
  pathname: string | null;
  isLocal: boolean;
  isConsentManaged: boolean;
  consent: ConsentState;
}

export function resolveAnalyticsLoading({
  host,
  pathname,
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
    // locally only send analytics after opting in.
    googleTag:
      (isProductionDomain && isConsentManaged) ||
      (isLocal && consent.analytics),
    dataFast:
      isProductionDomain &&
      (consent.analytics ||
        isDataFastConsentExempt(pathname, isConsentManaged)),
  };
}

// TODO(SECRT-2713): the public tour loads DataFast without the consent gate
// so tour funnel events fire for first-touch visitors. The old banner stayed
// hidden on /tour; Cookiebot's does not, so revisit this exemption there.
// Without a banner nothing optional loads, the tour included.
export function isDataFastConsentExempt(
  pathname: string | null,
  isConsentManaged: boolean,
): boolean {
  return isConsentManaged && isTourPath(pathname);
}

// Segment-boundary match: /tourism must not inherit the tour's consent
// exemption.
function isTourPath(pathname: string | null): boolean {
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
