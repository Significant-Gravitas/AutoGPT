"use client";

import { isConsentManagerConfigured } from "@/services/consent/consent";
import { useConsent } from "@/services/consent/useConsent";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { environment } from "../environment";
import { DATAFAST_SCRIPT_SRC, resolveAnalyticsLoading } from "./loading-policy";

export function useSetupAnalytics(host: string) {
  const consent = useConsent();
  const pathname = usePathname();
  const { googleTag, dataFast } = resolveAnalyticsLoading({
    host,
    pathname,
    isLocal: environment.isLocal(),
    isConsentManaged: isConsentManagerConfigured(),
    consent,
  });

  useEffect(() => {
    // DataFast's script tracks client-side navigation on its own and can't be
    // unloaded, so once the tour's exemption no longer covers it (the visitor
    // navigated into the app without consenting) the page reloads to shed it.
    // The element stays after unmount, even while still downloading, and is
    // only marked when the exemption loaded it: a script loaded with consent
    // that Cookiebot is asking about again waits for the visitor's reply.
    if (dataFast || !isDataFastLoadedWithoutConsent()) return;
    window.location.reload();
  }, [dataFast]);

  useEffect(() => {
    if (!googleTag) return;

    // Google Analytics: feature usage signal (same as original implementation)
    performance.mark("mark_feature_usage", {
      detail: {
        feature: "custom-ga",
      },
    });
  }, [googleTag]);

  return {
    googleTagEnabled: googleTag,
    dataFastEnabled: dataFast,
    dataFastWithoutConsent: dataFast && !consent.analytics,
  };
}

function isDataFastLoadedWithoutConsent() {
  return Boolean(
    document.querySelector(
      `script[src="${DATAFAST_SCRIPT_SRC}"][data-consent-exempt]`,
    ),
  );
}
