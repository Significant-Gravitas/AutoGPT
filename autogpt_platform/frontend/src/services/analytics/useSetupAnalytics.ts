"use client";

import { isConsentManagerConfigured } from "@/services/consent/consent";
import { useConsent } from "@/services/consent/useConsent";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { environment } from "../environment";
import { resolveAnalyticsLoading } from "./loading-policy";

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
    if (dataFast || !window.datafast) return;
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
  };
}
