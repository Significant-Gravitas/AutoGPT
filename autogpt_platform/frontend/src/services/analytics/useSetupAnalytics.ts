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
