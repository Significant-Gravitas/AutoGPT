"use client";

import { isConsentManagerConfigured } from "@/services/consent/consent";
import { useConsent } from "@/services/consent/useConsent";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { environment } from "../environment";
import { resolveAnalyticsLoading } from "./loading-policy";
import { pageHoldsCredential } from "./url-credentials";

export function useSetupAnalytics(host: string) {
  const consent = useConsent();
  // Re-renders on navigation, so DataFast loads once the visitor leaves a
  // page whose URL holds a credential.
  usePathname();
  const { googleTag, dataFast } = resolveAnalyticsLoading({
    host,
    isLocal: environment.isLocal(),
    isConsentManaged: isConsentManagerConfigured(),
    consent,
    pageHoldsCredential: pageHoldsCredential(),
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
