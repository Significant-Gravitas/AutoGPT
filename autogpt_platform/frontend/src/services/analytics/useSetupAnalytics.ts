"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { consent, type ConsentPreferences } from "@/services/consent/cookies";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { environment } from "../environment";
import { resolveAnalyticsLoading } from "./loading-policy";

export function useSetupAnalytics(host: string) {
  // Stored consent is only readable in the browser; the tag waits for it so
  // the init script can replay the visitor's answer through Consent Mode.
  const [preferences, setPreferences] = useState<ConsentPreferences | null>(
    null,
  );
  useMountEffect(() => {
    setPreferences(consent.load());
  });

  const pathname = usePathname();
  const { googleTag, dataFast } = resolveAnalyticsLoading({
    host,
    pathname,
    isLocal: environment.isLocal(),
    preferences,
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
    preferences,
    googleTagEnabled: googleTag,
    dataFastEnabled: dataFast,
  };
}
