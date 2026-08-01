"use client";

import { consent } from "@/services/consent/cookies";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { useEffect, useState } from "react";

export function VercelAnalyticsWrapper({
  enabled = true,
}: {
  enabled?: boolean;
}) {
  const [hasAnalyticsConsent, setHasAnalyticsConsent] = useState(false);

  useEffect(() => {
    setHasAnalyticsConsent(consent.hasConsentFor("analytics"));
  }, []);

  if (!enabled || !hasAnalyticsConsent) {
    return null;
  }

  return (
    <>
      <SpeedInsights />
      <Analytics />
    </>
  );
}
