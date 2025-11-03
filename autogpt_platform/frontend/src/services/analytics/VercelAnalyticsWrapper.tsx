"use client";

import { hasConsentFor } from "@/lib/consent";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { useEffect, useState } from "react";

/**
 * Wrapper for Vercel Analytics and Speed Insights that respects cookie consent
 */
export function VercelAnalyticsWrapper() {
  const [hasAnalyticsConsent, setHasAnalyticsConsent] = useState(false);

  useEffect(() => {
    // Check consent on mount
    setHasAnalyticsConsent(hasConsentFor("analytics"));
  }, []);

  if (!hasAnalyticsConsent) {
    return null;
  }

  return (
    <>
      <SpeedInsights />
      <Analytics />
    </>
  );
}
