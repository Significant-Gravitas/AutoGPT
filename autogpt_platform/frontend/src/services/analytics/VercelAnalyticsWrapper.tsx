"use client";

import { useConsent } from "@/services/consent/useConsent";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";

export function VercelAnalyticsWrapper() {
  const { analytics } = useConsent();

  if (!analytics) {
    return null;
  }

  return (
    <>
      <SpeedInsights />
      <Analytics />
    </>
  );
}
