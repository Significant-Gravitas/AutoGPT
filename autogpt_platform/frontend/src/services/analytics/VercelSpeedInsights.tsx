"use client";

import { useConsent } from "@/services/consent/useConsent";
import { SpeedInsights } from "@vercel/speed-insights/next";

export function VercelSpeedInsights() {
  const { analytics } = useConsent();

  if (!analytics) {
    return null;
  }

  return <SpeedInsights />;
}
