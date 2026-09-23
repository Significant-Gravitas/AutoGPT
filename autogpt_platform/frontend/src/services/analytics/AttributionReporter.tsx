"use client";

import { useReportAttribution } from "@/services/analytics/useReportAttribution";

export function AttributionReporter() {
  useReportAttribution();
  return null;
}
