"use client";

import { usePostAnalyticsReportUserAttribution } from "@/app/api/__generated__/endpoints/analytics/analytics";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { buildAttributionPayload } from "@/services/analytics/attribution-payload";
import { useEffect } from "react";

const REPORTED_KEY = "agpt_attribution_reported";
let reportedInMemory: string | null = null;

/**
 * Once per user per browser, tell the backend where this user came from.
 * The backend only ever fills empty fields, so reporting from a second
 * browser later is harmless.
 */
export function useReportAttribution() {
  const { user } = useAuth();
  const { mutate: reportAttribution } = usePostAnalyticsReportUserAttribution();
  const userID = user?.id ?? null;

  useEffect(() => {
    if (!userID || !claimReport(userID)) return;
    reportAttribution(
      { data: buildAttributionPayload() },
      {
        // Only a report the backend accepted is remembered across loads; a
        // failed one is released so the next render (or load) retries it.
        onSuccess: () => markReported(userID),
        onError: () => releaseReport(userID),
      },
    );
  }, [userID, reportAttribution]);
}

function claimReport(userID: string): boolean {
  if (reportedInMemory === userID) return false;
  try {
    if (window.localStorage.getItem(REPORTED_KEY) === userID) return false;
  } catch {
    // Storage blocked: the in-memory guard still prevents repeats this load.
  }
  reportedInMemory = userID;
  return true;
}

function markReported(userID: string): void {
  try {
    window.localStorage.setItem(REPORTED_KEY, userID);
  } catch {
    // Storage blocked: the in-memory guard still prevents repeats this load.
  }
}

function releaseReport(userID: string): void {
  if (reportedInMemory === userID) reportedInMemory = null;
}

export function resetAttributionReportForTests(): void {
  reportedInMemory = null;
}
