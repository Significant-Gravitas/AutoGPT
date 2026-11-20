"use client";

import { useEffect } from "react";
import { followConsentForTrackerCookies } from "./tracker-cookies";

// Rendered ahead of ConsentWithdrawalReload, so a withdrawal deletes the
// trackers' cookies and storage before the page reloads.
export function TrackerCookieCleanup() {
  useEffect(() => followConsentForTrackerCookies(), []);
  return null;
}
