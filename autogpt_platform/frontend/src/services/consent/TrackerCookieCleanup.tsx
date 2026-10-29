"use client";

import { useEffect } from "react";
import { followConsentForTrackerCookies } from "./tracker-cookies";

// Rendered ahead of ConsentWithdrawalReload, so a withdrawal deletes the
// cookies before the page reloads; the reloaded page checks again.
export function TrackerCookieCleanup() {
  useEffect(() => followConsentForTrackerCookies(), []);
  return null;
}
