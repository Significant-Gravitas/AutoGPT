"use client";

import { useEffect } from "react";
import { followConsentForGoogleTag } from "./consent-mode";

// Rendered ahead of the Google tag's scripts: sibling effects run in order, so
// a returning visitor's stored answer is queued before the tag's config.
export function GoogleConsentModeSync() {
  useEffect(() => followConsentForGoogleTag(), []);
  return null;
}
