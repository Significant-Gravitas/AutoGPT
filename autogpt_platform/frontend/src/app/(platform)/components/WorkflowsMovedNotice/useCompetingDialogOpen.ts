"use client";

import { useSyncExternalStore } from "react";
import { hasCompetingDialog } from "./helpers";

// Whether another dialog is open right now, followed live. For a surface
// that steps aside while one is open and comes back once it closes, such
// as a contextual tip. A one-off notice that should rather wait for the
// next visit uses useNoticeAdmission.
export function useCompetingDialogOpen(isWatching: boolean) {
  return useSyncExternalStore(
    isWatching ? subscribeToDialogs : subscribeToNothing,
    isWatching ? hasCompetingDialog : isNeverOpen,
    isNeverOpen,
  );
}

function subscribeToDialogs(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["role", "hidden", "data-state"],
  });
  return () => observer.disconnect();
}

function subscribeToNothing() {
  return () => {};
}

function isNeverOpen() {
  return false;
}
