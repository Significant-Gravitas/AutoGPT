"use client";

import { useSyncExternalStore } from "react";
import {
  getConsent,
  getConsentManagerStatus,
  isSameConsent,
  NO_CONSENT,
  subscribeToConsent,
  subscribeToConsentManagerStatus,
  type ConsentManagerStatus,
  type ConsentState,
} from "./consent";

let snapshot: ConsentState = NO_CONSENT;

// useSyncExternalStore needs the same object back while nothing changed.
function getSnapshot(): ConsentState {
  const next = getConsent();
  if (!isSameConsent(snapshot, next)) snapshot = next;
  return snapshot;
}

function getServerSnapshot(): ConsentState {
  return NO_CONSENT;
}

/** Current consent; re-renders when the visitor answers or changes the banner. */
export function useConsent() {
  return useSyncExternalStore(
    subscribeToConsent,
    getSnapshot,
    getServerSnapshot,
  );
}

/** Whether the consent dialog can be opened; see getConsentManagerStatus. */
export function useConsentManagerStatus() {
  return useSyncExternalStore(
    subscribeToConsentManagerStatus,
    getConsentManagerStatus,
    (): ConsentManagerStatus => "loading",
  );
}
