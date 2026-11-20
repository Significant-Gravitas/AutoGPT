import * as SecureStore from "expo-secure-store";
import { useSyncExternalStore } from "react";

import { FRONTEND_URL } from "@/lib/env";

import { onboardedKey } from "./gate";

export { onboardedKey } from "./gate";

// Never cleared on sign-out: finishing onboarding is a fact about the account.
const known = new Map<string, boolean>();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isOnboarded(userID: string) {
  const cached = known.get(userID);
  if (cached !== undefined) return cached;
  let value = false;
  try {
    value = SecureStore.getItem(onboardedKey(FRONTEND_URL, userID)) === "1";
  } catch {
    // Unreadable: ask the server instead.
  }
  known.set(userID, value);
  return value;
}

export function markOnboarded(userID: string) {
  known.set(userID, true);
  try {
    SecureStore.setItem(onboardedKey(FRONTEND_URL, userID), "1");
  } catch {
    // Still holds until the app restarts; the server is asked again then.
  }
  listeners.forEach((listener) => listener());
}

export function useIsOnboarded(userID: string | undefined) {
  return useSyncExternalStore(subscribe, () =>
    userID ? isOnboarded(userID) : false,
  );
}
