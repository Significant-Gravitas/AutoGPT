import * as SecureStore from "expo-secure-store";

import { isConsent, type Consent } from "./rules";

const STORAGE_KEY = "autogpt.analyticsConsent";

export function readConsent(): Consent {
  try {
    const stored = SecureStore.getItem(STORAGE_KEY);
    return isConsent(stored) ? stored : "unset";
  } catch {
    return "unset";
  }
}

export function saveConsent(consent: Consent) {
  try {
    SecureStore.setItem(STORAGE_KEY, consent);
  } catch {
    // The choice still holds until the app restarts.
  }
}
