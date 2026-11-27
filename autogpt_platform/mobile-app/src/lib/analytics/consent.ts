import * as SecureStore from "expo-secure-store";
import Storage from "expo-sqlite/kv-store";

import type { Consent } from "./rules";

const STORAGE_KEY = "autogpt.analyticsConsent";

export function readConsent(): Consent {
  const secureChoice = tryRead(() => SecureStore.getItem(STORAGE_KEY));
  const localChoice = tryRead(() => Storage.getItemSync(STORAGE_KEY));
  if (secureChoice === "declined" || localChoice === "declined") return "declined";
  return secureChoice === "allowed" && localChoice === "allowed" ? "allowed" : "unset";
}

function tryRead(read: () => string | null) {
  try {
    return read();
  } catch {
    return null;
  }
}

// Either store can record a revocation. A saved opt-in requires both stores to agree.
export function saveConsent(consent: Exclude<Consent, "unset">) {
  const secureSaved = trySave(() => SecureStore.setItem(STORAGE_KEY, consent));
  const localSaved = trySave(() => Storage.setItemSync(STORAGE_KEY, consent));
  return consent === "allowed" ? secureSaved && localSaved : secureSaved || localSaved;
}

function trySave(write: () => void) {
  try {
    write();
    return true;
  } catch {
    return false;
  }
}
