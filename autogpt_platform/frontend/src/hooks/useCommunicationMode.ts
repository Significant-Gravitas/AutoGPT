import { Key, storage } from "@/services/storage/local-storage";
import { useSyncExternalStore } from "react";

export type CommunicationMode = "compact" | "technical";

const DEFAULT_MODE: CommunicationMode = "technical";
const listeners = new Set<() => void>();

function readMode(): CommunicationMode {
  return storage.get(Key.COMMUNICATION_MODE) === "compact"
    ? "compact"
    : DEFAULT_MODE;
}

function readServerMode(): CommunicationMode {
  return DEFAULT_MODE;
}

function onStorage(event: StorageEvent) {
  if (event.key === Key.COMMUNICATION_MODE) listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (listeners.size === 1) window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) window.removeEventListener("storage", onStorage);
  };
}

// TODO(backend): persist on the user's preferences once the API exposes a
// communication-mode field. Until then the choice is per browser.
export function useCommunicationMode() {
  const mode = useSyncExternalStore(subscribe, readMode, readServerMode);

  function setMode(next: CommunicationMode) {
    storage.set(Key.COMMUNICATION_MODE, next);
    listeners.forEach((listener) => listener());
  }

  return { mode, setMode };
}
