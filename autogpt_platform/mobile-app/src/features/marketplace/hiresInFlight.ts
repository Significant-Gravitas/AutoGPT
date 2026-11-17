import { useSyncExternalStore } from "react";

// Shared by the list and the details page, so a hire started on one shows,
// and can't be sent again, on the other.
let inFlight: ReadonlySet<string> = new Set();
const listeners = new Set<() => void>();

function publish(next: ReadonlySet<string>) {
  inFlight = next;
  listeners.forEach((listener) => listener());
}

/** Marks a hire as started; false when that template's hire is already running. */
export function claimHire(templateID: string) {
  if (inFlight.has(templateID)) return false;
  publish(new Set([...inFlight, templateID]));
  return true;
}

export function releaseHire(templateID: string) {
  if (!inFlight.has(templateID)) return;
  const next = new Set(inFlight);
  next.delete(templateID);
  publish(next);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function snapshot() {
  return inFlight;
}

export function useHiresInFlight() {
  return useSyncExternalStore(subscribe, snapshot);
}
