import { useSyncExternalStore } from "react";

const CHECK_INTERVAL_MS = 60_000;

const listeners = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | null = null;
let current = localDay();

function localDay() {
  return new Date().toDateString();
}

function check() {
  const next = localDay();
  if (next === current) return;
  current = next;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (timer === null) {
    current = localDay();
    timer = setInterval(check, CHECK_INTERVAL_MS);
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size > 0 || timer === null) return;
    clearInterval(timer);
    timer = null;
  };
}

function getSnapshot() {
  return timer === null ? localDay() : current;
}

/**
 * The local calendar day, e.g. "Tue Oct 20 2026". One shared timer checks it
 * once a minute, and callers re-render only when the day changes, so relative
 * date labels ("Today", "Yesterday") in memoised lists move on at midnight.
 */
export function useLocalDay() {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
