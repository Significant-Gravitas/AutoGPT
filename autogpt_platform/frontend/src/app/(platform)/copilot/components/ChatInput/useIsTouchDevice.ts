import { useSyncExternalStore } from "react";

const COARSE_POINTER_QUERY = "(pointer: coarse)";

function subscribe(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(COARSE_POINTER_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function getSnapshot() {
  return (
    typeof window !== "undefined" &&
    !!window.matchMedia &&
    window.matchMedia(COARSE_POINTER_QUERY).matches
  );
}

/**
 * True when the primary input is a finger. Camera capture only makes sense
 * there: on a desktop browser `capture` degrades to a plain file chooser,
 * which "Attach file" already offers.
 */
export function useIsTouchDevice(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}
