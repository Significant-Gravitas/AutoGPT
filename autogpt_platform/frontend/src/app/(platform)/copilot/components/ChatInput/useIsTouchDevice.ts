import { COARSE_POINTER_QUERY, isTouchDevice } from "@/lib/platform";
import { useSyncExternalStore } from "react";

function subscribe(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(COARSE_POINTER_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/**
 * True when the primary input is a finger. Camera capture only makes sense
 * there: on a desktop browser `capture` degrades to a plain file chooser,
 * which "Attach file" already offers.
 */
export function useIsTouchDevice(): boolean {
  return useSyncExternalStore(subscribe, isTouchDevice, () => false);
}
