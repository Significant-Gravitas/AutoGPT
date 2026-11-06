import { useSyncExternalStore } from "react";

// Below Tailwind's `md`: a phone. Otto and his question come down a size.
const NARROW_SCREEN_QUERY = "(max-width: 767px)";

function subscribe(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(NARROW_SCREEN_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function isNarrowScreen() {
  return Boolean(window.matchMedia?.(NARROW_SCREEN_QUERY).matches);
}

export function useIsNarrowScreen(): boolean {
  return useSyncExternalStore(subscribe, isNarrowScreen, () => false);
}
