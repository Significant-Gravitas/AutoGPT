import { useSyncExternalStore } from "react";

// Tailwind's `xl`: from here the starter ideas fit beside the widest column
// (the typing composer). Narrower screens show the live prompt at rest
// instead, as a phone does.
const WIDE_LAYOUT_QUERY = "(min-width: 1280px)";

function subscribe(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(WIDE_LAYOUT_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function isWideLayout() {
  return Boolean(window.matchMedia?.(WIDE_LAYOUT_QUERY).matches);
}

export function useIsWideLayout(): boolean {
  return useSyncExternalStore(subscribe, isWideLayout, () => false);
}
