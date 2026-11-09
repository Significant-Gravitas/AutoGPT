import { useSyncExternalStore } from "react";

// A laptop at 1280x720 or 1366x768: the typing screen gives up composer
// rows there so "I'm done" and "record instead" stay above the fold.
const SHORT_SCREEN_QUERY = "(max-height: 800px)";

function subscribe(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(SHORT_SCREEN_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function isShortScreen() {
  return Boolean(window.matchMedia?.(SHORT_SCREEN_QUERY).matches);
}

export function useIsShortScreen(): boolean {
  return useSyncExternalStore(subscribe, isShortScreen, () => false);
}
