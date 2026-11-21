import { useEffect, useRef } from "react";

/**
 * Calls `onRestored` when Back or Forward brings the page back from the
 * browser's cache. The page returns exactly as it was left, partway through
 * starting a sign-in included, and none of its effects run again.
 */
export function usePageRestored(onRestored: () => void) {
  // The latest callback, without listening again on every render. Not
  // useEffectEvent: the React that Next's App Router renders with lacks it.
  const latest = useRef(onRestored);
  useEffect(() => {
    latest.current = onRestored;
  });
  useEffect(() => {
    function onPageShow(event: PageTransitionEvent) {
      if (event.persisted) latest.current();
    }
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, []);
}
