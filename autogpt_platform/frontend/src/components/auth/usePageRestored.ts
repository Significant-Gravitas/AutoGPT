import { useEffect } from "react";

/**
 * Calls `onRestored` when Back or Forward brings the page back from the
 * browser's cache. The page returns exactly as it was left, partway through
 * starting a sign-in included, and none of its effects run again.
 */
export function usePageRestored(onRestored: () => void) {
  useEffect(() => {
    function onPageShow(event: PageTransitionEvent) {
      if (event.persisted) onRestored();
    }
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, [onRestored]);
}
