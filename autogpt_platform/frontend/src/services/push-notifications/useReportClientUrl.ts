"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { useEffect } from "react";

/**
 * Tells the push service worker the current page URL on every navigation.
 *
 * Why: Chrome's `WindowClient.url` doesn't update for Next.js client-side
 * navigation (history.pushState), so the SW's suppression check sees stale
 * URLs and wrongly suppresses notifications when the user has navigated
 * away from the completed session's page.
 */
export function useReportClientUrl() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (typeof navigator === "undefined") return;
    if (!navigator.serviceWorker) return;
    const query = searchParams?.toString();
    const url = pathname + (query ? `?${query}` : "");

    function send() {
      navigator.serviceWorker.controller?.postMessage({
        type: "CLIENT_URL",
        url,
      });
    }

    send();
    // Also send when control of the page transfers to a new SW (first install
    // after subscribe, or after an update). Otherwise the very first navigation
    // after registration races against controller availability.
    navigator.serviceWorker.addEventListener("controllerchange", send);
    return () =>
      navigator.serviceWorker.removeEventListener("controllerchange", send);
  }, [pathname, searchParams]);
}
