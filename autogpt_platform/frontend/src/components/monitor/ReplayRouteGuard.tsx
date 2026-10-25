"use client";

import { syncReplayWithPage } from "@/lib/session-replay";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect } from "react";

/**
 * Brings session replay back once a navigation lands on a page whose address
 * carries no credential.
 */
export function ReplayRouteGuard() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    syncReplayWithPage(window.location.href);
  }, [pathname, searchParams]);

  return null;
}
