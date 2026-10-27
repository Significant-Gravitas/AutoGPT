"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { useFirstRunTourStore } from "./store";

// Opens a "Show me around" asked for on another page once the route has
// changed to Home, after the page left has handed the sidebar back.
export function useReplayOnArrival() {
  const pathname = usePathname();
  const isPending = useFirstRunTourStore((state) => state.pendingReplay);
  const startPendingReplay = useFirstRunTourStore(
    (state) => state.startPendingReplay,
  );

  useEffect(() => {
    if (isPending && pathname === "/home") startPendingReplay();
  }, [isPending, pathname, startPendingReplay]);
}
