import { useEffect, useRef, useState } from "react";
import { hasActiveBackendStream } from "./helpers";

/** Minimum time the page must have been hidden to trigger a wake re-sync. */
const WAKE_RESYNC_THRESHOLD_MS = 30_000;

interface UseWakeResyncArgs {
  sessionIdRef: React.MutableRefObject<string | null>;
  isMountedRef: React.MutableRefObject<boolean>;
  refetchSession: () => Promise<{ data?: unknown }>;
  resumeStreamRef: React.MutableRefObject<() => void>;
}

/**
 * Re-sync the chat state when the page becomes visible after a long hidden
 * period (device sleep, tab backgrounded for a long time). If the backend
 * stream is still active, resume via `resumeStreamRef`; otherwise rely on
 * React Query's refetch to update `hydratedMessages`, which the hydration
 * effect merges in.
 *
 * Returns `isSyncing` for the UI so it can block chat input during the
 * in-flight refetch.
 */
export function useWakeResync({
  sessionIdRef,
  isMountedRef,
  refetchSession,
  resumeStreamRef,
}: UseWakeResyncArgs) {
  const [isSyncing, setIsSyncing] = useState(false);
  const lastHiddenAtRef = useRef(Date.now());

  useEffect(() => {
    async function handleWakeResync() {
      const sid = sessionIdRef.current;
      if (!sid) return;

      const elapsed = Date.now() - lastHiddenAtRef.current;
      lastHiddenAtRef.current = Date.now();

      if (document.visibilityState !== "visible") return;
      if (elapsed < WAKE_RESYNC_THRESHOLD_MS) return;

      setIsSyncing(true);
      try {
        const result = await refetchSession();
        // Bail out if the session changed or the host unmounted while
        // the refetch was in flight.
        if (!isMountedRef.current) return;
        if (sessionIdRef.current !== sid) return;

        if (hasActiveBackendStream(result)) {
          // Stream is still running — resume SSE to pick up live chunks.
          resumeStreamRef.current();
        }
      } catch (err) {
        console.warn("[copilot] wake re-sync failed", err);
      } finally {
        if (isMountedRef.current) setIsSyncing(false);
      }
    }

    function onVisibilityChange() {
      if (document.visibilityState === "hidden") {
        lastHiddenAtRef.current = Date.now();
      } else {
        handleWakeResync();
      }
    }

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [refetchSession, isMountedRef, resumeStreamRef, sessionIdRef]);

  return { isSyncing };
}
