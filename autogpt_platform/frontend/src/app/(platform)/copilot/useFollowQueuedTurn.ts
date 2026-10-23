import { useEffect, useRef } from "react";

// The runtime chat reads a queued session at the same pace.
const QUEUED_POLL_MS = 10_000;

interface Args {
  enabled: boolean;
  sessionChatStatus: string | undefined;
  refetchSession: () => Promise<unknown>;
  followBackendTurn: () => void;
}

/**
 * A turn queued behind the user's running cap starts when another of their
 * turns ends, which nothing tells this tab. Re-read the session while it is
 * queued, and once it leaves the queue follow its turn as an approval's is
 * followed, so the answer streams in instead of a "Queued" badge staying up.
 */
export function useFollowQueuedTurn({
  enabled,
  sessionChatStatus,
  refetchSession,
  followBackendTurn,
}: Args) {
  const refetchRef = useRef(refetchSession);
  refetchRef.current = refetchSession;
  const followRef = useRef(followBackendTurn);
  followRef.current = followBackendTurn;
  const wasQueuedRef = useRef(false);

  useEffect(() => {
    if (!enabled) return;
    if (sessionChatStatus === "queued") {
      wasQueuedRef.current = true;
      const timer = setInterval(
        () => void refetchRef.current(),
        QUEUED_POLL_MS,
      );
      return () => clearInterval(timer);
    }
    if (!wasQueuedRef.current) return;
    wasQueuedRef.current = false;
    followRef.current();
  }, [enabled, sessionChatStatus]);
}
