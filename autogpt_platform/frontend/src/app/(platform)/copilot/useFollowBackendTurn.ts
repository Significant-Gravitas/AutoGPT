import { useEffect, useRef } from "react";
import { hasActiveBackendStream } from "./helpers";

interface Args {
  status: string;
  refetchSession: () => Promise<{ data?: unknown }>;
  hasResumedRef: React.MutableRefObject<boolean>;
}

// The answer's turn is dispatched as the POST returns, or when a running turn
// ends; its stream can register a beat after either, so the probe retries.
const FOLLOW_ATTEMPTS = 8;
const FOLLOW_INTERVAL_MS = 500;

// An answered approval card starts the chat's next turn on the server. The page
// resumes a server-started stream once per mount, after hydration; re-arming
// that resume lets the refetched rows land first, so the replay follows them.
export function useFollowBackendTurn({
  status,
  refetchSession,
  hasResumedRef,
}: Args) {
  const pendingRef = useRef(false);
  const isBusy = status === "streaming" || status === "submitted";
  // Read live: the caller holds this hook's function from the render where
  // the card was clicked, and the turn may have ended during the POST.
  const isBusyRef = useRef(isBusy);
  isBusyRef.current = isBusy;
  const isMountedRef = useRef(true);
  // One probe at a time; a second answer mid-probe restarts its budget.
  const attemptsLeftRef = useRef(0);
  const isProbingRef = useRef(false);
  const timerRef = useRef<{ id: number; wake: () => void } | null>(null);
  const followRef = useRef(follow);
  followRef.current = follow;

  async function follow() {
    pendingRef.current = false;
    attemptsLeftRef.current = FOLLOW_ATTEMPTS;
    if (isProbingRef.current) return;
    isProbingRef.current = true;
    try {
      while (attemptsLeftRef.current-- > 0) {
        if (!isMountedRef.current || isBusyRef.current) return;
        hasResumedRef.current = false;
        const result = await refetchSession();
        if (hasActiveBackendStream(result)) return;
        if (attemptsLeftRef.current > 0) await wait(FOLLOW_INTERVAL_MS);
      }
    } finally {
      isProbingRef.current = false;
    }
  }

  function wait(ms: number) {
    return new Promise<void>((wake) => {
      timerRef.current = { id: window.setTimeout(wake, ms), wake };
    });
  }

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (timerRef.current) {
        window.clearTimeout(timerRef.current.id);
        timerRef.current.wake();
      }
    };
  }, []);

  useEffect(() => {
    if (!isBusy && pendingRef.current) followRef.current();
  }, [isBusy]);

  function followBackendTurn() {
    pendingRef.current = true;
    if (!isBusyRef.current) followRef.current();
  }

  return { followBackendTurn };
}
