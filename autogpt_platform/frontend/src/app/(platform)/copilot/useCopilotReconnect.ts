import { toast } from "@/components/molecules/Toast/use-toast";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useEffect, useRef, useState } from "react";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_ATTEMPTS = 3;

/** Max time (ms) the UI can stay in "reconnecting" state before forcing idle. */
const RECONNECT_MAX_DURATION_MS = 30_000;

type UseChatStatus = "submitted" | "streaming" | "ready" | "error";

interface UseCopilotReconnectArgs {
  sessionId: string | null;
  hasActiveStream: boolean;
  /**
   * Current `useChat` status — used to clear the forced-idle timer the moment
   * the stream is back live, and to reset the reconnect counters once a turn
   * completes cleanly.
   */
  status: UseChatStatus;
  /** True once this mount has received meaningful replay/live content. */
  hasConnectedThisMountRef: React.MutableRefObject<boolean>;
  /**
   * Ref holding `useChat`'s latest `resumeStream`. Kept in a ref so the
   * timer callback always invokes the current value without stale closures.
   */
  resumeStreamRef: React.MutableRefObject<() => void>;
  /**
   * Flipped to `true` when a reconnect actually fires, so a deferred
   * page-load resume (queued in `pendingResumeRef` while hydration was in
   * flight) can't double-fire `resumeStream()` once hydration completes.
   */
  hasResumedRef: React.MutableRefObject<boolean>;
}

/**
 * Owns the reconnect lifecycle for `useCopilotStream`:
 *   - exponential backoff with 3-attempt cap + terminal toast
 *   - a concurrent 30 s forced-idle timer so the UI can never get stuck
 *     "reconnecting" if the backend is slow to clear `active_stream`
 *   - self-clears the forced-idle timer the moment status returns to active
 *   - self-resets counters + exhausted flag when a turn completes cleanly
 *   - self-clears armed timers on unmount so they can't toast into the void
 *
 * All state here is session-scoped by lifetime — the parent subtree is
 * keyed `key={sessionId}` by `CopilotPage`, so a session switch remounts
 * this hook and every ref resets naturally.
 */
export function useCopilotReconnect({
  sessionId,
  hasActiveStream,
  status,
  hasConnectedThisMountRef,
  resumeStreamRef,
  hasResumedRef,
}: UseCopilotReconnectArgs) {
  const [isReconnectScheduled, setIsReconnectScheduled] = useState(false);
  const [reconnectExhausted, setReconnectExhausted] = useState(false);

  const reconnectScheduledRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectStartedAtRef = useRef<number | null>(null);
  const hasShownDisconnectToastRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectTimeoutTimerRef = useRef<ReturnType<typeof setTimeout>>();

  function handleReconnect() {
    if (!sessionId) return;
    if (reconnectScheduledRef.current) return;

    const nextAttempt = reconnectAttemptsRef.current + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      setReconnectExhausted(true);
      toast({
        title: "Connection lost",
        description: "Unable to reconnect. Please refresh the page.",
        variant: "destructive",
      });
      return;
    }

    // Arm the forced-timeout the first time a reconnect starts in this cycle.
    if (reconnectStartedAtRef.current === null) {
      reconnectStartedAtRef.current = Date.now();
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = setTimeout(() => {
        // Cancel the pending reconnect timer so it can't fire resumeStream()
        // after the UI has been forced to idle — otherwise we'd end up in an
        // inconsistent state (reconnectExhausted=true + a fresh stream).
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = undefined;
        reconnectScheduledRef.current = false;
        reconnectStartedAtRef.current = null;
        setIsReconnectScheduled(false);
        setReconnectExhausted(true);
        toast({
          title: "Connection timed out",
          description:
            "AutoPilot may still be working. Refresh to check for updates.",
          variant: "destructive",
        });
      }, RECONNECT_MAX_DURATION_MS);
    }

    reconnectScheduledRef.current = true;
    reconnectAttemptsRef.current = nextAttempt;
    setIsReconnectScheduled(true);

    if (!hasShownDisconnectToastRef.current) {
      hasShownDisconnectToastRef.current = true;
      toast({ title: "Connection lost", description: "Reconnecting..." });
    }

    const delay = RECONNECT_BASE_DELAY_MS * 2 ** (nextAttempt - 1);
    reconnectTimerRef.current = setTimeout(() => {
      reconnectScheduledRef.current = false;
      // Mark resumed so a deferred page-load resume (queued in
      // pendingResumeRef while hydration was in flight) can't double-fire.
      hasResumedRef.current = true;
      setIsReconnectScheduled(false);
      resumeStreamRef.current();
    }, delay);
  }

  // Auto-clear the forced-idle timer the moment the stream is back live,
  // and reset counters once a turn completes cleanly.
  useEffect(() => {
    if (!sessionId) return;
    const isNowActive = status === "streaming" || status === "submitted";

    if (
      isNowActive &&
      reconnectStartedAtRef.current !== null &&
      hasConnectedThisMountRef.current
    ) {
      reconnectStartedAtRef.current = null;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
    }

    if (status === "ready" && !reconnectScheduledRef.current) {
      const stillRestoringWithoutConnection =
        hasActiveStream && !hasConnectedThisMountRef.current;
      if (stillRestoringWithoutConnection) return;
      reconnectAttemptsRef.current = 0;
      hasShownDisconnectToastRef.current = false;
      reconnectStartedAtRef.current = null;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      setReconnectExhausted(false);
    }
  }, [status, sessionId, hasActiveStream, hasConnectedThisMountRef]);

  // Clear armed timers on unmount so they can't toast into the void after
  // the host is gone.
  useMountEffect(() => {
    return () => {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = undefined;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
    };
  });

  return {
    isReconnectScheduled,
    reconnectExhausted,
    reconnectScheduledRef,
    handleReconnect,
  };
}
