import { useMountEffect } from "@/hooks/useMountEffect";
import { useEffect, useRef } from "react";
import { STREAM_STALL_TIMEOUT_MS } from "./copilotStreamFlight";

type UseChatStatus = "submitted" | "streaming" | "ready" | "error";

interface UseStreamActivityWatchdogArgs {
  sessionId: string | null;
  status: UseChatStatus;
  /**
   * A monotonic-ish signal that changes whenever the stream produces any
   * visible content. In practice: rawMessages.length + a hash of the last
   * assistant message's parts. Any change resets the watchdog timer.
   */
  activityToken: string | number;
  /**
   * When the live request last received any bytes. Heartbeats and status
   * updates do not change the messages, so the token above misses them; a
   * stream still receiving bytes is alive however quiet it looks.
   */
  getLastStreamActivityAt: () => number;
  /** True when a reconnect is already scheduled; the watchdog stays quiet. */
  isReconnectScheduled: boolean;
  /** True when the user explicitly stopped; the watchdog stays quiet. */
  isUserStoppingRef: React.MutableRefObject<boolean>;
  /**
   * Ref to the reconnect trigger. Called after ``STREAM_STALL_TIMEOUT_MS``
   * of no activity while the stream is active — this routes through the
   * existing reconnect cascade (toast + exponential backoff + 30 s forced
   * idle), so the UI can never get stuck on a dead stream.
   */
  handleReconnectRef: React.MutableRefObject<() => void>;
}

/**
 * Detect silent SSE stalls.
 *
 * AI SDK only surfaces ``onFinish`` / ``onError`` when the underlying
 * fetch resolves or rejects. If the stream sits open with no bytes at all
 * (backend stuck, half-open connection after a sleep), neither callback
 * fires and the UI stays in ``streaming`` / ``submitted`` state forever.
 * This hook closes the gap by resetting an internal timer on every
 * activity tick and calling ``handleReconnect`` once neither the messages
 * nor the stream's bytes have moved for ``STREAM_STALL_TIMEOUT_MS``.
 */
export function useStreamActivityWatchdog({
  sessionId,
  status,
  activityToken,
  getLastStreamActivityAt,
  isReconnectScheduled,
  isUserStoppingRef,
  handleReconnectRef,
}: UseStreamActivityWatchdogArgs) {
  const getLastStreamActivityAtRef = useRef(getLastStreamActivityAt);
  getLastStreamActivityAtRef.current = getLastStreamActivityAt;
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = undefined;

    if (!sessionId) return;
    if (status !== "streaming" && status !== "submitted") return;
    if (isReconnectScheduled) return;
    if (isUserStoppingRef.current) return;

    function arm(delay: number) {
      timerRef.current = setTimeout(() => {
        if (isUserStoppingRef.current) return;
        const quietFor = Date.now() - getLastStreamActivityAtRef.current();
        if (quietFor < STREAM_STALL_TIMEOUT_MS) {
          arm(STREAM_STALL_TIMEOUT_MS - quietFor);
          return;
        }
        handleReconnectRef.current();
      }, delay);
    }
    arm(STREAM_STALL_TIMEOUT_MS);

    return () => {
      clearTimeout(timerRef.current);
      timerRef.current = undefined;
    };
  }, [
    sessionId,
    status,
    activityToken,
    isReconnectScheduled,
    isUserStoppingRef,
    handleReconnectRef,
  ]);

  useMountEffect(() => {
    return () => {
      clearTimeout(timerRef.current);
      timerRef.current = undefined;
    };
  });
}
