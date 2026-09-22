import { useMountEffect } from "@/hooks/useMountEffect";
import { useEffect, useRef } from "react";

/**
 * Number of ms the stream can sit in an active state with no message/part
 * activity before the watchdog assumes the SSE has silently stalled and
 * triggers a reconnect. Tuned to be long enough that legitimate quiet
 * gaps (tool calls that don't emit progress, slow model turns) don't fire
 * false positives, short enough that the user isn't staring at a dead
 * chat for minutes.
 */
const STREAM_STALL_TIMEOUT_MS = 60_000;

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
 * fetch resolves or rejects. If the stream sits open with heartbeats but
 * no data events (backend stuck, Redis zombie, etc.), neither callback
 * fires and the UI stays in ``streaming`` / ``submitted`` state forever.
 * This hook closes the gap by resetting an internal timer on every
 * activity tick and calling ``handleReconnect`` when the timer elapses.
 */
export function useStreamActivityWatchdog({
  sessionId,
  status,
  activityToken,
  isReconnectScheduled,
  isUserStoppingRef,
  handleReconnectRef,
}: UseStreamActivityWatchdogArgs) {
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = undefined;

    if (!sessionId) return;
    if (status !== "streaming" && status !== "submitted") return;
    if (isReconnectScheduled) return;
    if (isUserStoppingRef.current) return;

    timerRef.current = setTimeout(() => {
      if (isUserStoppingRef.current) return;
      handleReconnectRef.current();
    }, STREAM_STALL_TIMEOUT_MS);

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
