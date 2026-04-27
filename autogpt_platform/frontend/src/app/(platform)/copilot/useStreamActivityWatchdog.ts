import { useEffect, useRef } from "react";

const STREAM_STALL_TIMEOUT_MS = 60_000;

type UseChatStatus = "submitted" | "streaming" | "ready" | "error";

interface UseStreamActivityWatchdogArgs {
  sessionId: string | null;
  status: UseChatStatus;
  activityToken: string | number;
  isReconnectScheduled: boolean;
  isUserStoppingRef: React.MutableRefObject<boolean>;
  handleReconnectRef: React.MutableRefObject<(sid: string) => void>;
}

export function useStreamActivityWatchdog({
  sessionId,
  status,
  activityToken,
  isReconnectScheduled,
  isUserStoppingRef,
  handleReconnectRef,
}: UseStreamActivityWatchdogArgs) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = undefined;

    if (!sessionId) return;
    if (status !== "streaming" && status !== "submitted") return;
    if (isReconnectScheduled) return;
    if (isUserStoppingRef.current) return;

    const sid = sessionId;
    timerRef.current = setTimeout(() => {
      if (isUserStoppingRef.current) return;
      handleReconnectRef.current(sid);
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
}
