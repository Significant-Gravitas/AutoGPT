import { useEffect, useRef } from "react";
import { useScheduledSendHandoff } from "./scheduledSendHandoff";

/** Sets the send a new-chat screen handed this session when it created it. */
export function useHandedOffSend(
  sessionId: string | null,
  scheduleQuietly: (message: string) => Promise<void>,
) {
  const waiting = useScheduledSendHandoff((state) =>
    sessionId ? (state.messages[sessionId] ?? null) : null,
  );
  const latest = useRef(scheduleQuietly);
  latest.current = scheduleQuietly;
  useEffect(() => {
    if (!sessionId || waiting === null) return;
    const message = useScheduledSendHandoff.getState().take(sessionId);
    if (message !== null) void latest.current(message);
  }, [sessionId, waiting]);
}
