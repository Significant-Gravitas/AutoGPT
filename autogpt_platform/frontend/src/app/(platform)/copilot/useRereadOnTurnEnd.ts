import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { useEffect } from "react";

import type { TurnRuntime } from "./stream/turnRuntime";

/**
 * The chat can run turns in another tab or on another device, which this
 * tab never streams. The server says over the live connection when any turn
 * in the chat ends, so an idle chat reads itself again then, and once that
 * connection is back, since an end said while it was down is not said again.
 * It only listens: something else in the app has to open the live
 * connection, and without it the push never comes.
 */
export function useRereadOnTurnEnd(runtime: TurnRuntime | null) {
  const api = useBackendAPI();

  useEffect(() => {
    if (!runtime) return;
    const chat = runtime;
    function onNotification(notification: WebSocketNotification) {
      if (endsTurnIn(notification, chat.sessionId)) chat.rereadIfIdle();
    }
    const detachMessage = api.onWebSocketMessage(
      "notification",
      onNotification,
    );
    const detachReconnect = api.onWebSocketReconnect(() => chat.rereadIfIdle());
    return () => {
      detachMessage();
      detachReconnect();
    };
  }, [api, runtime]);
}

function endsTurnIn(notification: WebSocketNotification, sessionId: string) {
  return (
    notification.type === "copilot_completion" &&
    notification.event === "session_completed" &&
    (notification as Record<string, unknown>).session_id === sessionId
  );
}
