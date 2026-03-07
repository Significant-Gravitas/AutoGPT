import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "./store";

const ORIGINAL_TITLE = "AutoGPT";
const NOTIFICATION_SOUND_PATH = "/sounds/notification.mp3";

/**
 * Listens for copilot completion notifications via WebSocket.
 * Updates the Zustand store, plays a sound, and updates document.title.
 */
export function useCopilotNotifications(activeSessionID: string | null) {
  const api = useBackendAPI();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const activeSessionRef = useRef(activeSessionID);
  activeSessionRef.current = activeSessionID;

  // Pre-load audio element
  useEffect(() => {
    if (typeof window === "undefined") return;
    const audio = new Audio(NOTIFICATION_SOUND_PATH);
    audio.volume = 0.5;
    audioRef.current = audio;
  }, []);

  // Listen for WebSocket notifications
  useEffect(() => {
    function handleNotification(notification: WebSocketNotification) {
      if (notification.type !== "copilot_completion") return;
      if (notification.event !== "session_completed") return;

      const sessionID = (notification as Record<string, unknown>).session_id;
      if (typeof sessionID !== "string") return;

      const state = useCopilotUIStore.getState();

      // Play sound if enabled
      if (state.isSoundEnabled && audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
      }

      // If the completed session is NOT the active one, track it
      if (sessionID !== activeSessionRef.current) {
        state.addCompletedSession(sessionID);

        // Update document title to show count
        const count = state.completedSessionIDs.size + 1; // +1 for the one we just added
        document.title = `(${count}) Otto is ready - ${ORIGINAL_TITLE}`;
      }
    }

    const detach = api.onWebSocketMessage("notification", handleNotification);
    return () => {
      detach();
    };
  }, [api]);

  // Reset document title when tab gains focus
  useEffect(() => {
    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        document.title = ORIGINAL_TITLE;
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);
}
