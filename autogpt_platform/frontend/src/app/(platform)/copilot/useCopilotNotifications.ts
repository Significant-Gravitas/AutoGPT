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

      // Skip all notifications if disabled
      if (!state.isNotificationsEnabled) return;

      // Only notify for background sessions
      if (sessionID === activeSessionRef.current) return;

      // Skip if we already notified for this session (e.g. WS replay)
      if (state.completedSessionIDs.has(sessionID)) return;

      // Play sound if enabled
      if (state.isSoundEnabled && audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
      }

      state.addCompletedSession(sessionID);

      // Update document title to show count (read fresh state after add)
      const count = useCopilotUIStore.getState().completedSessionIDs.size;
      document.title = `(${count}) Otto is ready - ${ORIGINAL_TITLE}`;

      // Send browser notification if permitted
      if (
        typeof Notification !== "undefined" &&
        Notification.permission === "granted" &&
        document.visibilityState === "hidden"
      ) {
        const n = new Notification("Otto is ready", {
          body: "A response is waiting for you.",
          icon: "/favicon.ico",
        });
        n.onclick = () => {
          window.focus();
          window.location.href = `/copilot?sessionId=${sessionID}`;
          n.close();
        };
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
      if (
        document.visibilityState === "visible" &&
        useCopilotUIStore.getState().completedSessionIDs.size === 0
      ) {
        document.title = ORIGINAL_TITLE;
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);
}
