import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { Key } from "@/services/storage/local-storage";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "./store";

const ORIGINAL_TITLE = "AutoGPT";
const NOTIFICATION_SOUND_PATH = "/sounds/notification.mp3";

/**
 * Show a browser notification with click-to-navigate behaviour.
 * Wrapped in try-catch so it degrades gracefully in service-worker or
 * other restricted contexts where the Notification constructor throws.
 */
function showBrowserNotification(
  title: string,
  opts: { body: string; icon: string; sessionID: string },
) {
  try {
    const n = new Notification(title, { body: opts.body, icon: opts.icon });
    n.onclick = () => {
      window.focus();
      const url = new URL(window.location.href);
      url.searchParams.set("sessionId", opts.sessionID);
      window.history.pushState({}, "", url.toString());
      window.dispatchEvent(new PopStateEvent("popstate"));
      n.close();
    };
  } catch {
    // Notification constructor is unavailable (e.g. service-worker context).
    // The user will still see the in-app badge and title update.
  }
}

/**
 * Listens for copilot completion notifications via WebSocket.
 * Updates the Zustand store, plays a sound, and updates document.title.
 */
export function useCopilotNotifications(activeSessionID: string | null) {
  const api = useBackendAPI();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const activeSessionRef = useRef(activeSessionID);
  activeSessionRef.current = activeSessionID;
  const windowFocusedRef = useRef(true);

  // Pre-load audio element and sync document title with persisted state
  useEffect(() => {
    if (typeof window === "undefined") return;
    const audio = new Audio(NOTIFICATION_SOUND_PATH);
    audio.volume = 0.5;
    audioRef.current = audio;

    const count = useCopilotUIStore.getState().completedSessionIDs.size;
    if (count > 0) {
      document.title = `(${count}) AutoPilot is ready - ${ORIGINAL_TITLE}`;
    }
  }, []);

  // Listen for WebSocket notifications
  useEffect(() => {
    function handleNotification(notification: WebSocketNotification) {
      if (notification.type !== "copilot_completion") return;
      if (notification.event !== "session_completed") return;

      const sessionID = (notification as Record<string, unknown>).session_id;
      if (typeof sessionID !== "string") return;

      const state = useCopilotUIStore.getState();

      const isActiveSession = sessionID === activeSessionRef.current;
      const isUserAway =
        document.visibilityState === "hidden" || !windowFocusedRef.current;

      // Skip if viewing the active session and it's in focus
      if (isActiveSession && !isUserAway) return;

      // Skip if we already notified for this session (e.g. WS replay)
      if (state.completedSessionIDs.has(sessionID)) return;

      // Always update UI state (checkmark + title) regardless of notification setting
      state.addCompletedSession(sessionID);
      const count = useCopilotUIStore.getState().completedSessionIDs.size;
      document.title = `(${count}) AutoPilot is ready - ${ORIGINAL_TITLE}`;

      // Sound and browser notifications are gated by the user setting
      if (!state.isNotificationsEnabled) return;

      if (state.isSoundEnabled && audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
      }

      // Send browser notification when user is away
      if (
        typeof Notification !== "undefined" &&
        Notification.permission === "granted" &&
        isUserAway
      ) {
        showBrowserNotification("AutoPilot is ready", {
          body: "A response is waiting for you.",
          icon: "/favicon.ico",
          sessionID,
        });
      }
    }

    const detach = api.onWebSocketMessage("notification", handleNotification);
    return () => {
      detach();
    };
  }, [api]);

  // Track window focus for browser notifications when app is in background
  useEffect(() => {
    function handleFocus() {
      windowFocusedRef.current = true;
      if (useCopilotUIStore.getState().completedSessionIDs.size === 0) {
        document.title = ORIGINAL_TITLE;
      }
    }
    function handleBlur() {
      windowFocusedRef.current = false;
    }
    function handleVisibilityChange() {
      if (
        document.visibilityState === "visible" &&
        useCopilotUIStore.getState().completedSessionIDs.size === 0
      ) {
        document.title = ORIGINAL_TITLE;
      }
    }

    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  // Sync completedSessionIDs across tabs via localStorage storage events
  useEffect(() => {
    function handleStorage(e: StorageEvent) {
      if (e.key !== Key.COPILOT_COMPLETED_SESSIONS) return;
      let next: Set<string>;
      try {
        if (!e.newValue) {
          next = new Set<string>();
        } else {
          const parsed: unknown = JSON.parse(e.newValue);
          next = Array.isArray(parsed)
            ? new Set<string>(parsed.filter((v) => typeof v === "string"))
            : new Set<string>();
        }
      } catch {
        next = new Set<string>();
      }
      // localStorage is the shared source of truth — adopt it directly so both
      // additions (new completions) and removals (cleared sessions) propagate.
      useCopilotUIStore.setState({ completedSessionIDs: next });
      const count = next.size;
      document.title =
        count > 0
          ? `(${count}) AutoPilot is ready - ${ORIGINAL_TITLE}`
          : ORIGINAL_TITLE;
    }
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);
}
