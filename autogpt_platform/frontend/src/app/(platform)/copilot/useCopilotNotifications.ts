import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { Key } from "@/services/storage/local-storage";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useInsertionEffect, useRef } from "react";
import { COPILOT_COMPLETION_NOTIFICATION, parseSessionIDs } from "./helpers";
import { useCopilotUIStore } from "./store";
import { SESSION_LIST_QUERY_KEY } from "./useSessionList";

const NOTIFICATION_SOUND_PATH = "/notification.mp3";
const SEEN_RECHECK_MS = 2_000;

/**
 * Show a browser notification with click-to-navigate behaviour.
 * If a push service worker subscription is active, the SW handles
 * notifications via the Push API — skip the in-page Notification to
 * avoid duplicates.
 */
async function showBrowserNotification(
  title: string,
  opts: { body: string; icon: string; sessionID: string },
) {
  try {
    // If the push SW is handling notifications, skip the in-page fallback
    const swReg = await navigator.serviceWorker?.getRegistration?.("/");
    if (swReg) {
      const pushSub = await swReg.pushManager?.getSubscription?.();
      if (pushSub) return;
    }
  } catch {
    // serviceWorker API unavailable — fall through to Notification()
  }

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

// The user is here while the tab is visible and the document has focus. That
// stays true while they work inside an embedded iframe, where a window blur
// alone would say they had left. The handler and the clear effect share it so
// they never disagree about the same moment.
function isUserAway() {
  return document.visibilityState === "hidden" || !document.hasFocus();
}

/**
 * Listens for copilot completion notifications via WebSocket.
 * Updates the Zustand store and plays a sound.
 */
export function useCopilotNotifications(activeSessionID: string | null) {
  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const activeSessionRef = useRef(activeSessionID);
  useInsertionEffect(() => {
    activeSessionRef.current = activeSessionID;
  });
  const completedSessionIDs = useCopilotUIStore((s) => s.completedSessionIDs);

  // Seeing a finished chat takes it off the list. The handler below skips
  // chats already on it, so a chat left there never notifies again. Only a
  // tab the user is in counts: the list is shared across tabs, so a tab that
  // merely has the chat open must not mark it seen for the others.
  useEffect(() => {
    function clearIfSeen() {
      if (isUserAway()) return;
      if (!activeSessionID || !completedSessionIDs.has(activeSessionID)) return;
      useCopilotUIStore.getState().clearCompletedSession(activeSessionID);
    }

    clearIfSeen();
    document.addEventListener("visibilitychange", clearIfSeen);
    window.addEventListener("focus", clearIfSeen);
    // Coming back by clicking straight into an embedded frame (the desktop
    // preview) fires no event in this document, so re-check while the open
    // chat is still on the list. Clearing it ends the interval.
    const recheck =
      activeSessionID && completedSessionIDs.has(activeSessionID)
        ? window.setInterval(clearIfSeen, SEEN_RECHECK_MS)
        : undefined;
    return () => {
      document.removeEventListener("visibilitychange", clearIfSeen);
      window.removeEventListener("focus", clearIfSeen);
      window.clearInterval(recheck);
    };
  }, [activeSessionID, completedSessionIDs]);

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

      const isActiveSession = sessionID === activeSessionRef.current;
      const userAway = isUserAway();

      // Skip if viewing the active session and it's in focus
      if (isActiveSession && !userAway) return;

      // Skip if we already notified for this session (e.g. WS replay)
      if (state.completedSessionIDs.has(sessionID)) return;

      // Always update UI state (checkmark + title) regardless of notification setting
      state.addCompletedSession(sessionID);

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
        userAway
      ) {
        showBrowserNotification(COPILOT_COMPLETION_NOTIFICATION.title, {
          body: COPILOT_COMPLETION_NOTIFICATION.body,
          icon: COPILOT_COMPLETION_NOTIFICATION.icon,
          sessionID,
        });
      }
    }

    const detach = api.onWebSocketMessage("notification", handleNotification);
    return () => {
      detach();
    };
  }, [api]);

  // Sync completedSessionIDs across tabs via localStorage storage events
  useEffect(() => {
    function handleStorage(e: StorageEvent) {
      if (e.key !== Key.COPILOT_COMPLETED_SESSIONS) return;
      // localStorage is the shared source of truth — adopt it directly so both
      // additions (new completions) and removals (cleared sessions) propagate.
      const next = parseSessionIDs(e.newValue);
      useCopilotUIStore.setState({ completedSessionIDs: next });

      // Refetch the session list so the sidebar reflects the latest
      // is_processing state (avoids stale spinner after cross-tab clear).
      queryClient.invalidateQueries({
        queryKey: SESSION_LIST_QUERY_KEY,
      });
    }
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, [queryClient]);
}
