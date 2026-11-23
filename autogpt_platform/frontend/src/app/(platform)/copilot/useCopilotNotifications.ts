import { usePostV2MarkSessionRead } from "@/app/api/__generated__/endpoints/chat/chat";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useInsertionEffect, useRef, useState } from "react";
import { COPILOT_COMPLETION_NOTIFICATION } from "./helpers";
import { cachedSessionAttention } from "./sessionGroups";
import { useCopilotUIStore } from "./store";
import { SESSION_LIST_QUERY_KEY } from "./useSessionList";

const NOTIFICATION_SOUND_PATH = "/notification.mp3";
const SEEN_RECHECK_MS = 2_000;
// Without a turn id, the same chat finishing again this soon is taken for the
// same event delivered twice (a WebSocket replay).
const RENOTIFY_AFTER_MS = 10_000;
const NOTIFIED_TURNS_KEPT = 200;
// Failed acknowledgements of one chat before this tab stops re-sending them
// until the user opens a chat again.
const MAX_FAILED_READS = 3;

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
 * Marks the open chat read on the server once the user is here, and on each
 * chat's completion over the WebSocket refreshes the chat list and plays a
 * sound or shows a notification.
 */
export function useCopilotNotifications(activeSessionID: string | null) {
  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const activeSessionRef = useRef(activeSessionID);
  useInsertionEffect(() => {
    activeSessionRef.current = activeSessionID;
  });
  const notifiedAtRef = useRef(new Map<string, number>());
  const notifiedTurnsRef = useRef(new Set<string>());
  const failedReadsRef = useRef({ sessionID: "", count: 0 });
  const readInFlightRef = useRef(false);
  // The open chat while it has a finished turn the user hasn't seen here yet:
  // opening it, a turn ending in it, or a list showing it unread puts it here
  // until they are present.
  const [unseenSessionID, setUnseenSessionID] = useState(activeSessionID);
  const [openedSessionID, setOpenedSessionID] = useState(activeSessionID);
  if (openedSessionID !== activeSessionID) {
    setOpenedSessionID(activeSessionID);
    setUnseenSessionID(activeSessionID);
  }

  const { mutate: markRead } = usePostV2MarkSessionRead({
    mutation: {
      retry: 2,
      onSuccess: () => {
        failedReadsRef.current = { sessionID: "", count: 0 };
      },
      onError: (_error, { sessionId }) => {
        const failed = failedReadsRef.current;
        failedReadsRef.current = {
          sessionID: sessionId,
          count: failed.sessionID === sessionId ? failed.count + 1 : 1,
        };
      },
      // A read that still failed shows the chat unread again, which re-arms
      // it below while the user is here.
      // Held until the lists have re-read the chat, so their stale copy
      // doesn't send the read again.
      onSettled: async () => {
        await queryClient.invalidateQueries({
          queryKey: SESSION_LIST_QUERY_KEY,
        });
        readInFlightRef.current = false;
      },
    },
  });

  // A turn can end without its completion event reaching this tab (the
  // socket was down, or the event was lost), so a list showing the open chat
  // unread arms it too.
  useEffect(() => {
    if (!activeSessionID) return;
    const sessionID = activeSessionID;
    function armIfUnread() {
      if (readInFlightRef.current) return;
      const failed = failedReadsRef.current;
      if (failed.sessionID === sessionID && failed.count >= MAX_FAILED_READS) {
        return;
      }
      const attention = cachedSessionAttention(queryClient, sessionID);
      if (attention === "unread" || attention === "error") {
        setUnseenSessionID(sessionID);
      }
    }
    armIfUnread();
    return queryClient.getQueryCache().subscribe((event) => {
      if (event.type === "updated" && event.action.type === "success") {
        armIfUnread();
      }
    });
  }, [activeSessionID, queryClient]);

  // Seeing the chat marks it read on the server, so its marker clears in every
  // browser and on every device. Only a tab the user is in counts: a tab that
  // merely has the chat open must not mark it seen.
  useEffect(() => {
    if (!unseenSessionID) return;
    const sessionID = unseenSessionID;

    function markIfSeen() {
      if (isUserAway()) return;
      setUnseenSessionID(null);
      readInFlightRef.current = true;
      markRead({ sessionId: sessionID });
    }

    markIfSeen();
    document.addEventListener("visibilitychange", markIfSeen);
    window.addEventListener("focus", markIfSeen);
    // Coming back by clicking straight into an embedded frame (the desktop
    // preview) fires no event in this document, so re-check until it's seen.
    const recheck = window.setInterval(markIfSeen, SEEN_RECHECK_MS);
    return () => {
      document.removeEventListener("visibilitychange", markIfSeen);
      window.removeEventListener("focus", markIfSeen);
      window.clearInterval(recheck);
    };
  }, [unseenSessionID, markRead]);

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

      // The server has already marked the chat unread; refetch so the
      // sidebar shows it now rather than at its next poll.
      queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });

      const isActiveSession = sessionID === activeSessionRef.current;
      const userAway = isUserAway();
      if (isActiveSession) setUnseenSessionID(sessionID);

      // Skip if viewing the active session and it's in focus
      if (isActiveSession && !userAway) return;

      if (isRepeatedCompletion(sessionID, notification)) return;

      // Sound and browser notifications are gated by the user setting
      const state = useCopilotUIStore.getState();
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

    // The same turn's completion delivered twice (a WebSocket replay) sounds
    // once; the chat's next turn sounds again.
    function isRepeatedCompletion(
      sessionID: string,
      notification: WebSocketNotification,
    ) {
      const turnID = (notification as Record<string, unknown>).turn_id;
      if (typeof turnID === "string" && turnID) {
        const key = `${sessionID}:${turnID}`;
        const notified = notifiedTurnsRef.current;
        if (notified.has(key)) return true;
        notified.add(key);
        if (notified.size > NOTIFIED_TURNS_KEPT) {
          notified.delete(notified.values().next().value as string);
        }
        return false;
      }
      const now = Date.now();
      const notifiedAt = notifiedAtRef.current.get(sessionID);
      if (notifiedAt !== undefined && now - notifiedAt < RENOTIFY_AFTER_MS) {
        return true;
      }
      notifiedAtRef.current.set(sessionID, now);
      return false;
    }

    const detach = api.onWebSocketMessage("notification", handleNotification);
    return () => {
      detach();
    };
  }, [api, queryClient]);

  // Completions sent while the socket was down never arrive, so re-read the
  // session list after a reconnect, or it keeps showing those chats busy.
  useEffect(() => {
    return api.onWebSocketReconnect(() => {
      queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
    });
  }, [api, queryClient]);
}
