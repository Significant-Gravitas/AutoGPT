import {
  getGetV2GetSessionQueryKey,
  getGetV2ListSessionsQueryKey,
  postV2CancelSessionTask,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import type { UIMessage } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChatSession } from "./useChatSession";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;

/** Mark any in-progress tool parts as completed/errored so spinners stop. */
function resolveInProgressTools(
  messages: UIMessage[],
  outcome: "completed" | "cancelled",
): UIMessage[] {
  return messages.map((msg) => ({
    ...msg,
    parts: msg.parts.map((part) =>
      "state" in part &&
      (part.state === "input-streaming" || part.state === "input-available")
        ? outcome === "cancelled"
          ? { ...part, state: "output-error" as const, errorText: "Cancelled" }
          : { ...part, state: "output-available" as const, output: "" }
        : part,
    ),
  }));
}

/** Build a fingerprint from a message's role + text/tool content for cross-boundary dedup. */
function messageFingerprint(msg: UIMessage): string {
  const fragments = msg.parts.map((p) => {
    if ("text" in p && typeof p.text === "string") return p.text;
    if ("toolCallId" in p && typeof p.toolCallId === "string")
      return `tool:${p.toolCallId}`;
    return "";
  });
  return `${msg.role}::${fragments.join("\n")}`;
}

/**
 * Deduplicate messages by ID *and* by content fingerprint.
 * ID-based dedup catches duplicates within the same source (e.g. two
 * identical stream events).  Fingerprint-based dedup catches duplicates
 * across the hydration/stream boundary where IDs differ (synthetic
 * `${sessionId}-${index}` vs AI SDK nanoid).
 *
 * NOTE: Fingerprint dedup only applies to assistant messages, not user messages.
 * Users should be able to send the same message multiple times.
 */
function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  const seenFingerprints = new Set<string>();
  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);

    if (msg.role === "assistant") {
      const fp = messageFingerprint(msg);
      if (fp !== "::" && seenFingerprints.has(fp)) return false;
      seenFingerprints.add(fp);
    }

    return true;
  });
}

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [sessionToDelete, setSessionToDelete] = useState<{
    id: string;
    title: string | null | undefined;
  } | null>(null);
  const queryClient = useQueryClient();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
  } = useChatSession();

  const { mutate: deleteSessionMutation, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          if (sessionToDelete?.id === sessionId) {
            setSessionId(null);
          }
          setSessionToDelete(null);
        },
        onError: (error) => {
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setSessionToDelete(null);
        },
      },
    });

  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const transport = useMemo(
    () =>
      sessionId
        ? new DefaultChatTransport({
            api: `/api/chat/sessions/${sessionId}/stream`,
            prepareSendMessagesRequest: ({ messages }) => {
              const last = messages[messages.length - 1];
              return {
                body: {
                  message: (
                    last.parts?.map((p) => (p.type === "text" ? p.text : "")) ??
                    []
                  ).join(""),
                  is_user_message: last.role === "user",
                  context: null,
                },
              };
            },
            // Resume: GET goes to the same URL as POST (backend uses
            // method to distinguish).  Override the default formula which
            // would append /{chatId}/stream to the existing path.
            prepareReconnectToStreamRequest: () => ({
              api: `/api/chat/sessions/${sessionId}/stream`,
            }),
          })
        : null,
    [sessionId],
  );

  // Track reconnect attempts and timer for exponential backoff
  const reconnectAttemptsRef = useRef<Map<string, number>>(new Map());
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();

  // resumeStreamRef / setMessagesRef: always point to the latest functions
  // so the scheduleReconnect closure doesn't capture stale references.
  const resumeStreamRef = useRef<() => void>(() => {});
  const setMessagesRef = useRef<
    (fn: (prev: UIMessage[]) => UIMessage[]) => void
  >(() => {});

  const reconnectingRef = useRef(false);
  const shouldClearOnNextMessageRef = useRef(false);

  // Shared reconnect logic used by both onFinish (isDisconnect) and onError
  // (network abort where flush() never runs).
  function scheduleReconnect(sid: string) {
    if (reconnectingRef.current) return;
    reconnectingRef.current = true;

    const attempts = reconnectAttemptsRef.current.get(sid) ?? 0;
    reconnectAttemptsRef.current.set(sid, attempts + 1);

    const delay = Math.min(
      RECONNECT_BASE_DELAY_MS * 2 ** attempts,
      RECONNECT_MAX_DELAY_MS,
    );

    if (attempts === 0) {
      toast({
        title: "Connection lost",
        description: "Reconnecting...",
      });
    }

    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = undefined;
      reconnectingRef.current = false;
      console.info("[STREAM_DIAG] resumeStream fired", {
        sessionId: sid,
        attempt: attempts + 1,
        delay,
        ts: Date.now(),
      });
      // Mark that we should clear assistant messages once new ones arrive
      shouldClearOnNextMessageRef.current = true;
      resumeStreamRef.current();
    }, delay);
  }

  const {
    messages: rawMessages,
    sendMessage,
    stop: sdkStop,
    status,
    error,
    setMessages,
    resumeStream,
  } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
    // Don't use resume: true — it fires before hydration completes, causing
    // the hydrated messages to overwrite the resumed stream.  Instead we
    // call resumeStream() manually after hydration + active_stream detection.
    onFinish: async ({ isDisconnect, isAbort }) => {
      console.info("[STREAM_DIAG] onFinish", {
        sessionId,
        isDisconnect,
        isAbort,
        status,
        attempt: reconnectAttemptsRef.current.get(sessionId ?? "") ?? 0,
        ts: Date.now(),
      });
      if (isAbort || !sessionId) return;

      if (isDisconnect) {
        scheduleReconnect(sessionId);
        return;
      }

      // isDisconnect=false means the stream closed cleanly. But a proxy
      // timeout (e.g. Vercel maxDuration) also closes cleanly — the
      // function's finally block sends [DONE] before exiting. Check if the
      // backend executor is actually still running.
      const result = await refetchSession();
      const backendActive =
        result.data?.status === 200 && !!result.data.data.active_stream;

      console.info("[STREAM_DIAG] onFinish backend check", {
        sessionId,
        backendActive,
        ts: Date.now(),
      });

      if (backendActive) {
        scheduleReconnect(sessionId);
      }
    },
    onError: (error) => {
      console.warn("[STREAM_DIAG] onError", {
        sessionId,
        name: error.name,
        error: error.message,
        status,
        ts: Date.now(),
      });
      if (!sessionId) return;
      // Only reconnect on network-level disconnects (proxy/LB killing TCP).
      // TypeError = fetch body stream interrupted; AbortError = signal aborted.
      // HTTP errors from the backend should not trigger reconnect.
      const isNetworkError =
        error.name === "TypeError" || error.name === "AbortError";
      if (isNetworkError) {
        scheduleReconnect(sessionId);
      }
    },
  });
  resumeStreamRef.current = resumeStream;
  setMessagesRef.current = setMessages;

  // Clear assistant messages once new messages arrive after reconnect
  useEffect(() => {
    if (shouldClearOnNextMessageRef.current && rawMessages.length > 0) {
      shouldClearOnNextMessageRef.current = false;
      // Drop old assistant messages now that new ones are arriving
      setMessages((prev) => prev.filter((m) => m.role !== "assistant"));
    }
  }, [rawMessages.length, setMessages]);

  // Deduplicate messages continuously to prevent duplicates when resuming streams
  const messages = useMemo(
    () => deduplicateMessages(rawMessages),
    [rawMessages],
  );

  // Wrap AI SDK's stop() to also cancel the backend executor task.
  // sdkStop() aborts the SSE fetch instantly (UI feedback), then we fire
  // the cancel API to actually stop the executor and wait for confirmation.
  async function stop() {
    sdkStop();
    setMessages((prev) => resolveInProgressTools(prev, "cancelled"));

    if (!sessionId) return;
    try {
      const res = await postV2CancelSessionTask(sessionId);
      if (
        res.status === 200 &&
        "reason" in res.data &&
        res.data.reason === "cancel_published_not_confirmed"
      ) {
        toast({
          title: "Stop may take a moment",
          description:
            "The cancel was sent but not yet confirmed. The task should stop shortly.",
        });
      }
    } catch {
      toast({
        title: "Could not stop the task",
        description: "The task may still be running in the background.",
        variant: "destructive",
      });
    }
  }

  // Hydrate messages from the REST session endpoint.
  // Skip hydration while streaming or during reconnect to avoid overwriting
  // the SDK's in-flight messages (which would cause duplicates on resume).
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    if (reconnectingRef.current || reconnectTimerRef.current) return;
    setMessages((prev) => {
      if (prev.length >= hydratedMessages.length) return prev;
      return deduplicateMessages(hydratedMessages);
    });
  }, [hydratedMessages, setMessages, status]);

  // Ref: tracks whether we've already resumed for a given session.
  // Format: Map<sessionId, hasResumed>
  const hasResumedRef = useRef<Map<string, boolean>>(new Map());

  // Clean up reconnect and status state on session switch.
  const prevStatusRef = useRef(status);
  useEffect(() => {
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = undefined;
    reconnectingRef.current = false;
    reconnectAttemptsRef.current.delete(sessionId ?? "");
    prevStatusRef.current = status;
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // When the stream ends (or drops), invalidate the session cache so the
  // next hydration fetches fresh messages from the backend.
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    if (prev !== status) {
      console.info("[STREAM_DIAG] status transition", {
        prev,
        next: status,
        sessionId,
        messageCount: rawMessages.length,
        ts: Date.now(),
      });
    }

    const wasActive = prev === "streaming" || prev === "submitted";
    const isIdle = status === "ready" || status === "error";
    if (wasActive && isIdle && sessionId) {
      // Don't invalidate during reconnect — re-hydration would clobber
      // the SDK's in-flight messages, causing duplicates on resume.
      if (!reconnectingRef.current && !reconnectTimerRef.current) {
        queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(sessionId),
        });
      }
      if (status === "ready") {
        reconnectAttemptsRef.current.delete(sessionId);
      }
    }
  }, [status, sessionId, queryClient]);

  // Resume an active stream AFTER hydration completes.
  // IMPORTANT: Only runs when page loads with existing active stream (reconnection).
  // Does NOT run when new streams start during active conversation.
  useEffect(() => {
    if (!sessionId) return;
    if (!hasActiveStream) return;
    if (!hydratedMessages || hydratedMessages.length === 0) return;

    // Never resume if currently streaming
    if (status === "streaming" || status === "submitted") return;

    // Only resume once per session
    if (hasResumedRef.current.get(sessionId)) return;

    // Mark as resumed immediately to prevent race conditions
    hasResumedRef.current.set(sessionId, true);
    resumeStream();
  }, [sessionId, hasActiveStream, hydratedMessages, status, resumeStream]);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

  useEffect(() => {
    if (!sessionId || !pendingMessage) return;
    const msg = pendingMessage;
    setPendingMessage(null);
    sendMessage({ text: msg });
  }, [sessionId, pendingMessage, sendMessage]);

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;

    if (sessionId) {
      sendMessage({ text: trimmed });
      return;
    }

    setPendingMessage(trimmed);
    await createSession();
  }

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(
      { limit: 50 },
      { query: { enabled: !isUserLoading && isLoggedIn } },
    );

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  function handleOpenDrawer() {
    setIsDrawerOpen(true);
  }

  function handleCloseDrawer() {
    setIsDrawerOpen(false);
  }

  function handleDrawerOpenChange(open: boolean) {
    setIsDrawerOpen(open);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
    if (isMobile) setIsDrawerOpen(false);
  }

  function handleNewChat() {
    setSessionId(null);
    if (isMobile) setIsDrawerOpen(false);
  }

  const handleDeleteClick = useCallback(
    (id: string, title: string | null | undefined) => {
      if (isDeleting) return;
      setSessionToDelete({ id, title });
    },
    [isDeleting],
  );

  const handleConfirmDelete = useCallback(() => {
    if (sessionToDelete) {
      deleteSessionMutation({ sessionId: sessionToDelete.id });
    }
  }, [sessionToDelete, deleteSessionMutation]);

  const handleCancelDelete = useCallback(() => {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }, [isDeleting]);

  // True while we know the backend has an active stream but haven't
  // reconnected yet.  Used to disable the send button and show stop UI.
  const isReconnecting =
    hasActiveStream && status !== "streaming" && status !== "submitted";

  return {
    sessionId,
    messages,
    status,
    error: reconnectingRef.current ? undefined : error,
    stop,
    isReconnecting,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUserLoading,
    isLoggedIn,
    createSession,
    onSend,
    // Mobile drawer
    isMobile,
    isDrawerOpen,
    sessions,
    isLoadingSessions,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleSelectSession,
    handleNewChat,
    // Delete functionality
    sessionToDelete,
    isDeleting,
    handleDeleteClick,
    handleConfirmDelete,
    handleCancelDelete,
  };
}
