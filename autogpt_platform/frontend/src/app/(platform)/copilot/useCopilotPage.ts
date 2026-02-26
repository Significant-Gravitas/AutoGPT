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

const STREAM_START_TIMEOUT_MS = 12_000;
const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;
const STALL_DETECTION_MS = 30_000;

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

/**
 * Deduplicate messages by ID, content fingerprint, and toolCallId overlap.
 *
 * Three layers handle different duplicate scenarios:
 * 1. ID-based: same message processed twice (within a single stream).
 * 2. Fingerprint-based: hydration/stream boundary where IDs differ but
 *    the final content is identical.
 * 3. ToolCallId-based: reconnect replay from "0-0" where the SDK assigns
 *    a new nanoid to the replayed message.  During replay the message is
 *    still being built, so text fingerprints don't match — but toolCallIds
 *    are backend-generated UUIDs that stay the same across replays.
 */
function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  const seenToolCallIds = new Set<string>();
  const seenFingerprints = new Set<string>();

  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);

    if (msg.role === "assistant") {
      // Check toolCallId overlap — catches reconnect replay duplicates
      const toolIds = msg.parts
        .filter(
          (p): p is typeof p & { toolCallId: string } =>
            "toolCallId" in p &&
            typeof (p as Record<string, unknown>).toolCallId === "string",
        )
        .map((p) => p.toolCallId);

      if (toolIds.length > 0 && toolIds.some((id) => seenToolCallIds.has(id)))
        return false;
      toolIds.forEach((id) => seenToolCallIds.add(id));

      // Fingerprint dedup for messages without tool calls (pure text)
      const fragments = msg.parts.map((p) => {
        if ("text" in p && typeof p.text === "string") return p.text;
        if ("toolCallId" in p && typeof p.toolCallId === "string")
          return `tool:${p.toolCallId}`;
        return "";
      });
      const fp = `${msg.role}::${fragments.join("\n")}`;
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

  // resumeStreamRef: always points to the latest resumeStream function
  // so the onFinish closure doesn't capture a stale reference.
  const resumeStreamRef = useRef<() => void>(() => {});

  const reconnectingRef = useRef(false);

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

  // Abort the stream if the backend doesn't start sending data within 12s.
  const stopRef = useRef(stop);
  stopRef.current = stop;
  useEffect(() => {
    if (status !== "submitted") return;

    const timer = setTimeout(() => {
      console.warn("[STREAM_DIAG] stream start timeout", {
        sessionId,
        status,
        ts: Date.now(),
      });
      stopRef.current();
      toast({
        title: "Stream timed out",
        description: "The server took too long to respond. Please try again.",
        variant: "destructive",
      });
    }, STREAM_START_TIMEOUT_MS);

    return () => clearTimeout(timer);
  }, [status]);

  // Hydrate messages from the REST session endpoint.
  // Skip hydration while streaming to avoid overwriting the live stream.
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    setMessages((prev) => {
      if (prev.length >= hydratedMessages.length) return prev;
      // Deduplicate to handle rare cases where duplicate streams might occur
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
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
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

  // Stall detection: if streaming but no new messages for 30s, recheck
  // backend state via REST and recover if the stream silently died.
  const lastMessageCountRef = useRef(0);
  const stallTimerRef = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    if (status !== "streaming") {
      clearTimeout(stallTimerRef.current);
      return;
    }

    const currentCount = rawMessages.length;
    if (currentCount !== lastMessageCountRef.current) {
      lastMessageCountRef.current = currentCount;
      clearTimeout(stallTimerRef.current);
    }

    stallTimerRef.current = setTimeout(async () => {
      console.warn("[STREAM_DIAG] STALL DETECTED", {
        sessionId,
        status,
        messageCount: rawMessages.length,
        ts: Date.now(),
      });

      const result = await refetchSession();
      const data = result.data;
      const backendActive = data?.status === 200 && !!data.data.active_stream;

      console.info("[STREAM_DIAG] stall recheck result", {
        sessionId,
        backendActive,
        ts: Date.now(),
      });

      if (!backendActive) {
        // Backend finished but SDK missed it — force stop
        toast({
          title: "Stream ended",
          description: "The response finished. Refreshing...",
        });
        sdkStop();
        setMessages((prev) => resolveInProgressTools(prev, "completed"));
      } else {
        toast({
          title: "Still processing",
          description: "The agent is working on a long task...",
        });
      }
    }, STALL_DETECTION_MS);

    return () => clearTimeout(stallTimerRef.current);
  }, [
    status,
    rawMessages.length,
    sessionId,
    refetchSession,
    sdkStop,
    setMessages,
  ]); // eslint-disable-line react-hooks/exhaustive-deps

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
