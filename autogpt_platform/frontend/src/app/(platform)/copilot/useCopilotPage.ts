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

    // Only apply fingerprint deduplication to assistant messages
    // User messages should allow duplicates (same text sent multiple times)
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
  });

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

  // When the stream ends (or drops), invalidate the session cache so the
  // next hydration fetches fresh messages from the backend.  Without this,
  // staleTime: Infinity means the cache keeps the pre-stream data forever,
  // and any messages added during streaming are lost on remount/navigation.
  // Track status transitions for cache invalidation and auto-reconnect.
  // Auto-reconnect: GCP's L7 load balancer kills SSE connections at ~5 min.
  // When that happens the AI SDK goes "streaming" → "error".  If the backend
  // executor is still running (hasActiveStream), we call resumeStream() to
  // reconnect via GET and replay from Redis.
  const MAX_RECONNECT_ATTEMPTS = 3;
  const reconnectAttemptsRef = useRef(0);
  const prevStatusRef = useRef(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isIdle = status === "ready" || status === "error";

    // Invalidate session cache when stream ends so hydration fetches fresh data
    if (wasActive && isIdle && sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
    }

    // Auto-reconnect on mid-stream SSE drop
    if (
      prev === "streaming" &&
      status === "error" &&
      sessionId &&
      hasActiveStream
    ) {
      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttemptsRef.current += 1;
        const attempt = reconnectAttemptsRef.current;
        console.info(
          `[copilot] SSE dropped mid-stream, reconnecting (attempt ${attempt}/${MAX_RECONNECT_ATTEMPTS})...`,
        );
        const timer = setTimeout(() => resumeStream(), 1_000);
        return () => clearTimeout(timer);
      } else {
        toast({
          title: "Connection lost",
          description:
            "Could not reconnect to the stream. Please refresh the page.",
          variant: "destructive",
        });
      }
    }

    // Reset reconnect counter when stream completes normally or resumes
    if (status === "ready" || status === "streaming") {
      reconnectAttemptsRef.current = 0;
    }
  }, [status, sessionId, hasActiveStream, queryClient, resumeStream]);

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
    error,
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
