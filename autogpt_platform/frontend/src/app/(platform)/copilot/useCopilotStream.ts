import {
  getGetV2GetSessionQueryKey,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import type { UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { deduplicateMessages, resolveInProgressTools } from "./helpers";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_ATTEMPTS = 3;

/** Fetch a fresh JWT for direct backend requests (same pattern as WebSocket). */
async function getAuthHeaders(): Promise<Record<string, string>> {
  const { token, error } = await getWebSocketToken();
  if (error || !token) {
    console.warn("[Copilot] Failed to get auth token:", error);
    throw new Error("Authentication failed — please sign in again.");
  }
  return { Authorization: `Bearer ${token}` };
}

interface UseCopilotStreamArgs {
  sessionId: string | null;
  hydratedMessages: UIMessage[] | undefined;
  hasActiveStream: boolean;
  refetchSession: () => Promise<{ data?: unknown }>;
}

export function useCopilotStream({
  sessionId,
  hydratedMessages,
  hasActiveStream,
  refetchSession,
}: UseCopilotStreamArgs) {
  const queryClient = useQueryClient();

  // Connect directly to the Python backend for SSE, bypassing the Next.js
  // serverless proxy. This eliminates the Vercel 800s function timeout that
  // was the primary cause of stream disconnections on long-running tasks.
  // Auth uses the same server-action token pattern as the WebSocket connection.
  const transport = useMemo(
    () =>
      sessionId
        ? new DefaultChatTransport({
            api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
            prepareSendMessagesRequest: async ({ messages }) => {
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
                headers: await getAuthHeaders(),
              };
            },
            prepareReconnectToStreamRequest: async () => ({
              api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
              headers: await getAuthHeaders(),
            }),
          })
        : null,
    [sessionId],
  );

  // Reconnect state — use refs for values read inside callbacks to avoid
  // stale closures when multiple reconnect cycles fire in quick succession.
  const reconnectAttemptsRef = useRef(0);
  const isReconnectScheduledRef = useRef(false);
  const [isReconnectScheduled, setIsReconnectScheduled] = useState(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const hasShownDisconnectToast = useRef(false);
  // Set when the user explicitly clicks stop — prevents onError from
  // triggering a reconnect cycle for the resulting AbortError.
  const isUserStoppingRef = useRef(false);

  function handleReconnect(sid: string) {
    if (isReconnectScheduledRef.current || !sid) return;

    const nextAttempt = reconnectAttemptsRef.current + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      toast({
        title: "Connection lost",
        description: "Unable to reconnect. Please refresh the page.",
        variant: "destructive",
      });
      return;
    }

    isReconnectScheduledRef.current = true;
    setIsReconnectScheduled(true);
    reconnectAttemptsRef.current = nextAttempt;

    if (!hasShownDisconnectToast.current) {
      hasShownDisconnectToast.current = true;
      toast({
        title: "Connection lost",
        description: "Reconnecting...",
      });
    }

    const delay = RECONNECT_BASE_DELAY_MS * 2 ** (nextAttempt - 1);

    reconnectTimerRef.current = setTimeout(() => {
      isReconnectScheduledRef.current = false;
      setIsReconnectScheduled(false);
      resumeStream();
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
    onFinish: async ({ isDisconnect, isAbort }) => {
      if (isAbort || !sessionId) return;

      if (isDisconnect) {
        handleReconnect(sessionId);
        return;
      }

      // Check if backend executor is still running after clean close
      const result = await refetchSession();
      const d = result.data as
        | { status: number; data: { active_stream?: unknown } }
        | undefined;
      const backendActive = d?.status === 200 && !!d.data.active_stream;

      if (backendActive) {
        handleReconnect(sessionId);
      }
    },
    onError: (error) => {
      if (!sessionId) return;

      // Detect authentication failures (from getAuthHeaders or 401 responses)
      const isAuthError =
        error.message.includes("Authentication failed") ||
        error.message.includes("Unauthorized") ||
        error.message.includes("Not authenticated") ||
        error.message.toLowerCase().includes("401");
      if (isAuthError) {
        toast({
          title: "Authentication error",
          description: "Your session may have expired. Please sign in again.",
          variant: "destructive",
        });
        return;
      }

      // Only reconnect on network errors (not HTTP errors), and never
      // reconnect when the user explicitly stopped the stream.
      if (isUserStoppingRef.current) return;
      const isNetworkError =
        error.name === "TypeError" || error.name === "AbortError";
      if (isNetworkError) {
        handleReconnect(sessionId);
      }
    },
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
    isUserStoppingRef.current = true;
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

  // Hydrate messages from REST API when not actively streaming
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    if (isReconnectScheduled) return;
    setMessages((prev) => {
      if (prev.length >= hydratedMessages.length) return prev;
      return deduplicateMessages(hydratedMessages);
    });
  }, [hydratedMessages, setMessages, status, isReconnectScheduled]);

  // Track resume state per session
  const hasResumedRef = useRef<Map<string, boolean>>(new Map());

  // Clean up reconnect state on session switch
  useEffect(() => {
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = undefined;
    reconnectAttemptsRef.current = 0;
    isReconnectScheduledRef.current = false;
    setIsReconnectScheduled(false);
    hasShownDisconnectToast.current = false;
    isUserStoppingRef.current = false;
    hasResumedRef.current.clear();
  }, [sessionId]);

  // Invalidate session cache when stream completes
  const prevStatusRef = useRef(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isIdle = status === "ready" || status === "error";

    if (wasActive && isIdle && sessionId && !isReconnectScheduled) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
      if (status === "ready") {
        reconnectAttemptsRef.current = 0;
        hasShownDisconnectToast.current = false;
      }
    }
  }, [status, sessionId, queryClient, isReconnectScheduled]);

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

  // True while reconnecting or backend has active stream but we haven't connected yet
  const isReconnecting =
    isReconnectScheduled ||
    (hasActiveStream && status !== "streaming" && status !== "submitted");

  return {
    messages,
    sendMessage,
    stop,
    status,
    error: isReconnecting || isUserStoppingRef.current ? undefined : error,
    isReconnecting,
    isUserStoppingRef,
  };
}
