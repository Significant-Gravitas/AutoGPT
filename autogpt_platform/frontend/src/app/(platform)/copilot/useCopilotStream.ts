import {
  getGetV2GetCopilotUsageQueryKey,
  getGetV2GetSessionQueryKey,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import type { FileUIPart, UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  getCopilotAuthHeaders,
  deduplicateMessages,
  extractSendMessageText,
  hasActiveBackendStream,
  resolveInProgressTools,
  getSendSuppressionReason,
  disconnectSessionStream,
} from "./helpers";
import type { CopilotLlmModel, CopilotMode } from "./store";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_ATTEMPTS = 3;

/** Minimum time the page must have been hidden to trigger a wake re-sync. */
const WAKE_RESYNC_THRESHOLD_MS = 30_000;

interface UseCopilotStreamArgs {
  sessionId: string | null;
  hydratedMessages: UIMessage[] | undefined;
  hasActiveStream: boolean;
  refetchSession: () => Promise<{ data?: unknown }>;
  /** Autopilot mode to use for requests. `undefined` = let backend decide via feature flags. */
  copilotMode: CopilotMode | undefined;
  /** Model tier override. `undefined` = let backend decide. */
  copilotModel: CopilotLlmModel | undefined;
}

export function useCopilotStream({
  sessionId,
  hydratedMessages,
  hasActiveStream,
  refetchSession,
  copilotMode,
  copilotModel,
}: UseCopilotStreamArgs) {
  const queryClient = useQueryClient();
  const [rateLimitMessage, setRateLimitMessage] = useState<string | null>(null);
  function dismissRateLimit() {
    setRateLimitMessage(null);
  }
  // Use refs for copilotMode and copilotModel so the transport closure always reads
  // the latest value without recreating the DefaultChatTransport (which would
  // reset useChat's internal Chat instance and break mid-session streaming).
  const copilotModeRef = useRef(copilotMode);
  copilotModeRef.current = copilotMode;
  const copilotModelRef = useRef(copilotModel);
  copilotModelRef.current = copilotModel;

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
              // Extract file_ids from FileUIPart entries on the message
              const fileIds = last.parts
                ?.filter((p): p is FileUIPart => p.type === "file")
                .map((p) => {
                  // URL is like /api/proxy/api/workspace/files/{id}/download
                  const match = p.url.match(/\/workspace\/files\/([^/]+)\//);
                  return match?.[1];
                })
                .filter(Boolean) as string[] | undefined;
              return {
                body: {
                  message: (
                    last.parts?.map((p) => (p.type === "text" ? p.text : "")) ??
                    []
                  ).join(""),
                  is_user_message: last.role === "user",
                  context: null,
                  file_ids: fileIds && fileIds.length > 0 ? fileIds : null,
                  mode: copilotModeRef.current ?? null,
                  model: copilotModelRef.current ?? null,
                },
                headers: await getCopilotAuthHeaders(),
              };
            },
            prepareReconnectToStreamRequest: async () => ({
              api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
              headers: await getCopilotAuthHeaders(),
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
  // Set when all reconnect attempts are exhausted — prevents hasActiveStream
  // from keeping the UI blocked forever when the backend is slow to clear it.
  // Must be state (not ref) so that setting it triggers a re-render and
  // recomputes `isReconnecting`.
  const [reconnectExhausted, setReconnectExhausted] = useState(false);
  // True while performing a wake re-sync (blocks chat input).
  const [isSyncing, setIsSyncing] = useState(false);
  // Tracks the last time the page was hidden — used to detect sleep/wake gaps.
  const lastHiddenAtRef = useRef(Date.now());

  function handleReconnect(sid: string) {
    if (isReconnectScheduledRef.current || !sid) return;

    const nextAttempt = reconnectAttemptsRef.current + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      setReconnectExhausted(true);
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
      // Strip the stale in-progress assistant message before resuming —
      // the backend replays from "0-0", so keeping it would duplicate parts.
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === "assistant") {
          return prev.slice(0, -1);
        }
        return prev;
      });
      resumeStreamRef.current();
    }, delay);
  }

  // Tracks the ID of the last user message that was submitted via sendMessage.
  // During a reconnect cycle, if the session already contains this message, we
  // must not POST it again — only GET-resume is safe.
  const lastSubmittedMsgRef = useRef<string | null>(null);

  const {
    messages: rawMessages,
    sendMessage: sdkSendMessage,
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

      // Check if backend executor is still running after clean close.
      // Brief delay to let the backend clear active_stream — without this,
      // the refetch often races and sees stale active_stream=true, triggering
      // unnecessary reconnect cycles.
      await new Promise((r) => setTimeout(r, 500));
      const result = await refetchSession();
      if (hasActiveBackendStream(result)) {
        handleReconnect(sessionId);
      }
    },
    onError: (error) => {
      if (!sessionId) return;

      // Detect rate limit (429) responses and show reset time to the user.
      // The SDK throws a plain Error whose message is the raw response body
      // (FastAPI returns {"detail": "...usage limit..."} for 429s).
      let errorDetail: string = error.message;
      try {
        const parsed = JSON.parse(error.message) as unknown;
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          "detail" in parsed &&
          typeof (parsed as { detail: unknown }).detail === "string"
        ) {
          errorDetail = (parsed as { detail: string }).detail;
        }
      } catch {
        // Not JSON — use message as-is
      }
      const isRateLimited = errorDetail.toLowerCase().includes("usage limit");
      if (isRateLimited) {
        setRateLimitMessage(
          errorDetail ||
            "You've reached your usage limit. Please try again later.",
        );
        return;
      }

      // Detect authentication failures (from getCopilotAuthHeaders or 401 responses)
      const isAuthError =
        errorDetail.includes("Authentication failed") ||
        errorDetail.includes("Unauthorized") ||
        errorDetail.includes("Not authenticated") ||
        errorDetail.toLowerCase().includes("401");
      if (isAuthError) {
        toast({
          title: "Authentication error",
          description: "Your session may have expired. Please sign in again.",
          variant: "destructive",
        });
        return;
      }

      // Reconnect on network errors or transient API errors so the
      // persisted retryable-error marker is loaded and the "Try Again"
      // button appears.  Without this, transient errors only show in the
      // onError callback (where StreamError strips the retryable prefix).
      if (isUserStoppingRef.current) return;
      const isNetworkError =
        error.name === "TypeError" || error.name === "AbortError";
      const isTransientApiError = errorDetail.includes(
        "connection interrupted",
      );
      if (isNetworkError || isTransientApiError) {
        handleReconnect(sessionId);
      }
    },
  });

  // Keep stable refs to sdkStop and resumeStream so that async callbacks
  // (session-switch cleanup, wake re-sync, reconnect timer) always call the
  // latest version without stale-closure bugs.
  const sdkStopRef = useRef(sdkStop);
  sdkStopRef.current = sdkStop;
  const resumeStreamRef = useRef(resumeStream);
  resumeStreamRef.current = resumeStream;

  // Wrap sdkSendMessage to guard against re-sending the user message during a
  // reconnect cycle. If the session already has the message (i.e. we are in a
  // reconnect/resume flow), only GET-resume is safe — never re-POST.
  const sendMessage: typeof sdkSendMessage = async (...args) => {
    const text = extractSendMessageText(args[0]);

    const suppressReason = getSendSuppressionReason({
      text,
      isReconnectScheduled: isReconnectScheduledRef.current,
      lastSubmittedText: lastSubmittedMsgRef.current,
      messages: rawMessages,
    });

    if (suppressReason === "reconnecting") {
      // The ref flips to ``true`` synchronously while the React state that
      // drives the UI's disabled state only updates on the next render, so
      // the user may have clicked send against a still-enabled input. Tell
      // them their message wasn't dropped silently.
      toast({
        title: "Reconnecting",
        description: "Wait for the connection to resume before sending.",
      });
      return;
    }
    if (suppressReason === "duplicate") return;

    lastSubmittedMsgRef.current = text;
    return sdkSendMessage(...args);
  };

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
    // Resolve pending tool calls and inject a cancellation marker so the UI
    // shows "You manually stopped this chat" immediately (the backend writes
    // the same marker to the DB, but the SSE connection is already aborted).
    // Marker must match COPILOT_ERROR_PREFIX in ChatMessagesContainer/helpers.ts.
    setMessages((prev) => {
      const resolved = resolveInProgressTools(prev, "cancelled");
      const last = resolved[resolved.length - 1];
      if (last?.role === "assistant") {
        return [
          ...resolved.slice(0, -1),
          {
            ...last,
            parts: [
              ...last.parts,
              {
                type: "text" as const,
                text: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
              },
            ],
          },
        ];
      }
      return resolved;
    });

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

  // Keep a ref to sessionId so the async wake handler can detect staleness.
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  // ---------------------------------------------------------------------------
  // Wake detection: when the page becomes visible after being hidden for >30s
  // (device sleep, tab backgrounded for a long time), refetch the session to
  // pick up any messages the backend produced while the SSE was dead.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    async function handleWakeResync() {
      const sid = sessionIdRef.current;
      if (!sid) return;

      const elapsed = Date.now() - lastHiddenAtRef.current;
      lastHiddenAtRef.current = Date.now();

      if (document.visibilityState !== "visible") return;
      if (elapsed < WAKE_RESYNC_THRESHOLD_MS) return;

      setIsSyncing(true);
      try {
        const result = await refetchSession();
        // Bail out if the session changed while the refetch was in flight.
        if (sessionIdRef.current !== sid) return;

        if (hasActiveBackendStream(result)) {
          // Stream is still running — resume SSE to pick up live chunks.
          // Remove stale in-progress assistant message first (backend replays
          // from "0-0").
          setMessages((prev) => {
            if (prev.length > 0 && prev[prev.length - 1].role === "assistant") {
              return prev.slice(0, -1);
            }
            return prev;
          });
          await resumeStreamRef.current();
        }
        // If !backendActive, the refetch will update hydratedMessages via
        // React Query, and the hydration effect below will merge them in.
      } catch (err) {
        console.warn("[copilot] wake re-sync failed", err);
      } finally {
        setIsSyncing(false);
      }
    }

    function onVisibilityChange() {
      if (document.visibilityState === "hidden") {
        lastHiddenAtRef.current = Date.now();
      } else {
        handleWakeResync();
      }
    }

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [refetchSession, setMessages]);

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

  // Clean up reconnect state on session switch.
  // Abort the old stream's in-flight fetch and tell the backend to release
  // its XREAD listeners immediately (fire-and-forget).
  const prevStreamSessionRef = useRef(sessionId);
  useEffect(() => {
    const prevSid = prevStreamSessionRef.current;
    prevStreamSessionRef.current = sessionId;

    const isSwitching = Boolean(prevSid && prevSid !== sessionId);
    if (isSwitching) {
      // Mark BEFORE stopping so the old stream's async onError (which fires
      // after the abort) sees the flag and short-circuits the reconnect path.
      // Without this, the AbortError can queue a reconnect against the new
      // session's `sessionId` (captured in the fresh onError closure).
      isUserStoppingRef.current = true;
      sdkStopRef.current();
      disconnectSessionStream(prevSid!);
      // Schedule the reset as a task (not a microtask) so it runs AFTER the
      // aborted fetch's onError has fired — otherwise the new session would
      // be stuck with the "user stopping" flag set, preventing auto-resume
      // when hydration detects an active backend stream.
      setTimeout(() => {
        isUserStoppingRef.current = false;
      }, 0);
    } else {
      isUserStoppingRef.current = false;
    }

    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = undefined;
    reconnectAttemptsRef.current = 0;
    isReconnectScheduledRef.current = false;
    setIsReconnectScheduled(false);
    setRateLimitMessage(null);
    hasShownDisconnectToast.current = false;
    lastSubmittedMsgRef.current = null;
    setReconnectExhausted(false);
    setIsSyncing(false);
    hasResumedRef.current.clear();
    return () => {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = undefined;
    };
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
      queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
      if (status === "ready") {
        reconnectAttemptsRef.current = 0;
        hasShownDisconnectToast.current = false;
        lastSubmittedMsgRef.current = null;
        setReconnectExhausted(false);
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

    // Don't resume a stream the user just cancelled
    if (isUserStoppingRef.current) return;

    // Mark as resumed immediately to prevent race conditions
    hasResumedRef.current.set(sessionId, true);

    // Remove the in-progress assistant message before resuming.
    // The backend replays the stream from "0-0", so keeping the hydrated
    // version would cause the old parts to overlap with replayed parts.
    // Previous turns are preserved; the stream recreates the current turn.
    setMessages((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].role === "assistant") {
        return prev.slice(0, -1);
      }
      return prev;
    });

    resumeStreamRef.current();
  }, [sessionId, hasActiveStream, hydratedMessages, status, setMessages]);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

  // Reset the user-stop flag once the backend confirms the stream is no
  // longer active — this prevents the flag from staying stale forever.
  useEffect(() => {
    if (!hasActiveStream && isUserStoppingRef.current) {
      isUserStoppingRef.current = false;
    }
  }, [hasActiveStream]);

  // True while reconnecting or backend has active stream but we haven't connected yet.
  // Suppressed when the user explicitly stopped or when all reconnect attempts
  // are exhausted — the backend may be slow to clear active_stream but the UI
  // should remain responsive.
  const isReconnecting =
    !isUserStoppingRef.current &&
    !reconnectExhausted &&
    (isReconnectScheduled ||
      (hasActiveStream && status !== "streaming" && status !== "submitted"));

  return {
    messages,
    sendMessage,
    stop,
    status,
    error: isReconnecting || isUserStoppingRef.current ? undefined : error,
    isReconnecting,
    isSyncing,
    isUserStoppingRef,
    rateLimitMessage,
    dismissRateLimit,
  };
}
