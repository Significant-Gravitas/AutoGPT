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
import { useCopilotStreamStore } from "./copilotStreamStore";
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
import { useHydrateOnStreamEnd } from "./useHydrateOnStreamEnd";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_ATTEMPTS = 3;

/** Minimum time the page must have been hidden to trigger a wake re-sync. */
const WAKE_RESYNC_THRESHOLD_MS = 30_000;

/** Max time (ms) the UI can stay in "reconnecting" state before forcing idle. */
const RECONNECT_MAX_DURATION_MS = 30_000;

/**
 * Delay after a clean stream close before refetching the session to check
 * whether the backend executor is still running. Without this, the refetch
 * races with the backend clearing `active_stream` and often reads a stale
 * `active_stream=true`, triggering unnecessary reconnect cycles.
 */
const FINISH_REFETCH_SETTLE_MS = 500;

/**
 * Parses a backend-encoded error code from an `errorText` payload.
 *
 * The AI-SDK SSE protocol enforces `z.strictObject({type, errorText})`
 * on StreamError frames, so the backend cannot attach a top-level `code`
 * field. Instead it prefixes the message with `[code:<id>] <msg>` and
 * this helper extracts it client-side.
 */
function parseBackendErrorCode(raw: string): {
  code: string | null;
  message: string;
} {
  const match = raw.match(/^\s*\[code:([a-z0-9_]+)\]\s*(.*)$/is);
  if (!match) return { code: null, message: raw };
  return { code: match[1], message: match[2].trim() };
}

/**
 * User-facing toast copy for each backend error code we surface.
 * `description` defaults to the backend's error message when provided;
 * `fallbackDescription` is used when the backend sends only the code.
 */
const TOAST_BY_BACKEND_CODE: Record<
  string,
  { title: string; fallbackDescription: string }
> = {
  idle_timeout: {
    title: "AutoPilot stopped responding",
    fallbackDescription:
      "A tool call got stuck and the session timed out. Press Try Again to resume.",
  },
  tool_stalled: {
    title: "A tool call is taking too long",
    fallbackDescription:
      "The assistant is waiting on a tool that hasn't responded. Press Try Again to restart.",
  },
  transient_api_error: {
    title: "Connection hiccup",
    fallbackDescription:
      "We hit a temporary error talking to the model. Press Try Again to continue.",
  },
  circuit_breaker_empty_tool_calls: {
    title: "AutoPilot paused",
    fallbackDescription:
      "The assistant made too many empty tool calls in a row and was paused. Press Try Again to continue.",
  },
  all_attempts_exhausted: {
    title: "Conversation too long",
    fallbackDescription:
      "We couldn't fit this chat's history into the model after several attempts. Start a new chat or clear some history.",
  },
  sdk_stream_error: {
    title: "AutoPilot ran into an error",
    fallbackDescription:
      "Something went wrong while the assistant was responding. Press Try Again to retry.",
  },
  sdk_error: {
    title: "AutoPilot ran into an error",
    fallbackDescription:
      "The assistant couldn't complete this turn. Press Try Again to retry.",
  },
};

/** Fallback toast shown for any `[code:X]` we don't have specific copy for. */
const GENERIC_BACKEND_TOAST = {
  title: "AutoPilot ran into a problem",
  fallbackDescription:
    "The assistant stopped unexpectedly. Press Try Again to retry.",
};

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
            prepareReconnectToStreamRequest: async () => {
              const coord = sessionId
                ? useCopilotStreamStore.getState().getCoord(sessionId)
                : null;
              const cursor = coord?.lastChunkId ?? null;
              const base = `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`;
              return {
                api: cursor
                  ? `${base}?last_chunk_id=${encodeURIComponent(cursor)}`
                  : base,
                headers: await getCopilotAuthHeaders(),
              };
            },
          })
        : null,
    [sessionId],
  );

  // Transient per-mount flags. The parent keys this subtree by sessionId,
  // so every session-switch remounts and these refs reset naturally —
  // no cross-session bleed, no blanket "clearCoord" on the Zustand store.
  const hasResumedRef = useRef(false);
  const hydrateCompletedRef = useRef(false);
  const pendingResumeRef = useRef<(() => void) | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectScheduledRef = useRef(false);
  const reconnectStartedAtRef = useRef<number | null>(null);
  const hasShownDisconnectToastRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectTimeoutTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const lastHiddenAtRef = useRef(Date.now());
  // Reactive flag that drives the isReconnecting UI state.
  const [isReconnectScheduled, setIsReconnectScheduled] = useState(false);
  const [reconnectExhausted, setReconnectExhausted] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  // Synchronous flag read inside SDK callbacks — kept as a ref so callbacks
  // don't have to trigger re-renders to observe changes. Scoped to this
  // mount (= this session), so a boolean is enough; cross-session scoping
  // is no longer needed because the parent remounts on session switch.
  const isUserStoppingRef = useRef(false);
  // Flipped to `false` during mount cleanup so async callbacks that were
  // already in flight (e.g. the post-stream settle in `onFinish`) bail out
  // instead of arming new timers / HTTP requests against a torn-down mount.
  const isMountedRef = useRef(true);

  function handleReconnect() {
    if (!sessionId) return;
    if (reconnectScheduledRef.current) return;

    const nextAttempt = reconnectAttemptsRef.current + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      setReconnectExhausted(true);
      toast({
        title: "Connection lost",
        description: "Unable to reconnect. Please refresh the page.",
        variant: "destructive",
      });
      return;
    }

    // Track when reconnection first started for the forced timeout.
    if (reconnectStartedAtRef.current === null) {
      reconnectStartedAtRef.current = Date.now();
      // Schedule a forced timeout — if reconnecting takes longer than
      // RECONNECT_MAX_DURATION_MS, force the UI back to idle.
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = setTimeout(() => {
        // Cancel the pending reconnect timer so it can't fire resumeStream()
        // after the UI has been forced to idle — otherwise we'd end up in an
        // inconsistent state (reconnectExhausted=true + a fresh stream).
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = undefined;
        reconnectScheduledRef.current = false;
        reconnectStartedAtRef.current = null;
        setIsReconnectScheduled(false);
        setReconnectExhausted(true);
        toast({
          title: "Connection timed out",
          description:
            "AutoPilot may still be working. Refresh to check for updates.",
          variant: "destructive",
        });
      }, RECONNECT_MAX_DURATION_MS);
    }

    reconnectScheduledRef.current = true;
    reconnectAttemptsRef.current = nextAttempt;
    setIsReconnectScheduled(true);

    if (!hasShownDisconnectToastRef.current) {
      hasShownDisconnectToastRef.current = true;
      toast({ title: "Connection lost", description: "Reconnecting..." });
    }

    const delay = RECONNECT_BASE_DELAY_MS * 2 ** (nextAttempt - 1);
    reconnectTimerRef.current = setTimeout(() => {
      reconnectScheduledRef.current = false;
      // Mark resumed so a deferred page-load resume (queued in
      // pendingResumeRef while hydration was in-flight) can't double-fire.
      hasResumedRef.current = true;
      setIsReconnectScheduled(false);
      resumeStreamRef.current();
    }, delay);
  }

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
      // User-initiated stops should not trigger reconnection.
      if (isUserStoppingRef.current) return;

      // The AI SDK rarely sets isDisconnect — treat ANY non-user-initiated
      // finish as a potential disconnect when the backend stream is active.
      if (isDisconnect) {
        handleReconnect();
        return;
      }

      // Check if backend executor is still running after clean close.
      await new Promise((r) => setTimeout(r, FINISH_REFETCH_SETTLE_MS));
      if (!isMountedRef.current) return;
      const result = await refetchSession();
      if (!isMountedRef.current) return;
      if (hasActiveBackendStream(result)) {
        handleReconnect();
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

      // Backend error codes are encoded as `[code:<id>] <message>` because
      // the AI-SDK SSE schema (`z.strictObject({type, errorText})`) rejects
      // a top-level `code` field. Parse the prefix so we can surface a
      // focused toast for the specific failure mode; any coded error —
      // including ones we don't have curated copy for — gets a generic
      // "something went wrong" toast so backend hangs are never silent.
      const { code: backendCode, message: backendMessage } =
        parseBackendErrorCode(errorDetail);
      if (backendCode) {
        const userToast =
          TOAST_BY_BACKEND_CODE[backendCode] ?? GENERIC_BACKEND_TOAST;
        toast({
          title: userToast.title,
          description: backendMessage || userToast.fallbackDescription,
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
        handleReconnect();
      }
    },
  });

  // Keep stable refs to sdkStop and resumeStream so async callbacks
  // (wake re-sync, reconnect timer, unmount cleanup) always invoke the
  // latest version without stale-closure bugs.
  const sdkStopRef = useRef(sdkStop);
  sdkStopRef.current = sdkStop;
  const resumeStreamRef = useRef(resumeStream);
  resumeStreamRef.current = resumeStream;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  // Wrap sdkSendMessage to guard against re-sending the user message during a
  // reconnect cycle. If the session already has the message (i.e. we are in a
  // reconnect/resume flow), only GET-resume is safe — never re-POST.
  const sendMessage: typeof sdkSendMessage = async (...args) => {
    const text = extractSendMessageText(args[0]);
    const sid = sessionId;
    const coord = sid ? useCopilotStreamStore.getState().getCoord(sid) : null;

    const suppressReason = getSendSuppressionReason({
      text,
      isReconnectScheduled: reconnectScheduledRef.current,
      lastSubmittedText: coord?.lastSubmittedMessageText ?? null,
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

    if (sid) {
      useCopilotStreamStore
        .getState()
        .updateCoord(sid, { lastSubmittedMessageText: text });
    }
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

  // ---------------------------------------------------------------------------
  // Mount lifecycle:
  //  - On mount: reset useChat's Chat instance for this id. AI SDK caches
  //    Chat instances per id at module scope, so a revisit to a session
  //    would otherwise show whatever stale messages were left in the cache.
  //    Starting empty lets hydration + resume rebuild cleanly from server
  //    state, matching the behaviour of a full page reload.
  //  - On unmount: abort the in-flight fetch and tell the backend to release
  //    its XREAD listeners immediately rather than waiting for the timeout.
  // Effect body reads through refs only so its dep array is genuinely empty.
  // ---------------------------------------------------------------------------
  const setMessagesRef = useRef(setMessages);
  setMessagesRef.current = setMessages;
  useMountEffect(() => {
    setMessagesRef.current([]);
    return () => {
      isMountedRef.current = false;
      const sid = sessionIdRef.current;
      // Clear any armed reconnect / forced-timeout timers before they can
      // fire against a torn-down mount (and toast into the void).
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = undefined;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
      try {
        sdkStopRef.current();
      } catch {
        // Best-effort — aborting a non-running fetch is a no-op.
      }
      if (sid) {
        disconnectSessionStream(sid);
      }
    };
  });

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
        // Bail out if the session changed or the host unmounted while
        // the refetch was in flight.
        if (!isMountedRef.current) return;
        if (sessionIdRef.current !== sid) return;

        if (hasActiveBackendStream(result)) {
          // Stream is still running — resume SSE to pick up live chunks.
          // Backend picks up where we left off via the cursor stored on
          // the per-session coord (see prepareReconnectToStreamRequest).
          resumeStreamRef.current();
        }
        // If !backendActive, the refetch will update hydratedMessages via
        // React Query, and the hydration effect below will merge them in.
      } catch (err) {
        console.warn("[copilot] wake re-sync failed", err);
      } finally {
        if (isMountedRef.current) setIsSyncing(false);
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
  }, [refetchSession]);

  // After-stream hydration — force-replace AI-SDK state with the DB's view
  // once React Query has actually refetched, then keep length-gated top-ups
  // working for pagination. See useHydrateOnStreamEnd for the timing dance.
  useHydrateOnStreamEnd({
    status,
    hydratedMessages,
    isReconnectScheduled,
    setMessages,
  });

  // Mark hydration complete in the transient ref whenever the hydration gate
  // has effectively run (hydrated data is present and we're not mid-stream),
  // and flush any `pendingResume` that was deferred while hydration was
  // still pending.
  useEffect(() => {
    if (!sessionId) return;
    if (!hydratedMessages) return;
    if (status === "streaming" || status === "submitted") return;
    if (isReconnectScheduled) return;

    hydrateCompletedRef.current = true;
    const pending = pendingResumeRef.current;
    if (pending) {
      pendingResumeRef.current = null;
      pending();
    }
  }, [sessionId, hydratedMessages, status, isReconnectScheduled]);

  // Invalidate session cache when the stream completes and clear the forced
  // reconnect timeout the moment we're streaming again.
  const prevStatusRef = useRef(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isNowActive = status === "streaming" || status === "submitted";
    const isIdle = status === "ready" || status === "error";

    // Clear the forced reconnect timeout as soon as the stream resumes —
    // otherwise the stale 30s timer can fire mid-stream and show a
    // "timed out" toast even though reconnection succeeded.
    if (isNowActive && sessionId && reconnectStartedAtRef.current !== null) {
      reconnectStartedAtRef.current = null;
      clearTimeout(reconnectTimeoutTimerRef.current);
      reconnectTimeoutTimerRef.current = undefined;
    }

    if (wasActive && isIdle && sessionId && !isReconnectScheduled) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
      if (status === "ready") {
        reconnectAttemptsRef.current = 0;
        hasShownDisconnectToastRef.current = false;
        reconnectStartedAtRef.current = null;
        clearTimeout(reconnectTimeoutTimerRef.current);
        reconnectTimeoutTimerRef.current = undefined;
        // Intentionally NOT clearing lastSubmittedMessageText here: keeping
        // the last submitted text prevents getSendSuppressionReason from
        // allowing a duplicate POST of the same message immediately after a
        // successful turn (the "duplicate" branch checks both the store and
        // the visible last user message, so legitimate re-sends after a
        // different reply are still allowed).
        setReconnectExhausted(false);
      }
    }
  }, [status, sessionId, queryClient, isReconnectScheduled]);

  // Track the most recent `data-cursor` Redis stream ID emitted by the
  // backend (one is sent after every chunk — see StreamCursor). Stored on
  // the per-session coord in Zustand so it SURVIVES a session switch and
  // `prepareReconnectToStreamRequest` can append `?last_chunk_id=…` on the
  // next resume, making XREAD resume at the exclusive successor instead of
  // replaying the full turn.
  useEffect(() => {
    if (!sessionId) return;
    const latest = findLatestCursorChunkId(rawMessages);
    if (!latest) return;
    const coord = useCopilotStreamStore.getState().getCoord(sessionId);
    if (coord.lastChunkId === latest) return;
    useCopilotStreamStore
      .getState()
      .updateCoord(sessionId, { lastChunkId: latest });
  }, [rawMessages, sessionId]);

  // Resume an active stream AFTER hydration completes.
  // Only runs when this mount opens on a session with an already-active
  // backend stream (page reload OR session-switch-back). Does NOT run when
  // the user sends a new message mid-session (that goes through POST).
  // Gated on the transient `hydrateCompletedRef` to prevent racing the
  // hydration effect.
  useEffect(() => {
    if (!sessionId) return;
    if (!hasActiveStream) return;
    if (!hydratedMessages) return;

    // Never resume if currently streaming.
    if (status === "streaming" || status === "submitted") return;

    // Only resume once per mount.
    if (hasResumedRef.current) return;

    // Don't resume a stream the user just cancelled.
    if (isUserStoppingRef.current) return;

    function doResume() {
      if (!sessionId) return;
      if (hasResumedRef.current) return;
      if (isUserStoppingRef.current) return;
      hasResumedRef.current = true;
      resumeStreamRef.current();
    }

    // Wait for hydration to complete before resuming to prevent the two
    // effects from racing (duplicate messages / missing content).
    if (!hydrateCompletedRef.current) {
      pendingResumeRef.current = doResume;
      return;
    }

    doResume();
  }, [sessionId, hasActiveStream, hydratedMessages, status]);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

  // Reset the user-stop flag once the backend confirms the stream is no
  // longer active — this prevents the flag from staying stale forever.
  useEffect(() => {
    if (hasActiveStream) return;
    if (isUserStoppingRef.current) {
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
    setMessages,
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

/**
 * Named wrapper around `useEffect(fn, [])` so the intent ("run on mount
 * and clean up on unmount") is explicit and the exhaustive-deps lint
 * rule doesn't flag a legitimately empty dep array.
 */
function useMountEffect(effect: () => void | (() => void)): void {
  useEffect(effect, []);
}

/**
 * Scan a message list for the most recent `data-cursor` part — the Redis
 * Stream XADD id the backend emitted after its previous chunk — and return
 * the raw `chunkId`. Returns `null` if no cursor parts have been received
 * yet (e.g. the backend hasn't shipped the first chunk of the current turn,
 * or this is an older turn from before cursor emission existed).
 */
function findLatestCursorChunkId(messages: UIMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const parts = messages[i].parts;
    for (let j = parts.length - 1; j >= 0; j--) {
      const part = parts[j] as { type?: unknown; data?: unknown };
      if (part?.type !== "data-cursor") continue;
      const data = part.data as { chunkId?: unknown } | undefined;
      if (data && typeof data.chunkId === "string") return data.chunkId;
    }
  }
  return null;
}
