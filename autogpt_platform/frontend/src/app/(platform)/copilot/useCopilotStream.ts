import {
  getGetV2GetCopilotUsageQueryKey,
  getGetV2GetSessionQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import type { UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { handleStreamError } from "./copilotStreamErrorHandlers";
import { useCopilotStreamStore } from "./copilotStreamStore";
import { createCopilotTransport } from "./copilotStreamTransport";
import {
  deduplicateMessages,
  disconnectSessionStream,
  extractSendMessageText,
  findLatestCursorChunkId,
  getSendSuppressionReason,
  hasActiveBackendStream,
} from "./helpers";
import type { CopilotLlmModel, CopilotMode } from "./store";
import { useCopilotReconnect } from "./useCopilotReconnect";
import { useCopilotStop } from "./useCopilotStop";
import { useHydrateOnStreamEnd } from "./useHydrateOnStreamEnd";
import { useWakeResync } from "./useWakeResync";

/**
 * Delay after a clean stream close before refetching the session to check
 * whether the backend executor is still running. Without this, the refetch
 * races with the backend clearing `active_stream` and often reads a stale
 * `active_stream=true`, triggering unnecessary reconnect cycles.
 */
const FINISH_REFETCH_SETTLE_MS = 500;

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
  // This useMemo is load-bearing: `useChat` key-compares `transport` identity,
  // so recreating it mid-session resets the internal `Chat` instance.
  const transport = useMemo(
    () =>
      sessionId
        ? createCopilotTransport({
            sessionId,
            copilotModeRef,
            copilotModelRef,
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
  // Synchronous flag read inside SDK callbacks — kept as a ref so callbacks
  // don't have to trigger re-renders to observe changes. Scoped to this
  // mount (= this session), so a boolean is enough; cross-session scoping
  // is no longer needed because the parent remounts on session switch.
  const isUserStoppingRef = useRef(false);
  // Flipped to `false` during mount cleanup so async callbacks that were
  // already in flight (e.g. the post-stream settle in `onFinish`) bail out
  // instead of arming new timers / HTTP requests against a torn-down mount.
  const isMountedRef = useRef(true);

  // Stable refs for resumeStream + handleReconnect, filled after the
  // corresponding hooks run below. `useChat`'s onFinish/onError closures
  // capture these refs at init and read `.current` at fire time, so the
  // circular "useChat needs handleReconnect and useCopilotReconnect needs
  // resumeStream" is resolved by deferring both through refs.
  const resumeStreamRef = useRef<() => void>(() => {});
  const handleReconnectRef = useRef<() => void>(() => {});

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
        handleReconnectRef.current();
        return;
      }

      // Check if backend executor is still running after clean close.
      await new Promise((r) => setTimeout(r, FINISH_REFETCH_SETTLE_MS));
      if (!isMountedRef.current) return;
      const result = await refetchSession();
      if (!isMountedRef.current) return;
      if (hasActiveBackendStream(result)) {
        handleReconnectRef.current();
      }
    },
    onError: (error) => {
      if (!sessionId) return;
      handleStreamError({
        error,
        onRateLimit: setRateLimitMessage,
        onReconnect: () => handleReconnectRef.current(),
        isUserStoppingRef,
      });
    },
  });

  // Keep stable refs to sdkStop and resumeStream so async callbacks
  // (wake re-sync, reconnect timer, unmount cleanup) always invoke the
  // latest version without stale-closure bugs.
  const sdkStopRef = useRef(sdkStop);
  sdkStopRef.current = sdkStop;
  resumeStreamRef.current = resumeStream;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const {
    isReconnectScheduled,
    reconnectExhausted,
    reconnectScheduledRef,
    handleReconnect,
  } = useCopilotReconnect({
    sessionId,
    status,
    resumeStreamRef,
    hasResumedRef,
  });
  handleReconnectRef.current = handleReconnect;

  // Wrap sdkSendMessage to guard against re-sending the user message during a
  // reconnect cycle. If the session already has the message (i.e. we are in a
  // reconnect/resume flow), only GET-resume is safe — never re-POST.
  async function sendMessage(
    ...args: Parameters<typeof sdkSendMessage>
  ): ReturnType<typeof sdkSendMessage> {
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
  }

  // Deduplicate messages continuously to prevent duplicates when resuming streams.
  const messages = deduplicateMessages(rawMessages);

  const stop = useCopilotStop({
    sessionId,
    sdkStop,
    setMessages,
    isUserStoppingRef,
  });

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
      // Reconnect/forced-timeout timers are cleared by `useCopilotReconnect`
      // via its own mount cleanup — no need to touch them here.
      const sid = sessionIdRef.current;
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

  // Wake detection: refetch + optional resume when the page becomes visible
  // after being hidden for >30 s. See `useWakeResync` for details.
  const { isSyncing } = useWakeResync({
    sessionIdRef,
    isMountedRef,
    refetchSession,
    resumeStreamRef,
  });

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

  // Invalidate session + usage caches when the stream completes.
  // Reconnect counter/timer reset on the same transition is owned by
  // `useCopilotReconnect`, which watches `status` internally.
  // `lastSubmittedMessageText` is intentionally NOT cleared here: it prevents
  // `getSendSuppressionReason` from allowing a duplicate POST of the same
  // message immediately after a successful turn (the "duplicate" branch
  // checks both the store and the visible last user message, so legitimate
  // re-sends after a different reply are still allowed).
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
