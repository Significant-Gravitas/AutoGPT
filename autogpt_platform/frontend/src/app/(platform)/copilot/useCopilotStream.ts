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
import {
  getOrCreateCopilotChatRuntime,
  markCopilotChatRuntimeHealthy,
  resetCopilotChatRuntime,
  shouldReloadCopilotChatRuntime,
} from "./copilotChatRegistry";
import { handleStreamError } from "./copilotStreamErrorHandlers";
import { useCopilotStreamStore } from "./copilotStreamStore";
import {
  deduplicateMessages,
  extractSendMessageText,
  getLatestAssistantStatusMessage,
  getSendSuppressionReason,
  hasActiveBackendStream,
  hasInProgressAssistantParts,
  hasVisibleAssistantContent,
} from "./helpers";
import { useCopilotUIStore } from "./store";
import type { CopilotLlmModel, CopilotMode } from "./store";
import { useCopilotReconnect } from "./useCopilotReconnect";
import { useCopilotStop } from "./useCopilotStop";
import { useHydrateOnStreamEnd } from "./useHydrateOnStreamEnd";
import { RESTORE_STALL_TIMEOUT_MS } from "./restoreConstants";
import { useStreamActivityWatchdog } from "./useStreamActivityWatchdog";
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
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const [rateLimitMessage, setRateLimitMessage] = useState<string | null>(null);
  function dismissRateLimit() {
    setRateLimitMessage(null);
  }
  const chatRuntime = useMemo(() => {
    if (!sessionId) return null;
    if (shouldReloadCopilotChatRuntime(sessionId)) {
      resetCopilotChatRuntime(sessionId);
    }
    return getOrCreateCopilotChatRuntime(sessionId);
  }, [sessionId]);
  if (chatRuntime) {
    chatRuntime.copilotModeRef.current = copilotMode;
    chatRuntime.copilotModelRef.current = copilotModel;
  }

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
  // State mirror of ``isUserStoppingRef`` — the ref is read synchronously
  // inside SDK callbacks, the state drives UI so a click on the stop button
  // immediately overrides ``isStreaming`` regardless of whether AI SDK has
  // flipped ``status`` back to ``ready`` yet (which can lag by many seconds
  // when aborting a GET-based resume fetch).
  const [isUserStopping, setIsUserStopping] = useState(false);
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
  } = useChat(
    chatRuntime
      ? { chat: chatRuntime.chat }
      : {
          id: "new",
        },
  );

  useEffect(() => {
    if (!chatRuntime) return;

    async function handleFinish({
      isDisconnect,
      isAbort,
    }: {
      isDisconnect?: boolean;
      isAbort?: boolean;
    }) {
      if (isAbort || !sessionId) return;
      if (isUserStoppingRef.current) return;

      if (isDisconnect) {
        handleReconnectRef.current();
        return;
      }

      await new Promise((r) => setTimeout(r, FINISH_REFETCH_SETTLE_MS));
      if (!isMountedRef.current) return;
      const result = await refetchSession();
      if (!isMountedRef.current) return;
      if (hasActiveBackendStream(result)) {
        handleReconnectRef.current();
      }
    }

    function handleError(error: Error) {
      if (!sessionId) return;
      handleStreamError({
        error,
        onRateLimit: (message) => {
          // Backend raises 429 BEFORE persisting the user message, so the
          // optimistic user bubble added by useChat is a lie. Restore the text
          // into the composer (via the same store slot URL pre-fills use) and
          // drop the unsent bubble so the user can edit/resend after reset.
          const unsentText =
            useCopilotStreamStore.getState().getCoord(sessionId)
              .lastSubmittedMessageText ?? null;
          if (unsentText) {
            // The 429 callback fires async — by the time it lands, the user
            // may have started typing a new draft. Only restore + clear the
            // recovery slot when the composer is empty; otherwise leave the
            // unsent text in the per-session store so a reload / resume can
            // surface it later instead of silently dropping it.
            const composer = document.getElementById(
              "chat-input",
            ) as HTMLTextAreaElement | null;
            const composerEmpty = !composer || composer.value.length === 0;
            if (composerEmpty) {
              setInitialPrompt(unsentText);
              useCopilotStreamStore
                .getState()
                .updateCoord(sessionId, { lastSubmittedMessageText: null });
            }
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last?.role === "user") return prev.slice(0, -1);
              return prev;
            });
          }
          setRateLimitMessage(message);
        },
        onReconnect: () => handleReconnectRef.current(),
        isUserStoppingRef,
      });
    }

    chatRuntime.onFinish = handleFinish;
    chatRuntime.onError = handleError;

    return () => {
      if (chatRuntime.onFinish === handleFinish) {
        chatRuntime.onFinish = undefined;
      }
      if (chatRuntime.onError === handleError) {
        chatRuntime.onError = undefined;
      }
    };
  }, [chatRuntime, sessionId, refetchSession]);

  // Flipped to ``true`` the first time the user actually hits Send on this
  // mount. Lets the ``hasConnectedThisMountRef`` latch below distinguish
  // "resuming a turn that was already running" from "user sent a brand new
  // turn" — in the latter case the ThinkingIndicator is the right surface
  // from the first render, no "Retrieving latest messages" spinner needed.
  const hasSentThisMountRef = useRef(false);

  // Latch flips from ``false`` → ``true`` the first time the stream is
  // considered "live" on this mount. Observed by ``isRestoringActiveSession``
  // so the "Retrieving latest messages" spinner only shows while we haven't
  // yet connected.
  //
  // Flip conditions (either is sufficient):
  //  1. The user sent a fresh message this mount — we own the turn, so no
  //     restore UI is appropriate even though status is briefly "submitted"
  //     before the first byte lands.
  //  2. The stream is in an active state AND has produced at least one
  //     visible assistant content part (text / reasoning with non-empty
  //     text, any tool part, or a backend status). The ``isStreamLive``
  //     gate is what distinguishes "bytes from the live SSE" from "content
  //     just hydrated from the DB on a fresh mount" — without it, a
  //     mid-stream refresh that lands a partial assistant message in
  //     ``hydratedMessages`` would flip the latch before the GET-resume
  //     produced anything, suppressing the restore spinner and disabling
  //     the restore-stall watchdog. Checking content (not just status)
  //     still keeps the indicator up during the GET-resume-no-bytes window.
  const hasConnectedThisMountRef = useRef(false);
  if (!hasConnectedThisMountRef.current) {
    const isStreamLive = status === "streaming" || status === "submitted";
    if (
      hasSentThisMountRef.current ||
      (isStreamLive &&
        (hasVisibleAssistantContent(rawMessages) ||
          getLatestAssistantStatusMessage(rawMessages) !== null))
    ) {
      hasConnectedThisMountRef.current = true;
    }
  }

  function resumeStreamFromStart() {
    if (sessionId) {
      markCopilotChatRuntimeHealthy(sessionId);
    }
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      return hasInProgressAssistantParts(last) ? prev.slice(0, -1) : prev;
    });
    resumeStream();
  }
  resumeStreamRef.current = resumeStreamFromStart;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const {
    isReconnectScheduled,
    reconnectExhausted,
    reconnectScheduledRef,
    handleReconnect,
  } = useCopilotReconnect({
    sessionId,
    hasActiveStream,
    status,
    hasConnectedThisMountRef,
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
      markCopilotChatRuntimeHealthy(sid);
      useCopilotStreamStore
        .getState()
        .updateCoord(sid, { lastSubmittedMessageText: text });
    }
    hasSentThisMountRef.current = true;
    if (isUserStoppingRef.current) {
      isUserStoppingRef.current = false;
    }
    if (isUserStopping) {
      setIsUserStopping(false);
    }
    return sdkSendMessage(...args);
  }

  // Deduplicate messages continuously to prevent duplicates when resuming streams.
  const messages = deduplicateMessages(rawMessages);

  useEffect(() => {
    if (!sessionId) return;
    if (messages.length === 0) return;
    useCopilotStreamStore.getState().setMessageSnapshot(sessionId, messages);
  }, [sessionId, messages]);

  // Cheap signal that changes on any stream activity — drives the stall
  // watchdog. Counts messages, the last message's parts, and the total
  // text length on the last message so in-place part updates (token
  // streaming) also tick the signal.
  const streamActivityToken = useMemo(() => {
    const last = rawMessages[rawMessages.length - 1];
    let textLen = 0;
    if (last) {
      for (const part of last.parts) {
        if ("text" in part && typeof part.text === "string") {
          textLen += part.text.length;
        }
      }
    }
    return `${rawMessages.length}:${last?.parts.length ?? 0}:${textLen}`;
  }, [rawMessages]);

  const stop = useCopilotStop({
    sessionId,
    sdkStop,
    setMessages,
    isUserStoppingRef,
    setIsUserStopping,
  });

  // Silent-stall watchdog: triggers the reconnect cascade if the stream
  // sits in "submitted" / "streaming" with no activity for 60 s. Handles
  // the case where AI SDK's onFinish / onError never fire (backend hung,
  // Redis zombie, etc.) and the UI would otherwise stay stuck forever.
  useStreamActivityWatchdog({
    sessionId,
    status,
    activityToken: streamActivityToken,
    isReconnectScheduled,
    isUserStoppingRef,
    handleReconnectRef,
  });

  // ---------------------------------------------------------------------------
  // Mount lifecycle:
  //  - On unmount: mark this React subscriber as gone so async follow-up work
  //    cannot set state into a torn-down mount.
  //  - Do NOT abort the underlying session Chat runtime here. It is
  //    intentionally kept alive in JS state so switching away from a chat does
  //    not tear down its live SSE stream.
  // ---------------------------------------------------------------------------
  useMountEffect(() => {
    return () => {
      isMountedRef.current = false;
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
  // working for pagination. Also repairs zombie in-progress parts when the
  // backend confirms no active stream. See useHydrateOnStreamEnd for the
  // timing dance.
  useHydrateOnStreamEnd({
    sessionId,
    status,
    hydratedMessages,
    isReconnectScheduled,
    hasActiveStream,
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

  // Restore watchdog: if we reopened a session with an active backend stream
  // but still have not connected after 6 s of zero replay activity, verify
  // the backend still reports that stream as active and then kick the
  // reconnect cascade. This covers both "status stayed ready / no replay
  // chunks ever appeared" and "resume fetch is alive but only heartbeats are
  // flowing" — the latter never trips the normal stream-activity watchdog
  // because heartbeats do not mutate `messages`.
  useEffect(() => {
    if (!sessionId) return;
    if (!hasActiveStream) return;
    if (hasConnectedThisMountRef.current) return;
    if (isReconnectScheduled || reconnectExhausted) return;
    if (isUserStoppingRef.current) return;

    let cancelled = false;
    const timeout = setTimeout(async () => {
      if (cancelled) return;
      if (!isMountedRef.current) return;
      if (sessionIdRef.current !== sessionId) return;
      if (hasConnectedThisMountRef.current) return;
      if (isUserStoppingRef.current) return;

      const result = await refetchSession();
      if (!isMountedRef.current) return;
      if (sessionIdRef.current !== sessionId) return;
      if (hasConnectedThisMountRef.current) return;
      if (isUserStoppingRef.current) return;
      if (!hasActiveBackendStream(result)) return;

      handleReconnectRef.current();
    }, RESTORE_STALL_TIMEOUT_MS);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [
    sessionId,
    hasActiveStream,
    streamActivityToken,
    isReconnectScheduled,
    reconnectExhausted,
    refetchSession,
  ]);

  // Reset the user-stop flag once the backend confirms the stream is no
  // longer active — this prevents the flag from staying stale forever.
  useEffect(() => {
    if (hasActiveStream) return;
    if (isUserStoppingRef.current) {
      isUserStoppingRef.current = false;
    }
    if (isUserStopping) {
      setIsUserStopping(false);
    }
  }, [hasActiveStream, isUserStopping]);

  // True while reconnecting or backend has active stream but we haven't
  // connected yet on this mount.  Once we've seen visible content this mount,
  // a lingering ``hasActiveStream=true`` from a slow session refetch (e.g.
  // backend still clearing metadata after the SSE finish) must NOT lock the
  // input — legitimate reconnect cases set ``isReconnectScheduled`` via
  // ``handleFinish`` / the watchdogs.
  const isReconnecting =
    !isUserStoppingRef.current &&
    !reconnectExhausted &&
    (isReconnectScheduled ||
      (hasActiveStream &&
        !hasConnectedThisMountRef.current &&
        status !== "streaming" &&
        status !== "submitted"));

  const isRestoringActiveSession =
    !isUserStoppingRef.current &&
    !reconnectExhausted &&
    hasActiveStream &&
    !hasConnectedThisMountRef.current;

  return {
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error: isReconnecting || isUserStoppingRef.current ? undefined : error,
    isReconnecting,
    isRestoringActiveSession,
    isSyncing,
    isUserStoppingRef,
    isUserStopping,
    rateLimitMessage,
    dismissRateLimit,
  };
}
