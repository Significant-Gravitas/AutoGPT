import {
  getGetV2GetCopilotUsageQueryKey,
  getGetV2GetSessionQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
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
  clearKickoffPending,
  getKickoffAttemptTokenFromMetadata,
  getKickoffExpertId,
  getKickoffExpertIdFromMetadata,
  withKickoffLock,
} from "./expertKickoff";
import {
  deduplicateMessages,
  extractSendMessageText,
  getSendSuppressionReason,
  hasInProgressAssistantParts,
  hasVisibleAssistantContent,
  isEngineSwitchPart,
} from "./helpers";
import { asMidTurnFallbackRow } from "./components/ChatMessagesContainer/midTurnSplit";
import { extractDbSequence } from "./helpers/convertChatSessionToUiMessages";
import { getLatestAssistantStatusMessage } from "./messageParts";
import {
  latestProviderFailure,
  parseProviderFailurePart,
  providerFailureFingerprint,
  type ProviderFailure,
} from "./providerFailure";
import { useCopilotUIStore } from "./store";
import type { CopilotLlmModel } from "./store";
import { useCopilotStop } from "./useCopilotStop";
import { useCopilotStreamLifecycle } from "./useCopilotStreamLifecycle";
import { useHydrateOnStreamEnd } from "./useHydrateOnStreamEnd";

/**
 * Batch AI SDK message updates into ~30 ms paints. The smoothing transform in
 * the transport emits word-sized deltas ~10 ms apart; without throttling each
 * word would re-render the whole chat tree.
 */
const STREAM_RENDER_THROTTLE_MS = 30;

interface UseCopilotStreamArgs {
  userId?: string | null;
  sessionId: string | null;
  hydratedMessages: UIMessage[] | undefined;
  /**
   * The session's messages as the API sent them.
   *
   * Conversion to UIMessages merges rows and drops their metadata, so the
   * persisted failure envelope does not survive it. This is the same data
   * before that happens.
   */
  rawSessionMessages?: unknown[];
  /** The route currently persisted for this session. Historical provider
   * failures from a route the chat already left are resolved, not live. */
  sessionAuthProvider?: string | null;
  sessionCredentialId?: string | null;
  /** Id of the first hydrated message of the turn the backend is still
   *  running — the point the GET-resume replay starts from. */
  activeTurnStartMessageId?: string | null;
  hasActiveStream: boolean;
  /** The turn the session view reports running, when it names one. */
  activeTurnId?: string | null;
  refetchSession: () => Promise<{ data?: unknown }>;
  /** Model tier override. `undefined` = let backend decide. */
  copilotModel: CopilotLlmModel | undefined;
}

export function useCopilotStream({
  userId = null,
  sessionId,
  hydratedMessages,
  rawSessionMessages,
  sessionAuthProvider = null,
  sessionCredentialId = null,
  activeTurnStartMessageId = null,
  hasActiveStream,
  activeTurnId = null,
  refetchSession,
  copilotModel,
}: UseCopilotStreamArgs) {
  const queryClient = useQueryClient();
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const [rateLimitMessage, setRateLimitMessage] = useState<string | null>(null);
  // The envelope behind our own cap, when the backend sent one. It rides
  // along to the plan dialog so that dialog can also offer a linked
  // subscription to continue on: the cap does not apply to a turn billed to
  // the user's own credential, so a connected ChatGPT account is a way out
  // that costs them nothing more.
  const [platformLimitFailure, setPlatformLimitFailure] =
    useState<ProviderFailure | null>(null);
  // A linked subscription that stopped accepting turns, as opposed to our own
  // credits running out. Held separately because the answer is different.
  const [providerLimit, setProviderLimit] = useState<ProviderFailure | null>(
    null,
  );
  const dismissedProviderFailureRef = useRef<{
    sessionId: string;
    fingerprint: string;
  } | null>(null);
  function dismissRateLimit() {
    setRateLimitMessage(null);
    setPlatformLimitFailure(null);
  }
  const chatRuntime = useMemo(() => {
    if (!sessionId) return null;
    if (shouldReloadCopilotChatRuntime(sessionId)) {
      resetCopilotChatRuntime(sessionId);
    }
    return getOrCreateCopilotChatRuntime(sessionId);
  }, [sessionId]);
  if (chatRuntime) {
    chatRuntime.copilotModelRef.current = copilotModel;
  }

  // Synchronous flag read inside SDK callbacks — kept as a ref so callbacks
  // don't have to trigger re-renders to observe changes. Scoped to this
  // mount (= this session): the parent remounts on session switch, so a
  // plain boolean can't bleed state across sessions.
  const isUserStoppingRef = useRef(false);
  const pendingEngineSwitchRef = useRef(false);
  // Cleared once consumed, so a later failure without an envelope cannot
  // inherit the explanation of an earlier one.
  const providerFailureRef = useRef<ProviderFailure | null>(null);
  // State mirror of ``isUserStoppingRef`` — the ref is read synchronously
  // inside SDK callbacks, the state drives UI so a click on the stop button
  // immediately overrides ``isStreaming`` regardless of whether AI SDK has
  // flipped ``status`` back to ``ready`` yet.
  const [isUserStopping, setIsUserStopping] = useState(false);
  // Filled once the lifecycle hook below runs; `useChat`'s callbacks read it
  // at fire time. A send that never reached the backend may still have
  // started a turn, so it is looked for rather than retried.
  const followBackendTurnRef = useRef<() => void>(() => {});

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
      ? {
          chat: chatRuntime.chat,
          experimental_throttle: STREAM_RENDER_THROTTLE_MS,
        }
      : {
          id: "new",
        },
  );

  useEffect(() => {
    if (!chatRuntime) return;

    function handleFinish({
      isDisconnect,
      isAbort,
    }: {
      isDisconnect?: boolean;
      isAbort?: boolean;
    }) {
      if (isAbort || !sessionId) return;
      if (isUserStoppingRef.current) return;
      providerFailureRef.current = null;
      if (isDisconnect) followBackendTurnRef.current();
    }

    function handleError(error: Error) {
      if (!sessionId) return;
      const coord = useCopilotStreamStore.getState().getCoord(sessionId);
      const kickoffExpertId = coord.lastSubmittedKickoffExpertId;
      const kickoffAttemptToken = coord.lastSubmittedKickoffAttemptToken;
      function releaseKickoffPending() {
        if (!userId || !kickoffExpertId || !kickoffAttemptToken) return;
        void withKickoffLock(userId, kickoffExpertId, async () => {
          clearKickoffPending(userId, kickoffExpertId, kickoffAttemptToken);
        }).catch(() => undefined);
      }
      const failureForThisTurn = providerFailureRef.current;
      providerFailureRef.current = null;
      handleStreamError({
        error,
        providerFailure: failureForThisTurn,
        onRateLimit: (message, limitFailure, origin) => {
          // Backend raises 429 BEFORE persisting the user message, so the
          // optimistic user bubble added by useChat is a lie. Restore the text
          // into the composer (via the same store slot URL pre-fills use) and
          // drop the unsent bubble so the user can edit/resend after reset.
          const coord = useCopilotStreamStore.getState().getCoord(sessionId);
          const unsentText = coord.lastSubmittedMessageText;
          if (unsentText) {
            // The expert-kickoff control prompt must never surface in the
            // user's composer — drop the recovery slot instead of restoring
            // it. Its bubble below is dropped too; the once-per-expert
            // pending latch expires so a later visit can retry the kickoff.
            if (kickoffExpertId) {
              useCopilotStreamStore.getState().updateCoord(sessionId, {
                lastSubmittedMessageText: null,
                lastSubmittedKickoffExpertId: null,
                lastSubmittedKickoffAttemptToken: null,
              });
            } else {
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
                useCopilotStreamStore.getState().updateCoord(sessionId, {
                  lastSubmittedMessageText: null,
                  lastSubmittedKickoffExpertId: null,
                  lastSubmittedKickoffAttemptToken: null,
                });
              }
            }
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              const next = last?.role === "user" ? prev.slice(0, -1) : prev;
              useCopilotStreamStore
                .getState()
                .setMessageSnapshot(sessionId, next);
              return next;
            });
          }
          // A provider's own limit is not answered by upgrading with us, so
          // it opens the continue path instead of the plan dialog.
          //
          // Which limit it is turns on where the turn was refused, not on
          // which connection it ran on. Our own budget is refused at
          // admission, before the stream opens; a provider refuses mid-turn,
          // on the stream. Both carry an envelope now, so the envelope alone
          // no longer says which. Reading the connection instead meant a
          // self-host -- where the route is "platform" because the deployment
          // holds the key -- was told "Daily usage limit reached, upgrade
          // your plan" when its own OpenRouter or local gateway had
          // rate-limited it. That is a claim about an account we do not bill,
          // offering a plan that would not help.
          if (limitFailure && origin === "provider") {
            // A later turn can fail in exactly the same way as an earlier one
            // the user dismissed. Live stream evidence is a new occurrence.
            dismissedProviderFailureRef.current = null;
            setProviderLimit(limitFailure);
          } else {
            // Our own cap. The envelope, when there is one, lets the plan
            // dialog offer a linked subscription to continue on beside the
            // upgrade it always offered. Older backends send a bare string
            // here and get the dialog exactly as it was.
            setPlatformLimitFailure(limitFailure ?? null);
            setRateLimitMessage(limitFailure?.message || message);
          }
        },
        onReconnect: () => followBackendTurnRef.current(),
        isUserStoppingRef,
      });
      if (kickoffExpertId) {
        releaseKickoffPending();
        useCopilotStreamStore.getState().updateCoord(sessionId, {
          lastSubmittedMessageText: null,
          lastSubmittedKickoffExpertId: null,
          lastSubmittedKickoffAttemptToken: null,
        });
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          const next = last?.role === "user" ? prev.slice(0, -1) : prev;
          useCopilotStreamStore.getState().setMessageSnapshot(sessionId, next);
          return next;
        });
      }
    }

    function handleData(dataPart: { type: string; data?: unknown }) {
      // The execution engine is an internal detail with no control and no
      // display — but a switch still takes longer to settle, so the signal
      // is kept to widen the post-finish refetch window below.
      if (isEngineSwitchPart(dataPart)) {
        pendingEngineSwitchRef.current = true;
      }
      // The envelope always precedes the error frame it explains, so
      // stashing it here means handleError has it in hand.
      const failure = parseProviderFailurePart(dataPart);
      if (failure) {
        providerFailureRef.current = failure;
      }
    }

    chatRuntime.onFinish = handleFinish;
    chatRuntime.onError = handleError;
    chatRuntime.onData = handleData;

    return () => {
      if (chatRuntime.onFinish === handleFinish) {
        chatRuntime.onFinish = undefined;
      }
      if (chatRuntime.onData === handleData) {
        chatRuntime.onData = undefined;
      }
      if (chatRuntime.onError === handleError) {
        chatRuntime.onError = undefined;
      }
    };
  }, [chatRuntime, sessionId, setInitialPrompt, setMessages, userId]);

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
  //     produced anything, suppressing the restore spinner. Checking
  //     content (not just status)
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

  function attachTurn(turnId: string | null) {
    if (!chatRuntime) return;
    if (sessionId) {
      markCopilotChatRuntimeHealthy(sessionId);
    }
    setMessages((prev) => {
      // The GET-resume replays the active turn from its start as a fresh
      // assistant message, so any HYDRATED partial of that turn must be
      // dropped first — keeping it splits one turn into two bubbles with
      // two tool chains. Trimming only in-progress tails is not enough: a
      // turn parked between tools (e.g. waiting on a handoff) hydrates
      // with every persisted part already complete. Only db-hydrated
      // messages (``-seq-N`` ids) are dropped — a streamed tail belongs
      // to a finished turn this mount ran (continuation reconnect) and
      // the resume will NOT replay it.
      // The trim starts at the running turn's first hydrated message, not
      // at the last user message: a turn the backend started on its own
      // (engine-switch continuation) has no user row in front of it, so a
      // user-anchored cut would also delete the completed answer above it
      // — content the resume never replays. The turn's own opening prompt
      // is kept (the replay does not re-emit it); every assistant row after
      // it goes, and the cut does not stop at a user row the backend drained
      // into the middle of the turn — leaving the pre-drain chain above it
      // would show that chain twice. The drained row itself is kept, as a
      // fallback bubble above the replayed assistant: the pending buffer it
      // came from is empty, and a replay whose hint carries no text (older
      // backend) cannot redraw it. When the hint does carry the text, the
      // transcript draws the bubble at the drain point and drops the
      // matching fallback row (`splitMessagesAtDrainHints`).
      const lastUserIndex = prev.findLastIndex((m) => m.role === "user");
      const userCut = lastUserIndex === -1 ? -1 : lastUserIndex + 1;
      const activeTurnIndex = activeTurnStartMessageId
        ? prev.findIndex((m) => m.id === activeTurnStartMessageId)
        : -1;
      const cutIndex =
        activeTurnIndex === -1
          ? userCut
          : activeTurnIndex + (prev[activeTurnIndex].role === "user" ? 1 : 0);
      const tail = cutIndex === -1 ? [] : prev.slice(cutIndex);
      if (tail.length > 0 && tail.every((m) => extractDbSequence(m) !== null)) {
        const drainedFollowUps = tail
          .filter((m) => m.role === "user")
          .map(asMidTurnFallbackRow);
        return [...prev.slice(0, cutIndex), ...drainedFollowUps];
      }
      const last = prev[prev.length - 1];
      return hasInProgressAssistantParts(last) ? prev.slice(0, -1) : prev;
    });
    const chatMessages = chatRuntime.chat.messages;
    chatRuntime.transport.setResumeTarget({
      turnId,
      continuesLastMessage:
        chatMessages[chatMessages.length - 1]?.role === "assistant",
    });
    void resumeStream();
  }

  const streamLifecycle = useCopilotStreamLifecycle({
    runtime: chatRuntime,
    status,
    hasActiveStream,
    activeTurnId,
    canAttach: !!sessionId && !!hydratedMessages,
    refetchSession,
    attachTurn,
    consumeEngineSwitch() {
      const pending = pendingEngineSwitchRef.current;
      pendingEngineSwitchRef.current = false;
      return pending;
    },
  });
  followBackendTurnRef.current = streamLifecycle.followBackendTurn;
  const isReconnectScheduled = streamLifecycle.isReconnecting;
  const isFinishProbing = streamLifecycle.isFinishProbing;

  // Wrap sdkSendMessage to guard against re-sending the user message during a
  // reconnect cycle. If the session already has the message (i.e. we are in a
  // reconnect/resume flow), only GET-resume is safe — never re-POST.
  async function sendMessage(
    ...args: Parameters<typeof sdkSendMessage>
  ): ReturnType<typeof sdkSendMessage> {
    const text = extractSendMessageText(args[0]);
    const metadata =
      args[0] && typeof args[0] === "object" && "metadata" in args[0]
        ? args[0].metadata
        : undefined;
    const kickoffExpertId = getKickoffExpertIdFromMetadata(metadata);
    const kickoffAttemptToken = getKickoffAttemptTokenFromMetadata(metadata);
    const sid = sessionId;
    const coord = sid ? useCopilotStreamStore.getState().getCoord(sid) : null;

    const suppressReason = getSendSuppressionReason({
      text,
      isReconnectScheduled: isReconnectScheduled,
      lastSubmittedText: coord?.lastSubmittedMessageText ?? null,
      messages: chatRuntime?.chat.messages ?? rawMessages,
      status: chatRuntime?.chat.status ?? status,
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
      useCopilotStreamStore.getState().updateCoord(sid, {
        lastSubmittedMessageText: text,
        lastSubmittedKickoffExpertId: kickoffExpertId,
        lastSubmittedKickoffAttemptToken: kickoffAttemptToken,
      });
    }
    hasSentThisMountRef.current = true;
    streamLifecycle.clearStop();
    if (isUserStoppingRef.current) {
      isUserStoppingRef.current = false;
    }
    if (isUserStopping) {
      setIsUserStopping(false);
    }
    return sdkSendMessage(...args);
  }

  // Every entry reaches the parser once (the turn stream drops what is at or
  // before its cursor), so only ids are deduped, never content.
  const messages = deduplicateMessages(rawMessages);

  useEffect(() => {
    if (!sessionId) return;
    if (messages.length === 0) return;
    useCopilotStreamStore.getState().setMessageSnapshot(sessionId, messages);
  }, [sessionId, messages]);

  // A failure the chat is still sitting on survives a reload, because the
  // backend persisted the envelope onto the marker row for exactly this. It
  // was only ever read from the live stream before, so refreshing, opening the
  // chat in another tab, or closing the laptop took away the one control that
  // offered a way out -- leaving the chat latched to the connection that had
  // just refused it, with no way to say "continue on the other one".
  useEffect(() => {
    setProviderLimit(null);
    dismissedProviderFailureRef.current = null;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !rawSessionMessages?.length) return;
    const historical = latestProviderFailure(
      rawSessionMessages,
      sessionAuthProvider
        ? {
            authProvider: sessionAuthProvider,
            credentialId: sessionCredentialId,
          }
        : null,
    );
    const dismissed = dismissedProviderFailureRef.current;
    if (
      historical &&
      dismissed?.sessionId === sessionId &&
      dismissed.fingerprint === providerFailureFingerprint(historical)
    ) {
      return;
    }
    // A live failure is the fresher truth; never let history overwrite it.
    setProviderLimit((current) => current ?? historical);
  }, [rawSessionMessages, sessionAuthProvider, sessionCredentialId, sessionId]);

  useEffect(() => {
    if (!sessionId || !hydratedMessages) return;
    const coord = useCopilotStreamStore.getState().getCoord(sessionId);
    const kickoffExpertId = coord.lastSubmittedKickoffExpertId;
    if (!kickoffExpertId) return;
    if (
      !hydratedMessages.some(
        (message) => getKickoffExpertId(message) === kickoffExpertId,
      )
    ) {
      return;
    }
    useCopilotStreamStore.getState().updateCoord(sessionId, {
      lastSubmittedMessageText: null,
      lastSubmittedKickoffExpertId: null,
      lastSubmittedKickoffAttemptToken: null,
    });
  }, [hydratedMessages, sessionId]);

  const stop = useCopilotStop({
    sessionId,
    sdkStop,
    closeStream: streamLifecycle.stopTurn,
    setMessages,
    isUserStoppingRef,
    setIsUserStopping,
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
    isFinishProbing,
    turnVerified: streamLifecycle.turnVerified,
    setMessages,
  });

  // Invalidate session + usage caches when the stream completes.
  // `lastSubmittedMessageText` is intentionally NOT cleared here: it prevents
  // `getSendSuppressionReason` from allowing a duplicate POST of the same
  // message immediately after a successful turn. Failed turns are exempt
  // from duplicate suppression so the error card can retry the same text.
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
    if (isUserStopping) {
      setIsUserStopping(false);
    }
  }, [hasActiveStream, isUserStopping]);

  // True while reconnecting or backend has active stream but we haven't
  // connected yet on this mount.  Once we've seen visible content this mount,
  // a lingering ``hasActiveStream=true`` from a slow session refetch (e.g.
  // backend still clearing metadata after the SSE finish) must NOT lock the
  // input — a real reconnect shows through the stream lifecycle.
  const isReconnecting =
    !isUserStoppingRef.current &&
    (isReconnectScheduled ||
      (hasActiveStream &&
        !hasConnectedThisMountRef.current &&
        status !== "streaming" &&
        status !== "submitted"));

  const isRestoringActiveSession =
    !isUserStoppingRef.current &&
    hasActiveStream &&
    !hasConnectedThisMountRef.current;

  return {
    followBackendTurn: streamLifecycle.followBackendTurn,
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error: isReconnecting || isUserStoppingRef.current ? undefined : error,
    isReconnecting,
    isFinishProbing,
    isRestoringActiveSession,
    isSyncing: streamLifecycle.isSyncing,
    isUserStoppingRef,
    isUserStopping,
    rateLimitMessage,
    platformLimitFailure,
    providerLimit,
    dismissProviderLimit: () => {
      if (sessionId && providerLimit) {
        dismissedProviderFailureRef.current = {
          sessionId,
          fingerprint: providerFailureFingerprint(providerLimit),
        };
      }
      setProviderLimit(null);
    },
    dismissRateLimit,
  };
}
