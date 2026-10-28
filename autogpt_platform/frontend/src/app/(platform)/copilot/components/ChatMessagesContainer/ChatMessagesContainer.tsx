import { useCostReplyMoment } from "@/app/(platform)/components/ContextualTips/useTipMoment";
import { parseSpecialMarkers } from "../../helpers/messageMarkers";
import { CredentialMentionText } from "../CredentialMention/CredentialMentionText";
import { cn } from "@/lib/utils";
import { useMemo, useState } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/organisms/Conversation/Conversation";
import {
  Message,
  MessageContent,
} from "@/components/organisms/Message/Message";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useInsertionEffect, useLayoutEffect, useRef } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { useElapsedAnchor } from "../JobStatsBar/useElapsedTimer";
import { CopilotPendingReviews } from "../CopilotPendingReviews/CopilotPendingReviews";
import type { TurnStatsMap } from "../../helpers/convertChatSessionToUiMessages";
import { hideKickoffMessages } from "../../expertKickoff";
import { Text } from "@/components/atoms/Text/Text";
import {
  answeredLabel,
  countHeldCalls,
  getHeldCallRowKind,
} from "./heldCallRows";
import { extractReviewTargets, getLatestCompactionPhase } from "./helpers";
import {
  buildMessageIndexById,
  isMidTurnSegmentRow,
  splitMessagesAtDrainHints,
  turnMessagesForRow,
} from "./midTurnSplit";
import {
  isBookkeepingPart,
  PENDING_DRAINED_PART_TYPE,
} from "../../messageParts";
import { RESTORE_STALL_TIMEOUT_MS } from "../../restoreConstants";
import { TURN_PAUSED_PART_TYPE } from "../../turnPause";
import type { ExpertIdentity } from "../../useExpertMap";
import { ChatMinimap } from "../ChatMinimap/ChatMinimap";
import { getWorkRunMetadata } from "../WorkCard/helpers";
import { TailSpacer } from "./components/TailSpacer";
import { VirtualTranscript } from "./components/VirtualTranscript/VirtualTranscript";
import type { TranscriptRow } from "./components/VirtualTranscript/helpers";
import type { JumpToMessage } from "./components/VirtualTranscript/useVirtualTranscript";
import { MessageRow } from "./components/MessageRow/MessageRow";
import { PendingAnswerContexts } from "./components/PendingAnswerContexts";
import { ThreadHeader, type FloatingControls } from "./components/ThreadHeader";
import {
  PENDING_UPLOAD_MESSAGE_ID,
  PendingUploadMessage,
} from "./components/PendingUploadMessage";
import { ThinkingIndicator } from "./components/ThinkingIndicator";
import { isSessionOpeningMessage, type SentFrom } from "../../sentFrom";
import type { PendingUploadSend } from "../../copilotStreamStore";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ChatErrorBanner } from "./components/ChatErrorBanner/ChatErrorBanner";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

interface Props {
  messages: ChatMessage[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  isRestoringActiveSession?: boolean;
  restoreStatusMessage?: string | null;
  /** ISO start time of the active backend turn. Seeds the elapsed-time
   *  counter so restored turns show honest age instead of counting from
   *  zero on every fresh mount. */
  activeStreamStartedAt?: string | null;
  sessionID?: string | null;
  /** Session-level lifecycle: ``"idle" | "queued" | "running"``.
   *  The Queued badge anchors on the latest user message iff this is
   *  ``"queued"``. */
  sessionChatStatus?: string;
  hasMoreMessages?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
  onRetry?: () => void;
  turnStats?: TurnStatsMap;
  /** Pending queued messages waiting to be injected, shown at the end of chat. */
  queuedMessages?: string[];
  /** A just-sent message whose local attachments are still uploading. It is
   *  not in `messages` yet (the SDK only pushes it once `sendMessage` runs),
   *  so it renders as a placeholder bubble with an "Uploading files…"
   *  status in the thinking indicator's usual spot. */
  pendingSend?: PendingUploadSend | null;
  /** Extra bottom padding (px) applied to the scrollable message list so
   *  overlays pinned above the input area (e.g. the usage-limit card) can
   *  sit over the last message without permanently obscuring it. */
  bottomContentPadding?: number;
  /** Public-viewer mode: render messages exactly as the owner sees them,
   *  but hide every interactive affordance that depends on auth — feedback
   *  buttons, TTS, queue/streaming indicators, load-more, retry, pending-
   *  review banners, queued-message strip.  Anonymous viewers of a shared
   *  chat get the rich renderer without any controls that would 401. */
  readOnly?: boolean;
  /** URL→file-ID matcher used to decide whether a ``FileUIPart`` becomes
   *  an ArtifactCard.  Owner side defaults to the workspace-file URL
   *  shape; the public viewer passes a per-token pattern so its file
   *  URLs match without loosening the default. */
  filePattern?: RegExp;
  /** Override the URL emitted when rewriting ``workspace://`` references
   *  in markdown prose AND when building inline artifact source URLs.
   *  The public viewer passes a token-aware builder. */
  fileUrlBuilder?: (fileId: string) => string;
  /** Expert identity for expert-scoped sessions: drives the thread header
   *  and the assistant avatar/name. Null/undefined = default header. */
  expertIdentity?: ExpertIdentity | null;
  /** The roster is still loading for an expert-scoped session, so the
   *  header must not yet claim the thread is Otto's. */
  isResolvingExpertIdentity?: boolean;
  /** Where this thread's opening task came from (session-level delegation
   *  metadata). Shown on the row that opened the thread (DB sequence 0) when
   *  that row carries no provenance of its own. */
  sessionSentFrom?: SentFrom | null;
  /** Which layout floats its controls over the chat's top-left corner on
   *  small viewports (see ThreadHeader). */
  floatingControls?: FloatingControls;
  /** Compact thread for side panels: smaller text, tighter bubbles and
   *  spacing. */
  variant?: "default" | "compact";
  /** Hosts that already name the thread (e.g. the expert chat drawer)
   *  turn the floating identity chip off. */
  showThreadHeader?: boolean;
}

// Max consecutive auto-triggered loads where the container remains
// non-scrollable afterwards. Prevents chewing through history on
// sessions whose every page collapses below viewport height. The
// manual "Load older messages" button always remains clickable.
const MAX_AUTO_FILL_ROUNDS = 3;

/**
 * Triggers `onLoadMore` when scrolled near the top, preserves the
 * user's scroll position after older messages are prepended, and
 * exposes a manual "Load older messages" button as a fallback when
 * auto-fill backs off or the container isn't scrollable.
 *
 * Scroll preservation works by:
 * 1. Capturing `scrollHeight` / `scrollTop` just before `onLoadMore`
 *    (synchronous, before React re-renders).
 * 2. Restoring `scrollTop` in a `useLayoutEffect` keyed on
 *    `messageCount` so it only fires when messages actually change
 *    (not on intermediate renders like the loading-spinner toggle).
 */
export function LoadMoreSentinel({
  hasMore,
  isLoading,
  messageCount,
  onLoadMore,
}: {
  hasMore: boolean;
  isLoading: boolean;
  messageCount: number;
  onLoadMore: () => void;
}) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const onLoadMoreRef = useRef(onLoadMore);
  useInsertionEffect(() => {
    onLoadMoreRef.current = onLoadMore;
  });
  // Pre-mutation scroll snapshot, written synchronously before onLoadMore
  const scrollSnapshotRef = useRef({ scrollHeight: 0, scrollTop: 0 });
  // Consecutive auto-triggered loads that left the container non-scrollable
  const autoFillRoundsRef = useRef(0);
  // True if the pending load was triggered by the observer (not the button)
  const autoTriggeredRef = useRef(false);
  // Same-frame re-entry guard — the parent's `isLoading` flag lags by a
  // render, so the observer or button could otherwise fire a duplicate
  // load and overwrite the captured scroll snapshot before the first
  // load settles.
  const loadPendingRef = useRef(false);
  const { scrollRef } = useStickToBottomContext();

  useEffect(() => {
    if (!isLoading) loadPendingRef.current = false;
  }, [isLoading]);

  function captureAndLoad(fromObserver: boolean) {
    if (loadPendingRef.current) return;
    loadPendingRef.current = true;
    const el = scrollRef.current;
    if (el) {
      scrollSnapshotRef.current = {
        scrollHeight: el.scrollHeight,
        scrollTop: el.scrollTop,
      };
    }
    autoTriggeredRef.current = fromObserver;
    onLoadMoreRef.current();
  }

  useEffect(() => {
    if (!sentinelRef.current || !hasMore || isLoading) return;
    if (autoFillRoundsRef.current >= MAX_AUTO_FILL_ROUNDS) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        if (autoFillRoundsRef.current >= MAX_AUTO_FILL_ROUNDS) return;
        captureAndLoad(true);
      },
      { rootMargin: "200px 0px 0px 0px" },
    );
    observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [hasMore, isLoading, scrollRef]);

  // After React commits new DOM nodes (prepended messages), adjust
  // scrollTop so the user stays at the same visual position.
  // Keyed on messageCount so it only fires when messages actually
  // change — NOT on intermediate renders (loading spinner, etc.)
  // that would consume the snapshot too early.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const { scrollHeight: prevHeight, scrollTop: prevTop } =
      scrollSnapshotRef.current;
    if (!el || prevHeight === 0) return;
    const delta = el.scrollHeight - prevHeight;
    if (delta > 0) {
      el.scrollTop = prevTop + delta;
    }
    // Reset the auto-fill backoff whenever the container becomes
    // scrollable (from any load), so a manual button click can unstick
    // auto-fill after it has hit the cap. Only count non-scrollable
    // outcomes against the cap when the load itself was auto-triggered.
    if (el.scrollHeight > el.clientHeight) {
      autoFillRoundsRef.current = 0;
    } else if (autoTriggeredRef.current) {
      autoFillRoundsRef.current += 1;
    }
    scrollSnapshotRef.current = { scrollHeight: 0, scrollTop: 0 };
    autoTriggeredRef.current = false;
  }, [messageCount, scrollRef]);

  return (
    <div
      ref={sentinelRef}
      className="flex flex-col items-center justify-center gap-2 py-1"
    >
      {isLoading ? (
        <LoadingSpinner
          data-testid="load-more-spinner"
          // TODO(tokens): no exact role for text-neutral-400
          className="h-5 w-5 text-neutral-400"
        />
      ) : (
        hasMore && (
          <Button
            variant="ghost"
            size="small"
            onClick={() => captureAndLoad(false)}
          >
            Load older messages
          </Button>
        )
      )}
    </div>
  );
}

export function ChatMessagesContainer({
  messages: allMessages,
  status,
  error,
  isLoading,
  isRestoringActiveSession,
  restoreStatusMessage,
  activeStreamStartedAt,
  sessionID,
  sessionChatStatus,
  hasMoreMessages,
  isLoadingMore,
  onLoadMore,
  onRetry,
  turnStats,
  queuedMessages,
  pendingSend,
  bottomContentPadding,
  readOnly = false,
  filePattern,
  fileUrlBuilder,
  expertIdentity,
  isResolvingExpertIdentity = false,
  sessionSentFrom = null,
  floatingControls,
  variant = "default",
  showThreadHeader = true,
}: Props) {
  const isCompact = variant === "compact";
  const messages = useMemo(
    () => hideKickoffMessages(allMessages),
    [allMessages],
  );
  // Hide the container for one frame when messages first load so
  // StickToBottom can scroll to the bottom before the user sees it.
  const [settled, setSettled] = useState(false);
  const [prevSessionID, setPrevSessionID] = useState(sessionID);
  if (sessionID !== prevSessionID) {
    setPrevSessionID(sessionID);
    if (settled) setSettled(false);
  }
  const messagesReady = messages.length > 0 || !isLoading;
  useEffect(() => {
    if (settled || !messagesReady) return;
    const raf = requestAnimationFrame(() => setSettled(true));
    return () => cancelAnimationFrame(raf);
  }, [settled, messagesReady]);
  // opacity-0 only during the single frame between messages arriving and scroll settling
  const hideForScroll = messagesReady && !settled;

  // Rendered rows, not the array `useChat` owns: a turn whose pending buffer
  // was drained mid-stream renders as chain → follow-up bubble → chain, while
  // the underlying message stays whole. See `splitMessagesAtDrainHints`.
  // These lists stay memos with the compiler on: the transcript rows reach
  // the windowed transcript's hooks, which the compiler skips and which key
  // their row measurements on the rows' identity.
  const renderRows = useMemo(
    () => splitMessagesAtDrainHints(messages),
    [messages],
  );
  const messageIndexById = buildMessageIndexById(messages);
  const transcriptRows = useMemo(
    () => toTranscriptRows(renderRows),
    [renderRows],
  );
  // The minimap scrolls to rows by their DOM node; one the windowed
  // transcript has unmounted is reached through the transcript instead.
  const jumpRef = useRef<JumpToMessage | null>(null);
  const [jumpToUnmountedMessage] = useState(
    () => (messageID: string) => jumpRef.current?.(messageID),
  );
  const lastMessage = messages[messages.length - 1];
  const showPendingSend = !readOnly && !!pendingSend;
  // Read off the rendered rows: a fallback follow-up row the split drops in
  // favour of the drain-point bubble has no element to anchor the tail on.
  // While a send is still uploading, the placeholder is the last user row.
  const lastUserMessageID = showPendingSend
    ? PENDING_UPLOAD_MESSAGE_ID
    : (renderRows.findLast((row) => row.role === "user")?.id ?? null);
  const reviewTargets = extractReviewTargets(messages);

  // The backend appends a persisted error marker to ``session.messages`` AND
  // yields a ``StreamError`` SSE event on final-failure paths. Both surface
  // the same error string — the marker becomes an in-line ErrorCard bubble,
  // the SSE event sets ``error`` on ``useChat``. Without dedup, the user sees
  // the same error twice. Suppress the trailing banner whenever the last
  // assistant message already carries the marker.
  const lastAssistantHasErrorMarker = hasTrailingErrorMarker(messages);

  const hasInflight = (() => {
    if (lastMessage?.role !== "assistant") return false;
    // Ignore bookkeeping parts — none of them counts as "real" content that
    // hides the Thinking indicator. See `isBookkeepingPart` for the list.
    // A drain hint newer than the last content part is the one exception:
    // the follow-up just landed and nothing has been produced past it, so
    // the text or tool above it is settled and Thinking is the only sign
    // the assistant picked the follow-up up.
    let lastIndex = lastMessage.parts.length - 1;
    while (lastIndex >= 0 && isBookkeepingPart(lastMessage.parts[lastIndex])) {
      if (lastMessage.parts[lastIndex].type === PENDING_DRAINED_PART_TYPE)
        return false;
      lastIndex--;
    }
    if (lastIndex < 0) return false;

    const lastPart = lastMessage.parts[lastIndex];

    if (lastPart.type === "text" && lastPart.text.trim().length > 0)
      return true;

    // Reasoning chunks stream before the final text — while they have
    // rendered content, the "Thinking..." indicator should give way to the
    // reasoning view (e.g. Perplexity deep research streams minutes of
    // reasoning before any answer text).
    if (lastPart.type === "reasoning" && lastPart.text.trim().length > 0)
      return true;

    // step-start is a turn boundary emitted right before the next tool or
    // text chunk. Treat it as inflight so the bubble transitions straight
    // into the next part without flashing back to "Thinking...".
    if (lastPart.type === "step-start") return true;

    // A pause is how the turn ends, so nothing more is coming to wait on.
    if (lastPart.type === TURN_PAUSED_PART_TYPE) return true;

    if (
      lastPart.type.startsWith(TOOL_PART_PREFIX) &&
      "state" in lastPart &&
      (lastPart.state === "input-streaming" ||
        lastPart.state === "input-available")
    )
      return true;

    return false;
  })();

  const isActivelyStreaming = status === "streaming" || status === "submitted";
  useCostReplyMoment({
    messages,
    isStreaming: isActivelyStreaming,
    enabled: !readOnly,
  });

  // A live compaction narrates itself via CompactionCard — the generic
  // Thinking indicator must not stack a second spinner and timer under it.
  // The `rebuilding` phase arrives after the tool row has already closed,
  // so `hasInflight` alone cannot cover that window.
  const liveCompactionPhase =
    isActivelyStreaming && lastMessage?.role === "assistant"
      ? getLatestCompactionPhase(lastMessage.parts)
      : null;

  // Suppressed during active-session restore so the ThinkingIndicator and
  // the "Retrieving latest messages" spinner can't both render — the
  // restore spinner wins until real content arrives (see the
  // ``hasConnectedThisMountRef`` latch in useCopilotStream for why).
  const showThinking =
    !isRestoringActiveSession &&
    liveCompactionPhase === null &&
    (status === "submitted" || (status === "streaming" && !hasInflight));
  // Only the start time lives here: the indicator ticks its own label, so
  // the transcript does not re-render every second for the whole turn.
  const streamClock = useElapsedAnchor(
    isActivelyStreaming,
    activeStreamStartedAt,
  );
  const indicator = (
    <ThinkingIndicator
      active={showThinking}
      startedAtMs={streamClock.startedAtMs}
    />
  );
  // Public viewers of a shared chat never see a live turn, so the indicator
  // stays out of the transcript for them wherever it would render.
  const showIndicator = !readOnly && showThinking;
  const [showRestoreFallback, setShowRestoreFallback] = useState(false);
  useEffect(() => {
    if (!isRestoringActiveSession) {
      setShowRestoreFallback(false);
      return;
    }
    const timer = setTimeout(
      () => setShowRestoreFallback(true),
      RESTORE_STALL_TIMEOUT_MS,
    );
    return () => clearTimeout(timer);
  }, [isRestoringActiveSession]);
  const restoreClock = useElapsedAnchor(showRestoreFallback);

  function renderRow({ message, rowIndex }: TranscriptRow) {
    if (getRowHeldCallKind(message) === "answered") {
      return (
        <Text
          variant="small"
          className="py-1 text-center text-foreground-tertiary"
        >
          {answeredLabel(renderRows, rowIndex)}
        </Text>
      );
    }

    const isLastRow = rowIndex === renderRows.length - 1;
    const isAssistant = message.role === "assistant";
    const isLastAssistant = isLastRow && isAssistant;
    const isCurrentlyStreaming = isLastAssistant && isActivelyStreaming;
    const nextRow = renderRows[rowIndex + 1];
    // A segment that only runs up to a mid-turn drain is never the end
    // of its turn — the same backend turn continues under the follow-up
    // bubble, so the stats bar and the assistant actions belong to the
    // last segment alone.
    const isLastInTurn =
      isAssistant &&
      !isMidTurnSegmentRow(message) &&
      (!nextRow || nextRow.role === "user");
    const showsTurnStats = isLastInTurn && !isCurrentlyStreaming;

    return (
      <MessageRow
        message={message}
        isLastAssistant={isLastAssistant}
        isCurrentlyStreaming={isCurrentlyStreaming}
        isLastInTurn={isLastInTurn}
        turnStat={turnStats?.get(message.id)}
        turnMessages={
          showsTurnStats
            ? turnMessagesForRow(messages, message, messageIndexById)
            : undefined
        }
        elapsedSeconds={isLastRow ? streamClock.frozenSeconds : undefined}
        indicator={isLastAssistant && showIndicator ? indicator : null}
        isSessionQueued={
          message.role === "user" && sessionChatStatus === "queued"
        }
        sessionSentFrom={
          isSessionOpeningMessage(message) ? sessionSentFrom : null
        }
        sessionID={sessionID}
        readOnly={readOnly}
        compact={isCompact}
        filePattern={filePattern}
        fileUrlBuilder={fileUrlBuilder}
        onRetry={isLastAssistant ? onRetry : undefined}
      />
    );
  }

  const thread = (
    <>
      {showThreadHeader && (
        <ThreadHeader
          expertIdentity={expertIdentity}
          isResolvingExpertIdentity={isResolvingExpertIdentity}
          floatingControls={floatingControls}
        />
      )}
      {!isCompact && (
        <ChatMinimap
          messages={messages}
          onJumpToUnmounted={jumpToUnmountedMessage}
        />
      )}
      <Conversation
        key={sessionID ?? "new"}
        resize="instant"
        className={
          "min-h-0 flex-1 " +
          (hideForScroll
            ? "opacity-0"
            : "opacity-100 transition-opacity duration-100 ease-out")
        }
      >
        <ConversationContent
          className={cn(
            "mx-auto flex min-h-full w-full max-w-3xl flex-1 flex-col gap-6 px-4 pb-4 pt-14 md:px-6",
            isCompact && "gap-4 px-4 pt-4",
            showThreadHeader && floatingControls && "max-lg:pt-20",
            !showThreadHeader && "pt-4",
          )}
          style={
            bottomContentPadding
              ? { paddingBottom: bottomContentPadding + 24 }
              : undefined
          }
        >
          {!readOnly && hasMoreMessages && onLoadMore && (
            <LoadMoreSentinel
              hasMore={hasMoreMessages}
              isLoading={!!isLoadingMore}
              messageCount={messages.length}
              onLoadMore={onLoadMore}
            />
          )}
          {isLoading &&
            messages.length === 0 &&
            !isRestoringActiveSession &&
            !showPendingSend && (
              <div className="flex flex-1 items-center justify-center">
                <LoadingSpinner
                  // TODO(tokens): no exact role for text-neutral-600
                  className="text-neutral-600"
                />
              </div>
            )}
          {transcriptRows.length > 0 && (
            <VirtualTranscript
              rows={transcriptRows}
              renderRow={renderRow}
              lastUserMessageID={lastUserMessageID}
              compact={isCompact}
              jumpRef={jumpRef}
            />
          )}
          {showPendingSend && pendingSend && (
            <PendingUploadMessage
              pendingSend={pendingSend}
              isCompact={isCompact}
            />
          )}
          {showIndicator && lastMessage?.role !== "assistant" && (
            <Message
              from="assistant"
              className="duration-300 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
            >
              <MessageContent className="text-body-lg leading-relaxed">
                {indicator}
              </MessageContent>
            </Message>
          )}
          {!readOnly && isRestoringActiveSession && (
            <Message from="assistant">
              <MessageContent
                // TODO(tokens): no exact role for text-slate-900
                className="text-body-lg leading-relaxed text-slate-900"
              >
                {showRestoreFallback ? (
                  <div
                    // TODO(tokens): no exact role for text-slate-500
                    className="flex flex-col gap-1 text-sm text-slate-500"
                  >
                    <ThinkingIndicator
                      active
                      startedAtMs={restoreClock.startedAtMs}
                      statusMessage={restoreStatusMessage ?? "Reconnecting…"}
                    />
                    <span
                      // TODO(tokens): no exact role for text-slate-400
                      className="pl-6 text-xs text-slate-400"
                    >
                      Still syncing the latest progress.
                    </span>
                  </div>
                ) : (
                  <div
                    // TODO(tokens): no exact role for text-slate-500
                    className="flex items-center gap-2 text-sm text-slate-500"
                  >
                    <LoadingSpinner
                      // TODO(tokens): no exact role for text-neutral-500
                      className="h-4 w-4 text-neutral-500"
                    />
                    <span>Retrieving latest messages</span>
                  </div>
                )}
              </MessageContent>
            </Message>
          )}
          {!readOnly && reviewTargets.runs.length > 0 && (
            <CopilotPendingReviews runs={reviewTargets.runs} />
          )}
          {!readOnly && sessionID && (
            <CopilotPendingReviews
              chatSessionId={sessionID}
              pollWhileEmpty={reviewTargets.chat}
              refetchKey={countHeldCalls(messages)}
              expertName={expertIdentity?.name ?? null}
              expert={expertIdentity ?? null}
            />
          )}
          {!readOnly &&
            queuedMessages?.map((msg, idx) => (
              <Message key={idx} from="user">
                <MessageContent
                  className={cn(
                    "flex flex-col gap-1 border border-dashed border-border-strong bg-surface-hover px-4 py-2.5 text-foreground-strong opacity-60",
                    isCompact
                      ? "rounded-xl text-sm leading-6"
                      : "rounded-3xl text-body-lg leading-relaxed",
                  )}
                >
                  <span>
                    <CredentialMentionText text={msg} />
                  </span>
                  <span
                    // TODO(tokens): no exact role for text-slate-500
                    className="flex items-center gap-1 text-xs text-slate-500"
                  >
                    <Icon icon={Clock01Icon} className="size-3" />
                    Queued
                  </span>
                </MessageContent>
              </Message>
            ))}
          {!readOnly && error && !lastAssistantHasErrorMarker && (
            <ChatErrorBanner error={error} />
          )}
          <TailSpacer
            messageID={lastUserMessageID}
            bottomInset={bottomContentPadding ?? 0}
          />
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
    </>
  );

  // Every chat surface renders through here, so the pending-answer state
  // (clarifying questions, a hire's onboarding, held calls) lives here too
  // instead of in each host. Read-only viewers receive saved answers only.
  return (
    <PendingAnswerContexts messages={allMessages} readOnly={readOnly}>
      {thread}
    </PendingAnswerContexts>
  );
}

// A run-post renders as a WorkCard inside the row, whatever else its
// metadata carries.
function getRowHeldCallKind(message: ChatMessage) {
  return getWorkRunMetadata(message.metadata)
    ? null
    : getHeldCallRowKind(message.metadata);
}

// A held call's result row renders nothing, so it stays out of the list: an
// empty row would still take a gap.
function toTranscriptRows(renderRows: ChatMessage[]): TranscriptRow[] {
  const rows: TranscriptRow[] = [];
  renderRows.forEach((message, rowIndex) => {
    if (getRowHeldCallKind(message) !== "result") {
      rows.push({ message, rowIndex });
    }
  });
  return rows;
}

/** Whether the last assistant message's last text part is a persisted error
 *  marker (the in-line ErrorCard bubble). */
function hasTrailingErrorMarker(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "assistant") continue;
    for (let j = msg.parts.length - 1; j >= 0; j--) {
      const part = msg.parts[j];
      if (part.type !== "text") continue;
      const { markerType } = parseSpecialMarkers(part.text);
      return markerType === "error" || markerType === "retryable_error";
    }
    return false;
  }
  return false;
}
