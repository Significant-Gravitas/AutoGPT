import { useMemo, useState } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageActions,
  MessageContent,
} from "@/components/ai-elements/message";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Clock } from "@phosphor-icons/react";
import { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useLayoutEffect, useRef } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { TurnStatsBar } from "../JobStatsBar/TurnStatsBar";
import { useElapsedTimer } from "../JobStatsBar/useElapsedTimer";
import { CopilotPendingReviews } from "../CopilotPendingReviews/CopilotPendingReviews";
import type { TurnStatsMap } from "../../helpers/convertChatSessionToUiMessages";
import {
  buildRenderSegments,
  getTurnMessages,
  type MessagePart,
  type RenderSegment,
  parseSpecialMarkers,
  splitReasoningAndResponse,
} from "./helpers";
import { RESTORE_STALL_TIMEOUT_MS } from "../../restoreConstants";
import { AssistantMessageActions } from "./components/AssistantMessageActions";
import { CopyButton } from "./components/CopyButton";
import { CollapsedToolGroup } from "./components/CollapsedToolGroup";
import { MessageAttachments } from "./components/MessageAttachments";
import { MessagePartRenderer } from "./components/MessagePartRenderer";
import { QueueBadge } from "./components/QueueBadge";
import { StepsCollapse } from "./components/StepsCollapse";
import { ThinkingIndicator } from "./components/ThinkingIndicator";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
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
  /** Extra bottom padding (px) applied to the scrollable message list so
   *  overlays pinned above the input area (e.g. the usage-limit card) can
   *  sit over the last message without permanently obscuring it. */
  bottomContentPadding?: number;
}

function renderSegments(
  segments: RenderSegment[],
  messageID: string,
  onRetry?: () => void,
): React.ReactNode[] {
  return segments.map((seg, segIdx) => {
    if (seg.kind === "collapsed-group") {
      return <CollapsedToolGroup key={`group-${segIdx}`} parts={seg.parts} />;
    }
    return (
      <MessagePartRenderer
        key={`${messageID}-${seg.index}`}
        part={seg.part}
        messageID={messageID}
        partIndex={seg.index}
        onRetry={onRetry}
      />
    );
  });
}

/**
 * Extract graph_exec_id from tool outputs that need review.
 * Handles both:
 * - run_block ReviewRequiredResponse (has graph_exec_id directly)
 * - run_agent ExecutionStartedResponse with status "REVIEW" (has execution_id)
 */
function extractGraphExecId(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): string | null {
  // Scan backwards — the most recent review output has the ID
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    for (const part of msg.parts) {
      if ("output" in part && part.output) {
        const out =
          typeof part.output === "string"
            ? (() => {
                try {
                  return JSON.parse(part.output);
                } catch {
                  return null;
                }
              })()
            : part.output;
        if (out && typeof out === "object") {
          // run_block: ReviewRequiredResponse has graph_exec_id
          if ("graph_exec_id" in out) {
            return (out as { graph_exec_id: string }).graph_exec_id;
          }
          // run_agent: ExecutionStartedResponse with status "REVIEW"
          if (
            "execution_id" in out &&
            "status" in out &&
            (out as { status: string }).status === "REVIEW"
          ) {
            return (out as { execution_id: string }).execution_id;
          }
        }
      }
    }
  }
  return null;
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
  onLoadMoreRef.current = onLoadMore;
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
  messages,
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
  bottomContentPadding,
}: Props) {
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

  const lastMessage = messages[messages.length - 1];
  const graphExecId = useMemo(() => extractGraphExecId(messages), [messages]);

  // The backend appends a persisted error marker to ``session.messages`` AND
  // yields a ``StreamError`` SSE event on final-failure paths. Both surface
  // the same error string — the marker becomes an in-line ErrorCard bubble,
  // the SSE event sets ``error`` on ``useChat``. Without dedup, the user sees
  // the same error twice. Suppress the trailing banner whenever the last
  // assistant message already carries the marker.
  const lastAssistantHasErrorMarker = useMemo(() => {
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
  }, [messages]);

  const hasInflight = (() => {
    if (lastMessage?.role !== "assistant") return false;
    // Ignore bookkeeping parts. data-cursor is legacy resume metadata and
    // data-status is transient copy for the Thinking indicator; neither
    // counts as "real" content that hides the indicator.
    const parts = lastMessage.parts.filter(
      (p) => p.type !== "data-cursor" && p.type !== "data-status",
    );
    if (parts.length === 0) return false;

    const lastPart = parts[parts.length - 1];

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

    if (
      lastPart.type.startsWith(TOOL_PART_PREFIX) &&
      "state" in lastPart &&
      (lastPart.state === "input-streaming" ||
        lastPart.state === "input-available")
    )
      return true;

    return false;
  })();

  // Surface the latest `data-status` message from the live assistant when
  // the Thinking indicator is up — but only if it wasn't invalidated by a
  // more recent content part (in which case the model has moved on and the
  // status is stale).
  const latestStatusMessage = (() => {
    if (lastMessage?.role !== "assistant") return null;
    for (let i = lastMessage.parts.length - 1; i >= 0; i--) {
      const part = lastMessage.parts[i];
      if (part.type === "data-cursor") continue;
      if (part.type === "data-status") {
        const data = (part as { data?: { message?: unknown } }).data;
        return typeof data?.message === "string" ? data.message : null;
      }
      // Any other part = the model has produced output past the status.
      return null;
    }
    return null;
  })();

  // Suppressed during active-session restore so the ThinkingIndicator and
  // the "Retrieving latest messages" spinner can't both render — the
  // restore spinner wins until real content arrives (see the
  // ``hasConnectedThisMountRef`` latch in useCopilotStream for why).
  const showThinking =
    !isRestoringActiveSession &&
    (status === "submitted" || (status === "streaming" && !hasInflight));
  const isActivelyStreaming = status === "streaming" || status === "submitted";
  const { elapsedSeconds } = useElapsedTimer(
    isActivelyStreaming,
    activeStreamStartedAt,
  );
  const indicator = (
    <ThinkingIndicator
      active={showThinking}
      elapsedSeconds={elapsedSeconds}
      statusMessage={latestStatusMessage}
    />
  );
  const showIndicator = showThinking;
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
  const { elapsedSeconds: restoreElapsedSeconds } =
    useElapsedTimer(showRestoreFallback);

  // Freeze elapsed time when streaming ends so TurnStatsBar shows the final value.
  // Reset when a new streaming turn begins.
  const frozenElapsedRef = useRef(0);
  const wasStreamingRef = useRef(false);
  useEffect(() => {
    if (isActivelyStreaming) {
      if (!wasStreamingRef.current) {
        frozenElapsedRef.current = 0;
      }
      if (elapsedSeconds > 0) {
        frozenElapsedRef.current = elapsedSeconds;
      }
    }
    wasStreamingRef.current = isActivelyStreaming;
  });

  return (
    <Conversation
      key={sessionID ?? "new"}
      resize={settled ? "smooth" : "instant"}
      className={
        "min-h-0 flex-1 " +
        (hideForScroll
          ? "opacity-0"
          : "opacity-100 transition-opacity duration-100 ease-out")
      }
    >
      <ConversationContent
        className="flex min-h-full flex-1 flex-col gap-6 px-3 py-6"
        style={
          bottomContentPadding
            ? { paddingBottom: bottomContentPadding + 24 }
            : undefined
        }
      >
        {hasMoreMessages && onLoadMore && (
          <LoadMoreSentinel
            hasMore={hasMoreMessages}
            isLoading={!!isLoadingMore}
            messageCount={messages.length}
            onLoadMore={onLoadMore}
          />
        )}
        {isLoading && messages.length === 0 && !isRestoringActiveSession && (
          <div className="flex flex-1 items-center justify-center">
            <LoadingSpinner className="text-neutral-600" />
          </div>
        )}
        {messages.map((message, messageIndex) => {
          const isLastAssistant =
            messageIndex === messages.length - 1 &&
            message.role === "assistant";

          const isCurrentlyStreaming =
            isLastAssistant &&
            (status === "streaming" || status === "submitted");

          const isAssistant = message.role === "assistant";

          const nextMessage = messages[messageIndex + 1];
          const isLastInTurn =
            isAssistant &&
            messageIndex <= messages.length - 1 &&
            (!nextMessage || nextMessage.role === "user");
          // data-cursor / data-status parts are internal bookkeeping —
          // strip them before any render/split logic so they never reach
          // the user UI. data-status surfaces via ThinkingIndicator.
          const renderableParts = message.parts.filter(
            (p) => p.type !== "data-cursor" && p.type !== "data-status",
          );
          const textParts = renderableParts.filter(
            (p): p is Extract<typeof p, { type: "text" }> => p.type === "text",
          );
          const lastTextPart = textParts[textParts.length - 1];
          const markerType =
            lastTextPart !== undefined
              ? parseSpecialMarkers(lastTextPart.text).markerType
              : null;
          const hasErrorMarker =
            markerType === "error" || markerType === "retryable_error";
          const showActions =
            isLastInTurn &&
            !isCurrentlyStreaming &&
            textParts.length > 0 &&
            !hasErrorMarker;

          const fileParts = renderableParts.filter(
            (p): p is FileUIPart => p.type === "file",
          );

          // For finalized assistant messages, split into reasoning + response.
          // During streaming, show everything normally with tool collapsing.
          const isFinalized =
            message.role === "assistant" && !isCurrentlyStreaming;
          const { reasoning, response } = isFinalized
            ? splitReasoningAndResponse(renderableParts)
            : { reasoning: [] as MessagePart[], response: renderableParts };
          const hasReasoning = reasoning.length > 0;

          // Note: when interactive tools are pinned from reasoning into response,
          // this index approximates their position (used only for React keys).
          const responseStartIndex = renderableParts.length - response.length;
          const responseSegments =
            message.role === "assistant"
              ? buildRenderSegments(response, responseStartIndex)
              : null;
          const reasoningSegments = hasReasoning
            ? buildRenderSegments(reasoning, 0)
            : null;

          return (
            <Message from={message.role} key={message.id}>
              <MessageContent
                className={
                  "text-[1rem] leading-relaxed " +
                  "group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0] " +
                  "group-[.is-user]:[&_h1]:text-lg group-[.is-user]:[&_h1]:font-semibold group-[.is-user]:[&_h2]:text-lg group-[.is-user]:[&_h2]:font-semibold group-[.is-user]:[&_h3]:text-lg group-[.is-user]:[&_h3]:font-semibold group-[.is-user]:[&_h4]:text-lg group-[.is-user]:[&_h4]:font-semibold group-[.is-user]:[&_h5]:text-lg group-[.is-user]:[&_h5]:font-semibold group-[.is-user]:[&_h6]:text-lg group-[.is-user]:[&_h6]:font-semibold " +
                  "group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900"
                }
              >
                {hasReasoning && reasoningSegments && (
                  <StepsCollapse>
                    {renderSegments(reasoningSegments, message.id)}
                  </StepsCollapse>
                )}
                {responseSegments
                  ? renderSegments(
                      responseSegments,
                      message.id,
                      isLastAssistant ? onRetry : undefined,
                    )
                  : renderableParts.map((part, i) => (
                      <MessagePartRenderer
                        key={`${message.id}-${i}`}
                        part={part}
                        messageID={message.id}
                        partIndex={i}
                        onRetry={isLastAssistant ? onRetry : undefined}
                      />
                    ))}
                {isLastInTurn && !isCurrentlyStreaming && (
                  <TurnStatsBar
                    turnMessages={getTurnMessages(messages, messageIndex)}
                    elapsedSeconds={
                      messageIndex === messages.length - 1
                        ? frozenElapsedRef.current
                        : undefined
                    }
                    stats={turnStats?.get(message.id)}
                  />
                )}
                {isLastAssistant && showIndicator && indicator}
              </MessageContent>
              {message.role === "user" &&
                sessionChatStatus === "queued" &&
                (() => {
                  const stats = turnStats?.get(message.id);
                  if (!stats?.isLatestUserMessage) {
                    return null;
                  }
                  return (
                    <MessageActions
                      className="mt-1 items-center justify-end gap-1.5"
                      data-testid="queue-status-row"
                    >
                      <QueueBadge sessionID={sessionID ?? null} />
                    </MessageActions>
                  );
                })()}
              {message.role === "user" && textParts.length > 0 && (
                <MessageActions className="mt-1 items-center justify-end gap-2 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
                  {(() => {
                    const createdAt = turnStats?.get(message.id)?.createdAt;
                    if (!createdAt) return null;
                    const date = new Date(createdAt);
                    if (Number.isNaN(date.getTime())) return null;
                    return (
                      <span className="text-[11px] tabular-nums text-neutral-500">
                        {date.toLocaleString(undefined, {
                          dateStyle: "medium",
                          timeStyle: "short",
                        })}
                      </span>
                    );
                  })()}
                  <CopyButton text={textParts.map((p) => p.text).join("\n")} />
                </MessageActions>
              )}
              {fileParts.length > 0 && (
                <MessageAttachments
                  files={fileParts}
                  isUser={message.role === "user"}
                />
              )}
              {showActions && (
                <AssistantMessageActions
                  message={message}
                  sessionID={sessionID ?? null}
                />
              )}
            </Message>
          );
        })}
        {showIndicator && lastMessage?.role !== "assistant" && (
          <Message from="assistant">
            <MessageContent className="text-[1rem] leading-relaxed">
              {indicator}
            </MessageContent>
          </Message>
        )}
        {isRestoringActiveSession && (
          <Message from="assistant">
            <MessageContent className="text-[1rem] leading-relaxed text-slate-900">
              {showRestoreFallback ? (
                <div className="flex flex-col gap-1 text-sm text-slate-500">
                  <ThinkingIndicator
                    active
                    elapsedSeconds={restoreElapsedSeconds}
                    statusMessage={
                      restoreStatusMessage ?? "Reconnecting to live stream..."
                    }
                  />
                  <span className="pl-6 text-xs text-slate-400">
                    Still syncing the latest progress.
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <LoadingSpinner className="h-4 w-4 text-neutral-500" />
                  <span>Retrieving latest messages</span>
                </div>
              )}
            </MessageContent>
          </Message>
        )}
        {graphExecId && <CopilotPendingReviews graphExecId={graphExecId} />}
        {queuedMessages?.map((msg, idx) => (
          <Message key={idx} from="user">
            <MessageContent className="flex flex-col gap-1 rounded-xl border border-dashed border-purple-400 bg-purple-100 px-3 py-2.5 text-[1rem] leading-relaxed text-slate-900 opacity-60 [border-bottom-right-radius:0]">
              <span>{msg}</span>
              <span className="flex items-center gap-1 text-xs text-slate-500">
                <Clock className="size-3" weight="bold" />
                Queued
              </span>
            </MessageContent>
          </Message>
        ))}
        {error && !lastAssistantHasErrorMarker && (
          <details className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
            <summary className="cursor-pointer font-medium">
              The assistant encountered an error. Please try sending your
              message again.
            </summary>
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs text-red-600">
              {error instanceof Error ? error.message : String(error)}
            </pre>
          </details>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
