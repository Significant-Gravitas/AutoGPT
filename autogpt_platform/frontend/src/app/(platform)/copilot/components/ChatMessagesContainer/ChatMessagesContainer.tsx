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
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Clock } from "@phosphor-icons/react";
import { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useLayoutEffect, useRef } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { TurnStatsBar } from "../JobStatsBar/TurnStatsBar";
import { useElapsedTimer } from "../JobStatsBar/useElapsedTimer";
import { CopilotPendingReviews } from "../CopilotPendingReviews/CopilotPendingReviews";
import {
  buildRenderSegments,
  getTurnMessages,
  type MessagePart,
  type RenderSegment,
  parseSpecialMarkers,
  splitReasoningAndResponse,
} from "./helpers";
import { AssistantMessageActions } from "./components/AssistantMessageActions";
import { CopyButton } from "./components/CopyButton";
import { CollapsedToolGroup } from "./components/CollapsedToolGroup";
import { MessageAttachments } from "./components/MessageAttachments";
import { MessagePartRenderer } from "./components/MessagePartRenderer";
import { ReasoningCollapse } from "./components/ReasoningCollapse";
import { ThinkingIndicator } from "./components/ThinkingIndicator";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  sessionID?: string | null;
  hasMoreMessages?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
  onRetry?: () => void;
  historicalDurations?: Map<string, number>;
  /** Pending queued messages waiting to be injected, shown at the end of chat. */
  queuedMessages?: string[];
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

/**
 * Triggers `onLoadMore` when scrolled near the top, and preserves the
 * user's scroll position after older messages are prepended to the DOM.
 *
 * Scroll preservation works by:
 * 1. Capturing `scrollHeight` / `scrollTop` in the observer callback
 *    (synchronous, before React re-renders).
 * 2. Restoring `scrollTop` in a `useLayoutEffect` keyed on
 *    `messageCount` so it only fires when messages actually change
 *    (not on intermediate renders like the loading-spinner toggle).
 */
function LoadMoreSentinel({
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
  const { scrollRef } = useStickToBottomContext();

  // IntersectionObserver to trigger load when sentinel is near viewport.
  // Only fires when the container is actually scrollable to prevent
  // exhausting all pages when content fits without scrolling.
  useEffect(() => {
    if (!sentinelRef.current || !hasMore || isLoading) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        const scrollParent =
          sentinelRef.current?.closest('[role="log"]') ??
          sentinelRef.current?.parentElement;
        if (
          scrollParent &&
          scrollParent.scrollHeight <= scrollParent.clientHeight
        )
          return;
        // Capture scroll metrics *before* the state update
        const el = scrollRef.current;
        if (el) {
          scrollSnapshotRef.current = {
            scrollHeight: el.scrollHeight,
            scrollTop: el.scrollTop,
          };
        }
        onLoadMoreRef.current();
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
    scrollSnapshotRef.current = { scrollHeight: 0, scrollTop: 0 };
  }, [messageCount, scrollRef]);

  return (
    <div ref={sentinelRef} className="flex justify-center py-1">
      {isLoading && <LoadingSpinner className="h-5 w-5 text-neutral-400" />}
    </div>
  );
}

export function ChatMessagesContainer({
  messages,
  status,
  error,
  isLoading,
  sessionID,
  hasMoreMessages,
  isLoadingMore,
  onLoadMore,
  onRetry,
  historicalDurations,
  queuedMessages,
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

  const hasInflight = (() => {
    if (lastMessage?.role !== "assistant") return false;
    const parts = lastMessage.parts;
    if (parts.length === 0) return false;

    const lastPart = parts[parts.length - 1];

    if (lastPart.type === "text" && lastPart.text.trim().length > 0)
      return true;

    if (
      lastPart.type.startsWith(TOOL_PART_PREFIX) &&
      "state" in lastPart &&
      (lastPart.state === "input-streaming" ||
        lastPart.state === "input-available")
    )
      return true;

    return false;
  })();

  const showThinking =
    status === "submitted" || (status === "streaming" && !hasInflight);

  const isActivelyStreaming = status === "streaming" || status === "submitted";
  const { elapsedSeconds } = useElapsedTimer(isActivelyStreaming);

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
      <ConversationContent className="flex min-h-full flex-1 flex-col gap-6 px-3 py-6">
        {hasMoreMessages && onLoadMore && (
          <LoadMoreSentinel
            hasMore={hasMoreMessages}
            isLoading={!!isLoadingMore}
            messageCount={messages.length}
            onLoadMore={onLoadMore}
          />
        )}
        {isLoading && messages.length === 0 && (
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
          const textParts = message.parts.filter(
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

          const fileParts = message.parts.filter(
            (p): p is FileUIPart => p.type === "file",
          );

          // For finalized assistant messages, split into reasoning + response.
          // During streaming, show everything normally with tool collapsing.
          const isFinalized =
            message.role === "assistant" && !isCurrentlyStreaming;
          const { reasoning, response } = isFinalized
            ? splitReasoningAndResponse(message.parts)
            : { reasoning: [] as MessagePart[], response: message.parts };
          const hasReasoning = reasoning.length > 0;

          // Note: when interactive tools are pinned from reasoning into response,
          // this index approximates their position (used only for React keys).
          const responseStartIndex = message.parts.length - response.length;
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
                  <ReasoningCollapse>
                    {renderSegments(reasoningSegments, message.id)}
                  </ReasoningCollapse>
                )}
                {responseSegments
                  ? renderSegments(
                      responseSegments,
                      message.id,
                      isLastAssistant ? onRetry : undefined,
                    )
                  : message.parts.map((part, i) => (
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
                    durationMs={historicalDurations?.get(message.id)}
                  />
                )}
                {isLastAssistant && showThinking && (
                  <ThinkingIndicator
                    active={showThinking}
                    elapsedSeconds={elapsedSeconds}
                  />
                )}
              </MessageContent>
              {message.role === "user" && textParts.length > 0 && (
                <MessageActions className="mt-1 justify-end opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
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
        {showThinking && lastMessage?.role !== "assistant" && (
          <Message from="assistant">
            <MessageContent className="text-[1rem] leading-relaxed">
              <ThinkingIndicator
                active={showThinking}
                elapsedSeconds={elapsedSeconds}
              />
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
        {error && (
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
