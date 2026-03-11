import { useMemo } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { TurnStatsBar } from "../JobStatsBar/TurnStatsBar";
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
  headerSlot?: React.ReactNode;
  sessionID?: string | null;
}

function renderSegments(
  segments: RenderSegment[],
  messageID: string,
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

export function ChatMessagesContainer({
  messages,
  status,
  error,
  isLoading,
  headerSlot,
  sessionID,
}: Props) {
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

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="flex flex-1 flex-col gap-6 px-3 py-6">
        {headerSlot}
        {isLoading && messages.length === 0 && (
          <div
            className="flex flex-1 items-center justify-center"
            style={{ minHeight: "calc(100vh - 12rem)" }}
          >
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
          const hasErrorMarker =
            lastTextPart !== undefined &&
            parseSpecialMarkers(lastTextPart.text).markerType === "error";
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
                  "group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900"
                }
              >
                {hasReasoning && reasoningSegments && (
                  <ReasoningCollapse>
                    {renderSegments(reasoningSegments, message.id)}
                  </ReasoningCollapse>
                )}
                {responseSegments
                  ? renderSegments(responseSegments, message.id)
                  : message.parts.map((part, i) => (
                      <MessagePartRenderer
                        key={`${message.id}-${i}`}
                        part={part}
                        messageID={message.id}
                        partIndex={i}
                      />
                    ))}
                {isLastInTurn && !isCurrentlyStreaming && (
                  <TurnStatsBar
                    turnMessages={getTurnMessages(messages, messageIndex)}
                  />
                )}
                {isLastAssistant && showThinking && (
                  <ThinkingIndicator active={showThinking} />
                )}
              </MessageContent>
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
              <ThinkingIndicator active={showThinking} />
            </MessageContent>
          </Message>
        )}
        {graphExecId && <CopilotPendingReviews graphExecId={graphExecId} />}
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
