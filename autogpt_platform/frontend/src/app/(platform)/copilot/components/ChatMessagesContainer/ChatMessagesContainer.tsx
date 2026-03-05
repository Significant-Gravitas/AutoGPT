import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { FileUIPart, ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { TurnStatsBar } from "../JobStatsBar/TurnStatsBar";
import { parseSpecialMarkers } from "./helpers";
import { AssistantMessageActions } from "./components/AssistantMessageActions";
import { CollapsedToolGroup } from "./components/CollapsedToolGroup";
import { MessageAttachments } from "./components/MessageAttachments";
import { MessagePartRenderer } from "./components/MessagePartRenderer";
import { ReasoningCollapse } from "./components/ReasoningCollapse";
import { ThinkingIndicator } from "./components/ThinkingIndicator";

type MessagePart = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  headerSlot?: React.ReactNode;
  sessionID?: string | null;
}

function isCompletedToolPart(part: MessagePart): part is ToolUIPart {
  return (
    part.type.startsWith("tool-") &&
    "state" in part &&
    (part.state === "output-available" || part.state === "output-error")
  );
}

type RenderSegment =
  | { kind: "part"; part: MessagePart; index: number }
  | { kind: "collapsed-group"; parts: ToolUIPart[] };

// Tool types that have custom renderers and should NOT be collapsed
const CUSTOM_TOOL_TYPES = new Set([
  "tool-find_block",
  "tool-find_agent",
  "tool-find_library_agent",
  "tool-search_docs",
  "tool-get_doc_page",
  "tool-run_block",
  "tool-run_mcp_tool",
  "tool-run_agent",
  "tool-schedule_agent",
  "tool-create_agent",
  "tool-edit_agent",
  "tool-view_agent_output",
  "tool-search_feature_requests",
  "tool-create_feature_request",
]);

/**
 * Groups consecutive completed generic tool parts into collapsed segments.
 * Non-generic tools (those with custom renderers) and active/streaming tools
 * are left as individual parts.
 */
function buildRenderSegments(
  parts: MessagePart[],
  baseIndex = 0,
): RenderSegment[] {
  const segments: RenderSegment[] = [];
  let pendingGroup: Array<{ part: ToolUIPart; index: number }> | null = null;

  function flushGroup() {
    if (!pendingGroup) return;
    if (pendingGroup.length >= 2) {
      segments.push({
        kind: "collapsed-group",
        parts: pendingGroup.map((p) => p.part),
      });
    } else {
      for (const p of pendingGroup) {
        segments.push({ kind: "part", part: p.part, index: p.index });
      }
    }
    pendingGroup = null;
  }

  parts.forEach((part, i) => {
    const absoluteIndex = baseIndex + i;
    const isGenericCompletedTool =
      isCompletedToolPart(part) && !CUSTOM_TOOL_TYPES.has(part.type);

    if (isGenericCompletedTool) {
      if (!pendingGroup) pendingGroup = [];
      pendingGroup.push({ part: part as ToolUIPart, index: absoluteIndex });
    } else {
      flushGroup();
      segments.push({ kind: "part", part, index: absoluteIndex });
    }
  });

  flushGroup();
  return segments;
}

/**
 * For finalized assistant messages, split parts into "reasoning" (intermediate
 * text + tools before the final response) and "response" (final text after the
 * last tool). If there are no tools, everything is response.
 */
function splitReasoningAndResponse(parts: MessagePart[]): {
  reasoning: MessagePart[];
  response: MessagePart[];
} {
  // Find the index of the last tool part
  let lastToolIndex = -1;
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].type.startsWith("tool-")) {
      lastToolIndex = i;
      break;
    }
  }

  // No tools → everything is response
  if (lastToolIndex === -1) {
    return { reasoning: [], response: parts };
  }

  // Check if there's any text after the last tool
  const hasResponseAfterTools = parts
    .slice(lastToolIndex + 1)
    .some((p) => p.type === "text");

  if (!hasResponseAfterTools) {
    // No final text response → don't collapse anything
    return { reasoning: [], response: parts };
  }

  return {
    reasoning: parts.slice(0, lastToolIndex + 1),
    response: parts.slice(lastToolIndex + 1),
  };
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

/** Collect all messages belonging to a turn: the user message + every
 *  assistant message up to (but not including) the next user message. */
function getTurnMessages(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  lastAssistantIndex: number,
): UIMessage<unknown, UIDataTypes, UITools>[] {
  const userIndex = messages.findLastIndex(
    (m, i) => i < lastAssistantIndex && m.role === "user",
  );
  const nextUserIndex = messages.findIndex(
    (m, i) => i > lastAssistantIndex && m.role === "user",
  );
  const start = userIndex >= 0 ? userIndex : lastAssistantIndex;
  const end = nextUserIndex >= 0 ? nextUserIndex : messages.length;
  return messages.slice(start, end);
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
