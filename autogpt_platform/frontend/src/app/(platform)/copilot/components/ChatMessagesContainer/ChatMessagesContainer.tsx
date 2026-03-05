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
import { parseSpecialMarkers } from "./helpers";
import { AssistantMessageActions } from "./components/AssistantMessageActions";
import { MessageAttachments } from "./components/MessageAttachments";
import { MessagePartRenderer } from "./components/MessagePartRenderer";
import { ThinkingIndicator } from "./components/ThinkingIndicator";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  headerSlot?: React.ReactNode;
  sessionID?: string | null;
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

          return (
            <Message from={message.role} key={message.id}>
              <MessageContent
                className={
                  "text-[1rem] leading-relaxed " +
                  "group-[.is-user]:rounded-xl group-[.is-user]:bg-purple-100 group-[.is-user]:px-3 group-[.is-user]:py-2.5 group-[.is-user]:text-slate-900 group-[.is-user]:[border-bottom-right-radius:0] " +
                  "group-[.is-assistant]:bg-transparent group-[.is-assistant]:text-slate-900"
                }
              >
                {message.parts.map((part, i) => (
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
