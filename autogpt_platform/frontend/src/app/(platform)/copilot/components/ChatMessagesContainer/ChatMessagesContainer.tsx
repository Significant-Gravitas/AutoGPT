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
import { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { MessageAttachments } from "./components/MessageAttachments";
import { MessagePartRenderer } from "./components/MessagePartRenderer";
import { ThinkingIndicator } from "./components/ThinkingIndicator";
import { CopyButton } from "./components/CopyButton";
import { TTSButton } from "./components/TTSButton";
import { parseSpecialMarkers } from "./helpers";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  isLoading: boolean;
  headerSlot?: React.ReactNode;
}

export function ChatMessagesContainer({
  messages,
  status,
  error,
  isLoading,
  headerSlot,
}: Props) {
  const lastMessage = messages[messages.length - 1];

  // Determine if something is visibly "in-flight" in the last assistant message:
  // - Text is actively streaming (last part is non-empty text)
  // - A tool call is pending (state is input-streaming or input-available)
  const hasInflight = (() => {
    if (lastMessage?.role !== "assistant") return false;
    const parts = lastMessage.parts;
    if (parts.length === 0) return false;

    const lastPart = parts[parts.length - 1];

    // Text is actively being written
    if (lastPart.type === "text" && lastPart.text.trim().length > 0)
      return true;

    // A tool call is still pending (no output yet)
    if (
      lastPart.type.startsWith("tool-") &&
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

          const isAssistant = message.role === "assistant";

          // Past assistant messages are always done; the last one
          // is done only when the stream has finished.
          const isAssistantDone =
            isAssistant &&
            (!isLastAssistant ||
              (status !== "streaming" && status !== "submitted"));

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
              {isAssistantDone &&
                (() => {
                  const textParts = message.parts.filter(
                    (p): p is Extract<typeof p, { type: "text" }> =>
                      p.type === "text",
                  );

                  // Hide actions when the message ended with an error or cancellation
                  const lastTextPart = textParts[textParts.length - 1];
                  if (lastTextPart) {
                    const { markerType } = parseSpecialMarkers(
                      lastTextPart.text,
                    );
                    if (markerType === "error") return null;
                  }

                  const textContent = textParts.map((p) => p.text).join("\n");
                  return (
                    <MessageActions>
                      <CopyButton text={textContent} />
                      <TTSButton text={textContent} />
                    </MessageActions>
                  );
                })()}
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
