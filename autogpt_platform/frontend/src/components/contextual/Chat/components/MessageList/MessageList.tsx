"use client";

import { cn } from "@/lib/utils";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { StreamingMessage } from "../StreamingMessage/StreamingMessage";
import { ThinkingMessage } from "../ThinkingMessage/ThinkingMessage";
import { LastToolResponse } from "./components/LastToolResponse/LastToolResponse";
import { MessageItem } from "./components/MessageItem/MessageItem";
import { findLastMessageIndex, shouldSkipAgentOutput } from "./helpers";
import { useMessageList } from "./useMessageList";

export interface MessageListProps {
  messages: ChatMessageData[];
  streamingChunks?: string[];
  isStreaming?: boolean;
  className?: string;
  onStreamComplete?: () => void;
  onSendMessage?: (content: string) => void;
}

export function MessageList({
  messages,
  streamingChunks = [],
  isStreaming = false,
  className,
  onStreamComplete,
  onSendMessage,
}: MessageListProps) {
  const { messagesEndRef, messagesContainerRef } = useMessageList({
    messageCount: messages.length,
    isStreaming,
  });

  /**
   * Keeps this for debugging purposes 💆🏽
   */
  console.log(messages);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {/* Top fade shadow */}
      <div className="pointer-events-none absolute top-0 z-10 h-8 w-full bg-gradient-to-b from-[#f8f8f9] to-transparent" />

      <div
        ref={messagesContainerRef}
        className={cn(
          "flex-1 overflow-y-auto overflow-x-hidden",
          "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300",
          className,
        )}
      >
        <div className="mx-auto flex min-w-0 flex-col hyphens-auto break-words py-4">
          {/* Render all persisted messages */}
          {(() => {
            const lastAssistantMessageIndex = findLastMessageIndex(
              messages,
              (msg) => msg.type === "message" && msg.role === "assistant",
            );

            const lastToolResponseIndex = findLastMessageIndex(
              messages,
              (msg) => msg.type === "tool_response",
            );

            return messages.map((message, index) => {
              // Skip agent_output tool_responses that should be rendered inside assistant messages
              if (shouldSkipAgentOutput(message, messages[index - 1])) {
                return null;
              }

              // Render last tool_response as AIChatBubble
              if (
                message.type === "tool_response" &&
                index === lastToolResponseIndex
              ) {
                return (
                  <LastToolResponse
                    key={index}
                    message={message}
                    prevMessage={messages[index - 1]}
                  />
                );
              }

              return (
                <MessageItem
                  key={index}
                  message={message}
                  messages={messages}
                  index={index}
                  lastAssistantMessageIndex={lastAssistantMessageIndex}
                  isStreaming={isStreaming}
                  onSendMessage={onSendMessage}
                />
              );
            });
          })()}

          {/* Render thinking message when streaming but no chunks yet */}
          {isStreaming && streamingChunks.length === 0 && <ThinkingMessage />}

          {/* Render streaming message if active */}
          {isStreaming && streamingChunks.length > 0 && (
            <StreamingMessage
              chunks={streamingChunks}
              onComplete={onStreamComplete}
            />
          )}

          {/* Invisible div to scroll to */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Bottom fade shadow */}
      <div className="pointer-events-none absolute bottom-0 z-10 h-8 w-full bg-gradient-to-t from-[#f8f8f9] to-transparent" />
    </div>
  );
}
