"use client";

import { cn } from "@/lib/utils";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { ThinkingAccordion } from "../ThinkingAccordion/ThinkingAccordion";
import { LastToolResponse } from "./components/LastToolResponse/LastToolResponse";
import { MessageItem } from "./components/MessageItem/MessageItem";
import {
  createToolResponseMap,
  filterMessagesForDisplay,
  findLastMessageIndex,
  shouldSkipAgentOutput,
} from "./helpers";
import { useMessageList } from "./useMessageList";

export interface MessageListProps {
  messages: ChatMessageData[];
  streamingChunks?: string[];
  isStreaming?: boolean;
  className?: string;
  onSendMessage?: (content: string) => void;
}

export function MessageList({
  messages,
  streamingChunks = [],
  isStreaming = false,
  className,
  onSendMessage,
}: MessageListProps) {
  const { messagesEndRef, messagesContainerRef } = useMessageList({
    messageCount: messages.length,
    isStreaming,
  });

  // ChatGPT-style UX for tool messages:
  // - tool_call: Visible during streaming (icon + small grey text), hidden after final answer
  //   - Clicking a tool_call opens a dialog showing its corresponding tool_response
  // - tool_response: Always hidden from main list
  // - operation_*: Visible during streaming, hidden after final answer
  // - After streaming without final answer (error/cancel): Keep everything visible for debugging
  const displayMessages = filterMessagesForDisplay(messages, isStreaming);

  // Create a map from toolId -> tool_response for linking tool calls to their responses
  const toolResponseMap = createToolResponseMap(messages);

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
          {/* Render filtered messages (tool chatter hidden after streaming completes) */}
          {(() => {
            const lastAssistantMessageIndex = findLastMessageIndex(
              displayMessages,
              (msg) => msg.type === "message" && msg.role === "assistant",
            );

            const lastToolResponseIndex = findLastMessageIndex(
              displayMessages,
              (msg) => msg.type === "tool_response",
            );

            return displayMessages.map((message, index) => {
              // Skip agent_output tool_responses that should be rendered inside assistant messages
              if (shouldSkipAgentOutput(message, displayMessages[index - 1])) {
                return null;
              }

              // Render last tool_response as AIChatBubble (only during streaming)
              if (
                message.type === "tool_response" &&
                index === lastToolResponseIndex
              ) {
                return (
                  <LastToolResponse
                    key={index}
                    message={message}
                    prevMessage={displayMessages[index - 1]}
                  />
                );
              }

              return (
                <MessageItem
                  key={index}
                  message={message}
                  messages={displayMessages}
                  index={index}
                  lastAssistantMessageIndex={lastAssistantMessageIndex}
                  isStreaming={isStreaming}
                  onSendMessage={onSendMessage}
                  toolResponseMap={toolResponseMap}
                />
              );
            });
          })()}

          {/*
           * ChatGPT-style "Thinking" UX:
           * - During streaming: Show thinking indicator with rotating status labels
           *   - Click to open dialog with reasoning chunks
           *   - Tool responses are shown via clickable tool_call messages instead
           * - After streaming: Indicator is removed, only final answer remains
           */}
          {isStreaming && <ThinkingAccordion chunks={streamingChunks} />}

          {/* Invisible div to scroll to */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Bottom fade shadow */}
      <div className="pointer-events-none absolute bottom-0 z-10 h-8 w-full bg-gradient-to-t from-[#f8f8f9] to-transparent" />
    </div>
  );
}
