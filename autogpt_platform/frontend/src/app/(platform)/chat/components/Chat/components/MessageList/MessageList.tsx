"use client";

import { cn } from "@/lib/utils";
import { ChatMessage } from "../ChatMessage/ChatMessage";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { StreamingMessage } from "../StreamingMessage/StreamingMessage";
import { ThinkingMessage } from "../ThinkingMessage/ThinkingMessage";
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

  return (
    <div
      ref={messagesContainerRef}
      className={cn(
        "flex-1 overflow-y-auto",
        "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300",
        className,
      )}
    >
      <div className="mx-auto flex max-w-3xl flex-col py-4">
        {/* Render all persisted messages */}
        {messages.map((message, index) => {
          // Check if current message is an agent_output tool_response
          // and if previous message is an assistant message
          let agentOutput: ChatMessageData | undefined;

          if (message.type === "tool_response" && message.result) {
            let parsedResult: Record<string, unknown> | null = null;
            try {
              parsedResult =
                typeof message.result === "string"
                  ? JSON.parse(message.result)
                  : (message.result as Record<string, unknown>);
            } catch {
              parsedResult = null;
            }
            if (parsedResult?.type === "agent_output") {
              const prevMessage = messages[index - 1];
              if (
                prevMessage &&
                prevMessage.type === "message" &&
                prevMessage.role === "assistant"
              ) {
                // This agent output will be rendered inside the previous assistant message
                // Skip rendering this message separately
                return null;
              }
            }
          }

          // Check if next message is an agent_output tool_response to include in current assistant message
          if (message.type === "message" && message.role === "assistant") {
            const nextMessage = messages[index + 1];
            if (
              nextMessage &&
              nextMessage.type === "tool_response" &&
              nextMessage.result
            ) {
              let parsedResult: Record<string, unknown> | null = null;
              try {
                parsedResult =
                  typeof nextMessage.result === "string"
                    ? JSON.parse(nextMessage.result)
                    : (nextMessage.result as Record<string, unknown>);
              } catch {
                parsedResult = null;
              }
              if (parsedResult?.type === "agent_output") {
                agentOutput = nextMessage;
              }
            }
          }

          return (
            <ChatMessage
              key={index}
              message={message}
              onSendMessage={onSendMessage}
              agentOutput={agentOutput}
            />
          );
        })}

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
  );
}
