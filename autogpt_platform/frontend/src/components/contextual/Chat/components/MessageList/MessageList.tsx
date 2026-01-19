"use client";

import { cn } from "@/lib/utils";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";
import { ChatMessage } from "../ChatMessage/ChatMessage";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
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
        "flex-1 overflow-y-auto overflow-x-hidden",
        "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300",
        className,
      )}
    >
      <div className="mx-auto flex flex-col py-4 min-w-0 break-words hyphens-auto">
        {/* Render all persisted messages */}
        {(() => {
          let lastAssistantMessageIndex = -1;
          for (let i = messages.length - 1; i >= 0; i--) {
            const msg = messages[i];
            if (msg.type === "message" && msg.role === "assistant") {
              lastAssistantMessageIndex = i;
              break;
            }
          }

          let lastToolResponseIndex = -1;
          for (let i = messages.length - 1; i >= 0; i--) {
            const msg = messages[i];
            if (msg.type === "tool_response") {
              lastToolResponseIndex = i;
              break;
            }
          }

          return messages
            .map((message, index) => {
          // Log message for debugging
          if (message.type === "message" && message.role === "assistant") {
            const prevMessage = messages[index - 1];
            const prevMessageToolName = prevMessage?.type === "tool_call" ? prevMessage.toolName : undefined;
            console.log("[MessageList] Assistant message:", {
              index,
              content: message.content.substring(0, 200),
              fullContent: message.content,
              prevMessageType: prevMessage?.type,
              prevMessageToolName,
            });
          }

          // Check if current message is an agent_output tool_response
          // and if previous message is an assistant message
          let agentOutput: ChatMessageData | undefined;
          let messageToRender: ChatMessageData = message;

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

          // Check if assistant message follows a tool_call and looks like a tool output
          if (message.type === "message" && message.role === "assistant") {
            const prevMessage = messages[index - 1];
            
            // Check if next message is an agent_output tool_response to include in current assistant message
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

            // Only convert to tool_response if it follows a tool_call AND looks like a tool output
            if (prevMessage && prevMessage.type === "tool_call") {
              const content = message.content.toLowerCase().trim();
              // Patterns that indicate this is a tool output result, not an agent response
              const isToolOutputPattern = 
                content.startsWith("no agents found") ||
                content.startsWith("no results found") ||
                content.includes("no agents found matching") ||
                content.match(/^no \w+ found/i) ||
                (content.length < 150 && content.includes("try different")) ||
                (content.length < 200 && !content.includes("i'll") && !content.includes("let me") && !content.includes("i can") && !content.includes("i will"));

              console.log("[MessageList] Checking if assistant message is tool output:", {
                content: message.content.substring(0, 100),
                isToolOutputPattern,
                prevToolName: prevMessage.toolName,
              });

              if (isToolOutputPattern) {
                // Convert this message to a tool_response format for rendering
                messageToRender = {
                  type: "tool_response",
                  toolId: prevMessage.toolId,
                  toolName: prevMessage.toolName,
                  result: message.content,
                  success: true,
                  timestamp: message.timestamp,
                } as ChatMessageData;
              }
            }
          }

              const isFinalMessage =
                messageToRender.type !== "message" ||
                messageToRender.role !== "assistant" ||
                index === lastAssistantMessageIndex;

              // Render last tool_response as AIChatBubble (but skip agent_output that's rendered inside assistant message)
              if (
                messageToRender.type === "tool_response" &&
                message.type === "tool_response" &&
                index === lastToolResponseIndex
              ) {
                  // Check if this is an agent_output that should be rendered inside assistant message
                  let parsedResult: Record<string, unknown> | null = null;
                  try {
                    parsedResult =
                      typeof messageToRender.result === "string"
                        ? JSON.parse(messageToRender.result)
                        : (messageToRender.result as Record<string, unknown>);
                  } catch {
                    parsedResult = null;
                  }

                  const isAgentOutput = parsedResult?.type === "agent_output";
                  const prevMessage = messages[index - 1];
                  const shouldSkip =
                    isAgentOutput &&
                    prevMessage &&
                    prevMessage.type === "message" &&
                    prevMessage.role === "assistant";

                  if (shouldSkip) return null;

                  const resultValue =
                    typeof messageToRender.result === "string"
                      ? messageToRender.result
                      : messageToRender.result
                        ? JSON.stringify(messageToRender.result, null, 2)
                        : "";

                  return (
                    <div key={index} className="px-4 py-2 min-w-0 overflow-x-hidden break-words hyphens-auto">
                      <AIChatBubble>
                        <MarkdownContent content={resultValue} />
                      </AIChatBubble>
                    </div>
                  );
              }

              return (
                <ChatMessage
                  key={index}
                  message={messageToRender}
                  onSendMessage={onSendMessage}
                  agentOutput={agentOutput}
                  isFinalMessage={isFinalMessage}
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
  );
}
