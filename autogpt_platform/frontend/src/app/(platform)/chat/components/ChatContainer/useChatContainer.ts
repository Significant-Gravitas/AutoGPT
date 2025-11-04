import { useState, useCallback, useRef, useMemo } from "react";
import { toast } from "sonner";
import { useChatStream, type StreamChunk } from "@/hooks/useChatStream";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { ChatMessageData } from "@/components/molecules/ChatMessage/useChatMessage";
import {
  parseToolResponse,
  extractCredentialsNeeded,
  isValidMessage,
  isToolCallArray,
} from "./helpers";

interface UseChatContainerArgs {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  onRefreshSession: () => Promise<void>;
}

interface UseChatContainerResult {
  messages: ChatMessageData[];
  streamingChunks: string[];
  isStreaming: boolean;
  error: Error | null;
  sendMessage: (content: string) => Promise<void>;
}

export function useChatContainer({
  sessionId,
  initialMessages,
  onRefreshSession,
}: UseChatContainerArgs): UseChatContainerResult {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
  const [hasTextChunks, setHasTextChunks] = useState(false);

  // Track streaming chunks in a ref so we can access the latest value in callbacks
  const streamingChunksRef = useRef<string[]>([]);

  const { error, sendMessage: sendStreamMessage } = useChatStream();

  // Show streaming UI when we have text chunks, independent of connection state
  // This keeps the StreamingMessage visible during the transition to persisted message
  const isStreaming = hasTextChunks;

  /**
   * Convert initial messages to our format, filtering out empty messages.
   * Memoized to prevent expensive re-computation on every render.
   */
  const allMessages = useMemo((): ChatMessageData[] => {
    const processedInitialMessages = initialMessages
      .filter((msg: Record<string, unknown>) => {
        // Validate message structure first
        if (!isValidMessage(msg)) {
          console.warn("Invalid message structure from backend:", msg);
          return false;
        }

        // Include messages with content OR tool_calls (tool_call messages have empty content)
        const content = String(msg.content || "").trim();
        const toolCalls = msg.tool_calls;
        return content.length > 0 || (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0);
      })
      .map((msg: Record<string, unknown>): ChatMessageData | null => {
        const content = String(msg.content || "");
        const role = String(msg.role || "assistant").toLowerCase();

        // Check if this is a tool_call message (assistant message with tool_calls)
        const toolCalls = msg.tool_calls;

        // Validate tool_calls structure if present
        if (role === "assistant" && toolCalls && isToolCallArray(toolCalls) && toolCalls.length > 0) {
          // Skip tool_call messages from persisted history
          // We only show tool_calls during live streaming, not from history
          // The tool_response that follows it is what we want to display
          return null;
        }

        // Check if this is a tool response message (role="tool")
        if (role === "tool") {
          const timestamp = msg.timestamp
            ? new Date(msg.timestamp as string)
            : undefined;

          // Use helper function to parse tool response
          const toolResponse = parseToolResponse(
            content,
            (msg.tool_call_id as string) || "",
            "unknown",
            timestamp
          );

          // parseToolResponse returns null for setup_requirements
          // In that case, skip this message (it should be handled during streaming)
          if (!toolResponse) {
            return null;
          }

          return toolResponse;
        }

        // Return as regular message
        return {
          type: "message",
          role: role as "user" | "assistant" | "system",
          content,
          timestamp: msg.timestamp
            ? new Date(msg.timestamp as string)
            : undefined,
        };
      })
      .filter((msg): msg is ChatMessageData => msg !== null); // Remove null entries

    return [...processedInitialMessages, ...messages];
  }, [initialMessages, messages]);

  /**
   * Send a message and handle the streaming response.
   *
   * Message Flow:
   * 1. User message added immediately to local state
   * 2. text_chunk events accumulate in streaming box
   * 3. text_ended closes streaming box
   * 4. tool_call_start shows ToolCallMessage (spinning gear)
   * 5. tool_response replaces ToolCallMessage with ToolResponseMessage (result)
   * 6. stream_end finalizes, saves to backend, triggers refresh
   *
   * State Management:
   * - Local `messages` state tracks only new messages during streaming
   * - `streamingChunks` accumulates text as it arrives
   * - `streamingChunksRef` prevents stale closures in async handlers
   * - On stream_end, local messages cleared and replaced by refreshed initialMessages
   */
  const sendMessage = useCallback(
    async function sendMessage(content: string) {
      if (!sessionId) {
        console.error("Cannot send message: no session ID");
        return;
      }

      // Add user message immediately
      const userMessage: ChatMessageData = {
        type: "message",
        role: "user",
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Clear streaming chunks and reset text flag
      setStreamingChunks([]);
      streamingChunksRef.current = [];
      setHasTextChunks(false);

      try {
        // Stream the response
        await sendStreamMessage(
          sessionId,
          content,
          function handleChunk(chunk: StreamChunk) {
            if (chunk.type === "text_chunk" && chunk.content) {
              setHasTextChunks(true); // Mark that we have text chunks
              setStreamingChunks((prev) => {
                const updated = [...prev, chunk.content!];
                streamingChunksRef.current = updated;
                return updated;
              });
            } else if (chunk.type === "text_ended") {
              // Close the streaming text box
              console.log("[Text Ended] Closing streaming text box");
              setStreamingChunks([]);
              streamingChunksRef.current = [];
              setHasTextChunks(false);
            } else if (chunk.type === "tool_call_start") {
              // Show ToolCallMessage immediately when tool execution starts
              const toolCallMessage: ChatMessageData = {
                type: "tool_call",
                toolId: `tool-${Date.now()}-${chunk.idx || 0}`,
                toolName: "Executing...",
                arguments: {},
                timestamp: new Date(),
              };
              setMessages((prev) => [...prev, toolCallMessage]);

              console.log("[Tool Call Start]", {
                toolId: toolCallMessage.toolId,
                timestamp: new Date().toISOString(),
              });
            } else if (chunk.type === "tool_response") {
              console.log("[Tool Response] Received:", {
                toolId: chunk.tool_id,
                toolName: chunk.tool_name,
                timestamp: new Date().toISOString(),
              });

              // Use helper function to parse tool response
              const responseMessage = parseToolResponse(
                chunk.result!,
                chunk.tool_id!,
                chunk.tool_name!,
                new Date()
              );

              // If helper returns null (setup_requirements), handle credentials
              if (!responseMessage) {
                // Parse for credentials check
                let parsedResult: Record<string, unknown> | null = null;
                try {
                  parsedResult =
                    typeof chunk.result === "string"
                      ? JSON.parse(chunk.result)
                      : (chunk.result as Record<string, unknown>);
                } catch {
                  parsedResult = null;
                }

                // Check if this is get_required_setup_info with missing credentials
                if (
                  chunk.tool_name === "get_required_setup_info" &&
                  chunk.success &&
                  parsedResult
                ) {
                  const credentialsMessage = extractCredentialsNeeded(parsedResult);
                  if (credentialsMessage) {
                    setMessages((prev) => [...prev, credentialsMessage]);
                  }
                }

                // Don't add message if setup_requirements
                return;
              }

              // Replace the most recent tool_call message with the response
              setMessages((prev) => {
                const toolCallIndex = [...prev]
                  .reverse()
                  .findIndex((msg) => msg.type === "tool_call");

                if (toolCallIndex !== -1) {
                  const actualIndex = prev.length - 1 - toolCallIndex;
                  const newMessages = [...prev];
                  newMessages[actualIndex] = responseMessage;

                  console.log("[Tool Response] Replaced tool_call at index:", actualIndex);
                  return newMessages;
                }

                console.warn("[Tool Response] No tool_call found to replace, appending");
                return [...prev, responseMessage];
              });
            } else if (chunk.type === "login_needed") {
              // Add login needed message
              const loginNeededMessage: ChatMessageData = {
                type: "login_needed",
                message: chunk.message || "Authentication required to continue",
                sessionId: chunk.session_id || sessionId,
                timestamp: new Date(),
              };
              setMessages((prev) => [...prev, loginNeededMessage]);
            } else if (chunk.type === "stream_end") {
              console.log("[Stream End] Final state:", {
                localMessages: messages.map((m) => ({
                  type: m.type,
                  ...(m.type === "message" && {
                    role: m.role,
                    contentLength: m.content.length,
                  }),
                  ...(m.type === "tool_call" && {
                    toolId: m.toolId,
                    toolName: m.toolName,
                  }),
                  ...(m.type === "tool_response" && {
                    toolId: m.toolId,
                    toolName: m.toolName,
                    success: m.success,
                  }),
                })),
                streamingChunks: streamingChunksRef.current,
                timestamp: new Date().toISOString(),
              });

              // Convert streaming chunks into a completed assistant message
              // This provides seamless transition without flash or resize
              // Use ref to get the latest chunks value (not stale closure value)
              const completedContent = streamingChunksRef.current.join("");

              if (completedContent) {
                const assistantMessage: ChatMessageData = {
                  type: "message",
                  role: "assistant",
                  content: completedContent,
                  timestamp: new Date(),
                };

                // Add the completed assistant message to local state
                // It will be visible immediately while backend updates
                setMessages((prev) => [...prev, assistantMessage]);
              }

              // Clear streaming state immediately now that we have the message
              setStreamingChunks([]);
              streamingChunksRef.current = [];
              setHasTextChunks(false);

              // Messages are now in local state and will be displayed
              // No need to refresh from backend - local state is the source of truth
              console.log("[Stream End] Stream complete, messages in local state");
            } else if (chunk.type === "error") {
              const errorMessage =
                chunk.message || chunk.content || "An error occurred";
              console.error("Stream error:", errorMessage);
              toast.error("Chat Error", {
                description: errorMessage,
              });
            }
            // TODO: Handle usage for display
          },
        );
      } catch (err) {
        console.error("Failed to send message:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to send message";
        toast.error("Failed to send message", {
          description: errorMessage,
        });
      }
    },
    [sessionId, sendStreamMessage, onRefreshSession],
  );

  return {
    messages: allMessages,
    streamingChunks,
    isStreaming,
    error,
    sendMessage,
  };
}
