import { useState, useCallback, useRef, useMemo } from "react";
import { toast } from "sonner";
import { useChatStream } from "@/hooks/useChatStream";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { ChatMessageData } from "@/components/molecules/ChatMessage/useChatMessage";
import {
  parseToolResponse,
  isValidMessage,
  isToolCallArray,
  createUserMessage,
  filterAuthMessages,
} from "./helpers";
import { createStreamEventDispatcher } from "./createStreamEventDispatcher";

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
  sendMessage: (content: string, isUserMessage?: boolean) => Promise<void>;
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
        return (
          content.length > 0 ||
          (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0)
        );
      })
      .map((msg: Record<string, unknown>): ChatMessageData | null => {
        const content = String(msg.content || "");
        const role = String(msg.role || "assistant").toLowerCase();

        // Check if this is a tool_call message (assistant message with tool_calls)
        const toolCalls = msg.tool_calls;

        // Validate tool_calls structure if present
        if (
          role === "assistant" &&
          toolCalls &&
          isToolCallArray(toolCalls) &&
          toolCalls.length > 0
        ) {
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
            timestamp,
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
    async function sendMessage(content: string, isUserMessage: boolean = true) {
      if (!sessionId) {
        console.error("Cannot send message: no session ID");
        return;
      }

      // Update message state: add user message and remove stale auth prompts
      if (isUserMessage) {
        const userMessage = createUserMessage(content);
        setMessages((prev) => [...filterAuthMessages(prev), userMessage]);
      } else {
        // For system messages, just remove the login/credentials prompts
        setMessages((prev) => filterAuthMessages(prev));
      }

      // Clear streaming state
      setStreamingChunks([]);
      streamingChunksRef.current = [];
      setHasTextChunks(false);

      // Create event dispatcher with all handler dependencies
      const dispatcher = createStreamEventDispatcher({
        setHasTextChunks,
        setStreamingChunks,
        streamingChunksRef,
        setMessages,
        sessionId,
      });

      try {
        // Stream the response using the event dispatcher
        await sendStreamMessage(sessionId, content, dispatcher, isUserMessage);
      } catch (err) {
        console.error("Failed to send message:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to send message";
        toast.error("Failed to send message", {
          description: errorMessage,
        });
      }
    },
    [sessionId, sendStreamMessage],
  );

  return {
    messages: allMessages,
    streamingChunks,
    isStreaming,
    error,
    sendMessage,
  };
}
