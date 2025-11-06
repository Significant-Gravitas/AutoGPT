import { useState, useCallback, useRef, useMemo } from "react";
import { toast } from "sonner";
import { useChatStream } from "@/app/(platform)/chat/useChatStream";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { ChatMessageData } from "@/app/(platform)/chat/components/ChatMessage/useChatMessage";
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

export function useChatContainer({
  sessionId,
  initialMessages,
}: UseChatContainerArgs) {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
  const [hasTextChunks, setHasTextChunks] = useState(false);
  const streamingChunksRef = useRef<string[]>([]);
  const { error, sendMessage: sendStreamMessage } = useChatStream();
  const isStreaming = hasTextChunks;

  const allMessages = useMemo(() => {
    const processedInitialMessages = initialMessages
      .filter((msg: Record<string, unknown>) => {
        if (!isValidMessage(msg)) {
          console.warn("Invalid message structure from backend:", msg);
          return false;
        }
        const content = String(msg.content || "").trim();
        const toolCalls = msg.tool_calls;
        return (
          content.length > 0 ||
          (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0)
        );
      })
      .map((msg: Record<string, unknown>) => {
        const content = String(msg.content || "");
        const role = String(msg.role || "assistant").toLowerCase();
        const toolCalls = msg.tool_calls;
        if (
          role === "assistant" &&
          toolCalls &&
          isToolCallArray(toolCalls) &&
          toolCalls.length > 0
        ) {
          return null;
        }
        if (role === "tool") {
          const timestamp = msg.timestamp
            ? new Date(msg.timestamp as string)
            : undefined;
          const toolResponse = parseToolResponse(
            content,
            (msg.tool_call_id as string) || "",
            "unknown",
            timestamp,
          );
          if (!toolResponse) {
            return null;
          }
          return toolResponse;
        }
        return {
          type: "message",
          role: role as "user" | "assistant" | "system",
          content,
          timestamp: msg.timestamp
            ? new Date(msg.timestamp as string)
            : undefined,
        };
      })
      .filter((msg): msg is ChatMessageData => msg !== null);

    return [...processedInitialMessages, ...messages];
  }, [initialMessages, messages]);

  const sendMessage = useCallback(
    async function sendMessage(content: string, isUserMessage: boolean = true) {
      if (!sessionId) {
        console.error("Cannot send message: no session ID");
        return;
      }
      if (isUserMessage) {
        const userMessage = createUserMessage(content);
        setMessages((prev) => [...filterAuthMessages(prev), userMessage]);
      } else {
        setMessages((prev) => filterAuthMessages(prev));
      }
      setStreamingChunks([]);
      streamingChunksRef.current = [];
      setHasTextChunks(false);
      const dispatcher = createStreamEventDispatcher({
        setHasTextChunks,
        setStreamingChunks,
        streamingChunksRef,
        setMessages,
        sessionId,
      });
      try {
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
