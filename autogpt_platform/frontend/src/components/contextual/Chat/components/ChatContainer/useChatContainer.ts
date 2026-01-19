import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useCallback, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatStream } from "../../useChatStream";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { createStreamEventDispatcher } from "./createStreamEventDispatcher";
import {
  createUserMessage,
  filterAuthMessages,
  isToolCallArray,
  isValidMessage,
  parseToolResponse,
  removePageContext,
} from "./helpers";

interface Args {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
}

export function useChatContainer({ sessionId, initialMessages }: Args) {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
  const [hasTextChunks, setHasTextChunks] = useState(false);
  const [isStreamingInitiated, setIsStreamingInitiated] = useState(false);
  const streamingChunksRef = useRef<string[]>([]);
  const { error, sendMessage: sendStreamMessage } = useChatStream();
  const isStreaming = isStreamingInitiated || hasTextChunks;

  const allMessages = useMemo(() => {
    const processedInitialMessages: ChatMessageData[] = [];
    // Map to track tool calls by their ID so we can look up tool names for tool responses
    const toolCallMap = new Map<string, string>();

    for (const msg of initialMessages) {
      if (!isValidMessage(msg)) {
        console.warn("Invalid message structure from backend:", msg);
        continue;
      }

      let content = String(msg.content || "");
      const role = String(msg.role || "assistant").toLowerCase();
      const toolCalls = msg.tool_calls;
      const timestamp = msg.timestamp
        ? new Date(msg.timestamp as string)
        : undefined;

      // Remove page context from user messages when loading existing sessions
      if (role === "user") {
        content = removePageContext(content);
        // Skip user messages that become empty after removing page context
        if (!content.trim()) {
          continue;
        }
        processedInitialMessages.push({
          type: "message",
          role: "user",
          content,
          timestamp,
        });
        continue;
      }

      // Handle assistant messages first (before tool messages) to build tool call map
      if (role === "assistant") {
        // Strip <thinking> tags from content
        content = content
          .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
          .trim();

        // If assistant has tool calls, create tool_call messages for each
        if (toolCalls && isToolCallArray(toolCalls) && toolCalls.length > 0) {
          for (const toolCall of toolCalls) {
            const toolName = toolCall.function.name;
            const toolId = toolCall.id;
            // Store tool name for later lookup
            toolCallMap.set(toolId, toolName);

            try {
              const args = JSON.parse(toolCall.function.arguments || "{}");
              processedInitialMessages.push({
                type: "tool_call",
                toolId,
                toolName,
                arguments: args,
                timestamp,
              });
            } catch (err) {
              console.warn("Failed to parse tool call arguments:", err);
              processedInitialMessages.push({
                type: "tool_call",
                toolId,
                toolName,
                arguments: {},
                timestamp,
              });
            }
          }
          // Only add assistant message if there's content after stripping thinking tags
          if (content.trim()) {
            processedInitialMessages.push({
              type: "message",
              role: "assistant",
              content,
              timestamp,
            });
          }
        } else if (content.trim()) {
          // Assistant message without tool calls, but with content
          processedInitialMessages.push({
            type: "message",
            role: "assistant",
            content,
            timestamp,
          });
        }
        continue;
      }

      // Handle tool messages - look up tool name from tool call map
      if (role === "tool") {
        const toolCallId = (msg.tool_call_id as string) || "";
        const toolName = toolCallMap.get(toolCallId) || "unknown";
        const toolResponse = parseToolResponse(
          content,
          toolCallId,
          toolName,
          timestamp,
        );
        if (toolResponse) {
          processedInitialMessages.push(toolResponse);
        }
        continue;
      }

      // Handle other message types (system, etc.)
      if (content.trim()) {
        processedInitialMessages.push({
          type: "message",
          role: role as "user" | "assistant" | "system",
          content,
          timestamp,
        });
      }
    }

    return [...processedInitialMessages, ...messages];
  }, [initialMessages, messages]);

  const sendMessage = useCallback(
    async function sendMessage(
      content: string,
      isUserMessage: boolean = true,
      context?: { url: string; content: string },
    ) {
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
      setIsStreamingInitiated(true);
      const dispatcher = createStreamEventDispatcher({
        setHasTextChunks,
        setStreamingChunks,
        streamingChunksRef,
        setMessages,
        sessionId,
        setIsStreamingInitiated,
      });
      try {
        await sendStreamMessage(
          sessionId,
          content,
          dispatcher,
          isUserMessage,
          context,
        );
      } catch (err) {
        console.error("Failed to send message:", err);
        setIsStreamingInitiated(false);
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
