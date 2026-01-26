import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useChatStreamStore } from "@/providers/chat-stream/chat-stream-store";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatStream } from "../../useChatStream";
import { usePageContext } from "../../usePageContext";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { createStreamEventDispatcher } from "./createStreamEventDispatcher";
import {
  createUserMessage,
  filterAuthMessages,
  hasSentInitialPrompt,
  isToolCallArray,
  isValidMessage,
  markInitialPromptSent,
  parseToolResponse,
  removePageContext,
} from "./helpers";

interface Args {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  initialPrompt?: string;
}

export function useChatContainer({
  sessionId,
  initialMessages,
  initialPrompt,
}: Args) {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
  const [hasTextChunks, setHasTextChunks] = useState(false);
  const [isStreamingInitiated, setIsStreamingInitiated] = useState(false);
  const [isRegionBlockedModalOpen, setIsRegionBlockedModalOpen] =
    useState(false);
  const hasResponseRef = useRef(false);
  const streamingChunksRef = useRef<string[]>([]);
  const previousSessionIdRef = useRef<string | null>(null);
  const {
    error,
    sendMessage: sendStreamMessage,
    stopStreaming,
  } = useChatStream();
  const streamStore = useChatStreamStore();
  const isStreaming = isStreamingInitiated || hasTextChunks;

  useEffect(
    function handleSessionChange() {
      if (sessionId === previousSessionIdRef.current) return;

      const prevSession = previousSessionIdRef.current;
      if (prevSession) {
        stopStreaming(prevSession);
      }
      previousSessionIdRef.current = sessionId;
      setMessages([]);
      setStreamingChunks([]);
      streamingChunksRef.current = [];
      setHasTextChunks(false);
      setIsStreamingInitiated(false);
      hasResponseRef.current = false;

      if (!sessionId) return;

      const completedStream = streamStore.getCompletedStream(sessionId);
      const activeStream = streamStore.activeStreams.get(sessionId);
      const chunksToReplay = completedStream?.chunks || activeStream?.chunks;

      if (chunksToReplay && chunksToReplay.length > 0) {
        const dispatcher = createStreamEventDispatcher({
          setHasTextChunks,
          setStreamingChunks,
          streamingChunksRef,
          hasResponseRef,
          setMessages,
          setIsRegionBlockedModalOpen,
          sessionId,
          setIsStreamingInitiated,
        });

        for (const chunk of chunksToReplay) {
          dispatcher(chunk);
        }

        if (activeStream && activeStream.status === "streaming") {
          setIsStreamingInitiated(true);
          const unsubscribe = streamStore.subscribeToStream(
            sessionId,
            dispatcher,
          );
          return unsubscribe;
        }
      }
    },
    [sessionId, stopStreaming, streamStore],
  );

  const allMessages = useMemo(() => {
    const processedInitialMessages: ChatMessageData[] = [];
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

      if (role === "user") {
        content = removePageContext(content);
        if (!content.trim()) continue;
        processedInitialMessages.push({
          type: "message",
          role: "user",
          content,
          timestamp,
        });
        continue;
      }

      if (role === "assistant") {
        content = content
          .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
          .trim();

        if (toolCalls && isToolCallArray(toolCalls) && toolCalls.length > 0) {
          for (const toolCall of toolCalls) {
            const toolName = toolCall.function.name;
            const toolId = toolCall.id;
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
          if (content.trim()) {
            processedInitialMessages.push({
              type: "message",
              role: "assistant",
              content,
              timestamp,
            });
          }
        } else if (content.trim()) {
          processedInitialMessages.push({
            type: "message",
            role: "assistant",
            content,
            timestamp,
          });
        }
        continue;
      }

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
        console.error("[useChatContainer] Cannot send message: no session ID");
        return;
      }
      setIsRegionBlockedModalOpen(false);
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
      hasResponseRef.current = false;

      const dispatcher = createStreamEventDispatcher({
        setHasTextChunks,
        setStreamingChunks,
        streamingChunksRef,
        hasResponseRef,
        setMessages,
        setIsRegionBlockedModalOpen,
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
        console.error("[useChatContainer] Failed to send message:", err);
        setIsStreamingInitiated(false);

        // Don't show error toast for AbortError (expected during cleanup)
        if (err instanceof Error && err.name === "AbortError") return;

        const errorMessage =
          err instanceof Error ? err.message : "Failed to send message";
        toast.error("Failed to send message", {
          description: errorMessage,
        });
      }
    },
    [sessionId, sendStreamMessage],
  );

  const handleStopStreaming = useCallback(() => {
    stopStreaming();
    setStreamingChunks([]);
    streamingChunksRef.current = [];
    setHasTextChunks(false);
    setIsStreamingInitiated(false);
  }, [stopStreaming]);

  const { capturePageContext } = usePageContext();

  // Send initial prompt if provided (for new sessions from homepage)
  useEffect(
    function handleInitialPrompt() {
      if (!initialPrompt || !sessionId) return;
      if (initialMessages.length > 0) return;
      if (hasSentInitialPrompt(sessionId)) return;

      markInitialPromptSent(sessionId);
      const context = capturePageContext();
      sendMessage(initialPrompt, true, context);
    },
    [
      initialPrompt,
      sessionId,
      initialMessages.length,
      sendMessage,
      capturePageContext,
    ],
  );

  async function sendMessageWithContext(
    content: string,
    isUserMessage: boolean = true,
  ) {
    const context = capturePageContext();
    await sendMessage(content, isUserMessage, context);
  }

  function handleRegionModalOpenChange(open: boolean) {
    setIsRegionBlockedModalOpen(open);
  }

  function handleRegionModalClose() {
    setIsRegionBlockedModalOpen(false);
  }

  return {
    messages: allMessages,
    streamingChunks,
    isStreaming,
    error,
    isRegionBlockedModalOpen,
    setIsRegionBlockedModalOpen,
    sendMessageWithContext,
    handleRegionModalOpenChange,
    handleRegionModalClose,
    sendMessage,
    stopStreaming: handleStopStreaming,
  };
}
