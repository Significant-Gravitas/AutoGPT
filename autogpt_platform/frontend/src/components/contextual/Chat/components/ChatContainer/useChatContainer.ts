import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatStore } from "../../chat-store";
import { toast } from "sonner";
import { useChatStream } from "../../useChatStream";
import { usePageContext } from "../../usePageContext";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import { createStreamEventDispatcher } from "./createStreamEventDispatcher";
import {
  createUserMessage,
  filterAuthMessages,
  hasSentInitialPrompt,
  markInitialPromptSent,
  processInitialMessages,
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
  const activeStreams = useChatStore((s) => s.activeStreams);
  const subscribeToStream = useChatStore((s) => s.subscribeToStream);
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

      const activeStream = activeStreams.get(sessionId);
      if (!activeStream || activeStream.status !== "streaming") return;

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

      setIsStreamingInitiated(true);
      const skipReplay = initialMessages.length > 0;
      return subscribeToStream(sessionId, dispatcher, skipReplay);
    },
    [sessionId, stopStreaming, activeStreams, subscribeToStream],
  );

  // Combine initial messages from backend with local streaming messages,
  // then deduplicate to prevent duplicates when polling refreshes initialMessages
  const allMessages = useMemo(() => {
    const processedInitial = processInitialMessages(initialMessages);

    // Collect toolIds that have completed (non-operation) results from DB
    // These indicate operations that finished - we should filter out their
    // corresponding operation_started/pending messages from local state
    const completedToolIds = new Set<string>();
    for (const msg of processedInitial) {
      if (
        msg.type === "tool_response" ||
        msg.type === "agent_carousel" ||
        msg.type === "execution_started"
      ) {
        const toolId = (msg as any).toolId;
        if (toolId) {
          completedToolIds.add(toolId);
        }
      }
    }

    // Debug: Log operation filtering when relevant
    const operationMsgs = messages.filter(
      (m) =>
        m.type === "operation_started" ||
        m.type === "operation_pending" ||
        m.type === "operation_in_progress",
    );
    if (operationMsgs.length > 0) {
      console.info("[useChatContainer] Operation deduplication check:", {
        completedToolIds: Array.from(completedToolIds),
        localOperations: operationMsgs.map((m) => ({
          type: m.type,
          toolId: (m as any).toolId,
          toolName: (m as any).toolName,
        })),
        initialMessagesCount: initialMessages.length,
        processedInitialCount: processedInitial.length,
      });
    }

    // Filter local messages to remove operation messages for completed tools
    const filteredLocalMessages = messages.filter((msg) => {
      if (
        msg.type === "operation_started" ||
        msg.type === "operation_pending" ||
        msg.type === "operation_in_progress"
      ) {
        const toolId = (msg as any).toolId || (msg as any).toolCallId;
        if (toolId && completedToolIds.has(toolId)) {
          return false; // Filter out - operation completed
        }
      }
      return true;
    });

    const combined = [...processedInitial, ...filteredLocalMessages];

    // Deduplicate by content+role. When initialMessages is refreshed via polling,
    // it may contain messages that are also in the local `messages` state.
    const seen = new Set<string>();
    return combined.filter((msg) => {
      // Create a key based on type, role, and content for deduplication
      let key: string;
      if (msg.type === "message") {
        key = `msg:${msg.role}:${msg.content}`;
      } else if (msg.type === "tool_call") {
        key = `toolcall:${msg.toolId}`;
      } else if (
        msg.type === "operation_started" ||
        msg.type === "operation_pending" ||
        msg.type === "operation_in_progress"
      ) {
        // Dedupe operation messages by toolId or operationId
        key = `op:${(msg as any).toolId || (msg as any).operationId || (msg as any).toolCallId || ""}:${msg.toolName}`;
      } else {
        // For other types, use a combination of type and first few fields
        key = `${msg.type}:${JSON.stringify(msg).slice(0, 100)}`;
      }
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [initialMessages, messages]);

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

      if (err instanceof Error && err.name === "AbortError") return;

      const errorMessage =
        err instanceof Error ? err.message : "Failed to send message";
      toast.error("Failed to send message", {
        description: errorMessage,
      });
    }
  }

  function handleStopStreaming() {
    stopStreaming();
    setStreamingChunks([]);
    streamingChunksRef.current = [];
    setHasTextChunks(false);
    setIsStreamingInitiated(false);
  }

  const { capturePageContext } = usePageContext();
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;

  useEffect(
    function handleInitialPrompt() {
      if (!initialPrompt || !sessionId) return;
      if (initialMessages.length > 0) return;
      if (hasSentInitialPrompt(sessionId)) return;

      markInitialPromptSent(sessionId);
      const context = capturePageContext();
      sendMessageRef.current(initialPrompt, true, context);
    },
    [initialPrompt, sessionId, initialMessages.length, capturePageContext],
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
