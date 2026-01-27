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

      if (initialMessages.length === 0 && activeStream.chunks.length > 0) {
        for (const chunk of activeStream.chunks) {
          dispatcher(chunk);
        }
      }

      setIsStreamingInitiated(true);
      return subscribeToStream(sessionId, dispatcher);
    },
    [sessionId, stopStreaming, activeStreams, subscribeToStream],
  );

  const allMessages = useMemo(
    () => [...processInitialMessages(initialMessages), ...messages],
    [initialMessages, messages],
  );

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
