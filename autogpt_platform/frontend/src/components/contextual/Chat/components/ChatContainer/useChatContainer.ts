import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useEffect, useMemo, useRef, useState } from "react";
import { INITIAL_STREAM_ID } from "../../chat-constants";
import { useChatStore } from "../../chat-store";
import { toast } from "sonner";
import { useChatStream } from "../../useChatStream";
import { usePageContext } from "../../usePageContext";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import {
  getToolIdFromMessage,
  hasToolId,
  isOperationMessage,
} from "../../chat-types";
import { createStreamEventDispatcher } from "./createStreamEventDispatcher";
import {
  createUserMessage,
  filterAuthMessages,
  hasSentInitialPrompt,
  markInitialPromptSent,
  processInitialMessages,
} from "./helpers";

/**
 * Dependencies for creating a stream event dispatcher.
 * Extracted to allow helper function creation.
 */
interface DispatcherDeps {
  setHasTextChunks: React.Dispatch<React.SetStateAction<boolean>>;
  setStreamingChunks: React.Dispatch<React.SetStateAction<string[]>>;
  streamingChunksRef: React.MutableRefObject<string[]>;
  hasResponseRef: React.MutableRefObject<boolean>;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessageData[]>>;
  setIsRegionBlockedModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  sessionId: string;
  setIsStreamingInitiated: React.Dispatch<React.SetStateAction<boolean>>;
  onOperationStarted?: () => void;
  onActiveTaskStarted: (taskInfo: {
    taskId: string;
    operationId: string;
    toolName: string;
    toolCallId: string;
  }) => void;
}

/**
 * Create a stream event dispatcher with the given dependencies.
 */
function createDispatcher(deps: DispatcherDeps) {
  return createStreamEventDispatcher({
    setHasTextChunks: deps.setHasTextChunks,
    setStreamingChunks: deps.setStreamingChunks,
    streamingChunksRef: deps.streamingChunksRef,
    hasResponseRef: deps.hasResponseRef,
    setMessages: deps.setMessages,
    setIsRegionBlockedModalOpen: deps.setIsRegionBlockedModalOpen,
    sessionId: deps.sessionId,
    setIsStreamingInitiated: deps.setIsStreamingInitiated,
    onOperationStarted: deps.onOperationStarted,
    onActiveTaskStarted: deps.onActiveTaskStarted,
  });
}

// Helper to generate deduplication key for a message
function getMessageKey(msg: ChatMessageData): string {
  if (msg.type === "message") {
    // Don't include timestamp - dedupe by role + content only
    // This handles the case where local and server timestamps differ
    // Server messages are authoritative, so duplicates from local state are filtered
    return `msg:${msg.role}:${msg.content}`;
  } else if (msg.type === "tool_call") {
    return `toolcall:${msg.toolId}`;
  } else if (msg.type === "tool_response") {
    const toolId = hasToolId(msg) ? msg.toolId : "";
    return `toolresponse:${toolId}`;
  } else if (isOperationMessage(msg)) {
    const toolId = getToolIdFromMessage(msg) || "";
    return `op:${toolId}:${msg.toolName}`;
  } else {
    return `${msg.type}:${JSON.stringify(msg).slice(0, 100)}`;
  }
}

interface Args {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  initialPrompt?: string;
  onOperationStarted?: () => void;
  /** Active stream info from the server for reconnection */
  activeStream?: {
    taskId: string;
    lastMessageId: string;
  };
}

export function useChatContainer({
  sessionId,
  initialMessages,
  initialPrompt,
  onOperationStarted,
  activeStream,
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
  const setActiveTask = useChatStore((s) => s.setActiveTask);
  const getActiveTask = useChatStore((s) => s.getActiveTask);
  const reconnectToTask = useChatStore((s) => s.reconnectToTask);
  const isStreaming = isStreamingInitiated || hasTextChunks;
  // Track whether we've already connected to this activeStream to avoid duplicate connections
  const connectedActiveStreamRef = useRef<string | null>(null);

  // Callback to store active task info for SSE reconnection
  function handleActiveTaskStarted(taskInfo: {
    taskId: string;
    operationId: string;
    toolName: string;
    toolCallId: string;
  }) {
    if (!sessionId) return;
    setActiveTask(sessionId, {
      taskId: taskInfo.taskId,
      operationId: taskInfo.operationId,
      toolName: taskInfo.toolName,
      lastMessageId: INITIAL_STREAM_ID,
    });
  }

  useEffect(
    function handleSessionChange() {
      const isSessionChange = sessionId !== previousSessionIdRef.current;

      console.info("[SSE-RECONNECT] handleSessionChange effect running:", {
        sessionId,
        previousSessionId: previousSessionIdRef.current,
        isSessionChange,
        hasActiveStream: !!activeStream,
        activeStreamTaskId: activeStream?.taskId,
        connectedActiveStream: connectedActiveStreamRef.current,
      });

      // Handle session change - reset state
      if (isSessionChange) {
        console.info("[SSE-RECONNECT] Session changed, resetting state");
        const prevSession = previousSessionIdRef.current;
        if (prevSession) {
          stopStreaming(prevSession);
        }
        previousSessionIdRef.current = sessionId;
        connectedActiveStreamRef.current = null; // Reset connected stream tracker
        setMessages([]);
        setStreamingChunks([]);
        streamingChunksRef.current = [];
        setHasTextChunks(false);
        setIsStreamingInitiated(false);
        hasResponseRef.current = false;
      }

      if (!sessionId) {
        console.info("[SSE-RECONNECT] No sessionId, skipping reconnection check");
        return;
      }

      // Priority 1: Check if server told us there's an active stream (most authoritative)
      // Also handles the case where activeStream arrives after initial session load
      if (activeStream) {
        // Skip if we've already connected to this exact stream
        // Check and set immediately to prevent race conditions from effect re-runs
        const streamKey = `${sessionId}:${activeStream.taskId}`;
        if (connectedActiveStreamRef.current === streamKey) {
          console.info(
            "[SSE-RECONNECT] Already connected to this stream, skipping:",
            { streamKey },
          );
          return;
        }

        // Also skip if there's already an active stream for this session in the store
        // (handles case where effect re-runs due to activeStreams state change)
        const existingStream = activeStreams.get(sessionId);
        if (existingStream && existingStream.status === "streaming") {
          console.info(
            "[SSE-RECONNECT] Active stream already exists in store, skipping:",
            { sessionId, status: existingStream.status },
          );
          connectedActiveStreamRef.current = streamKey;
          return;
        }

        // Set immediately after check to prevent race conditions
        connectedActiveStreamRef.current = streamKey;

        console.info(
          "[SSE-RECONNECT] Server reports active stream, initiating reconnection:",
          {
            sessionId,
            taskId: activeStream.taskId,
            lastMessageId: activeStream.lastMessageId,
            streamKey,
          },
        );

        const dispatcher = createDispatcher({
          setHasTextChunks,
          setStreamingChunks,
          streamingChunksRef,
          hasResponseRef,
          setMessages,
          setIsRegionBlockedModalOpen,
          sessionId,
          setIsStreamingInitiated,
          onOperationStarted,
          onActiveTaskStarted: handleActiveTaskStarted,
        });

        setIsStreamingInitiated(true);
        // Store this as the active task for future reconnects
        setActiveTask(sessionId, {
          taskId: activeStream.taskId,
          operationId: activeStream.taskId,
          toolName: "chat",
          lastMessageId: activeStream.lastMessageId,
        });
        // Reconnect to the task stream
        console.info("[SSE-RECONNECT] Calling reconnectToTask...");
        reconnectToTask(
          sessionId,
          activeStream.taskId,
          activeStream.lastMessageId,
          dispatcher,
        );
        return;
      }

      // Only check localStorage/in-memory on session change, not on every render
      if (!isSessionChange) {
        console.info(
          "[SSE-RECONNECT] No active stream and not a session change, skipping fallbacks",
        );
        return;
      }

      // Priority 2: Check localStorage for active task (client-side state)
      console.info("[SSE-RECONNECT] Checking localStorage for active task...");
      const activeTask = getActiveTask(sessionId);
      if (activeTask) {
        console.info(
          "[SSE-RECONNECT] Found active task in localStorage, attempting reconnect:",
          {
            sessionId,
            taskId: activeTask.taskId,
            lastMessageId: activeTask.lastMessageId,
          },
        );

        const dispatcher = createDispatcher({
          setHasTextChunks,
          setStreamingChunks,
          streamingChunksRef,
          hasResponseRef,
          setMessages,
          setIsRegionBlockedModalOpen,
          sessionId,
          setIsStreamingInitiated,
          onOperationStarted,
          onActiveTaskStarted: handleActiveTaskStarted,
        });

        setIsStreamingInitiated(true);
        // Reconnect to the task stream
        console.info("[SSE-RECONNECT] Calling reconnectToTask from localStorage...");
        reconnectToTask(
          sessionId,
          activeTask.taskId,
          activeTask.lastMessageId,
          dispatcher,
        );
        return;
      } else {
        console.info("[SSE-RECONNECT] No active task in localStorage");
      }

      // Priority 3: Check for an in-memory active stream (same-tab scenario)
      console.info("[SSE-RECONNECT] Checking in-memory active streams...");
      const inMemoryStream = activeStreams.get(sessionId);
      if (!inMemoryStream || inMemoryStream.status !== "streaming") {
        console.info("[SSE-RECONNECT] No in-memory active stream found:", {
          hasStream: !!inMemoryStream,
          status: inMemoryStream?.status,
        });
        return;
      }

      const dispatcher = createDispatcher({
        setHasTextChunks,
        setStreamingChunks,
        streamingChunksRef,
        hasResponseRef,
        setMessages,
        setIsRegionBlockedModalOpen,
        sessionId,
        setIsStreamingInitiated,
        onOperationStarted,
        onActiveTaskStarted: handleActiveTaskStarted,
      });

      setIsStreamingInitiated(true);
      const skipReplay = initialMessages.length > 0;
      return subscribeToStream(sessionId, dispatcher, skipReplay);
    },
    [
      sessionId,
      stopStreaming,
      activeStreams,
      subscribeToStream,
      onOperationStarted,
      getActiveTask,
      reconnectToTask,
      activeStream,
      setActiveTask,
    ],
  );

  // Collect toolIds from completed tool results in initialMessages
  // Used to filter out operation messages when their results arrive
  const completedToolIds = useMemo(() => {
    const processedInitial = processInitialMessages(initialMessages);
    const ids = new Set<string>();
    for (const msg of processedInitial) {
      if (
        msg.type === "tool_response" ||
        msg.type === "agent_carousel" ||
        msg.type === "execution_started"
      ) {
        const toolId = hasToolId(msg) ? msg.toolId : undefined;
        if (toolId) {
          ids.add(toolId);
        }
      }
    }
    return ids;
  }, [initialMessages]);

  // Clean up local operation messages when their completed results arrive from polling
  // This effect runs when completedToolIds changes (i.e., when polling brings new results)
  useEffect(
    function cleanupCompletedOperations() {
      if (completedToolIds.size === 0) return;

      setMessages((prev) => {
        const filtered = prev.filter((msg) => {
          if (isOperationMessage(msg)) {
            const toolId = getToolIdFromMessage(msg);
            if (toolId && completedToolIds.has(toolId)) {
              return false; // Remove - operation completed
            }
          }
          return true;
        });
        // Only update state if something was actually filtered
        return filtered.length === prev.length ? prev : filtered;
      });
    },
    [completedToolIds],
  );

  // Combine initial messages from backend with local streaming messages,
  // Server messages maintain correct order; only append truly new local messages
  const allMessages = useMemo(() => {
    const processedInitial = processInitialMessages(initialMessages);

    // Build a set of keys from server messages for deduplication
    const serverKeys = new Set<string>();
    for (const msg of processedInitial) {
      serverKeys.add(getMessageKey(msg));
    }

    // Filter local messages: remove duplicates and completed operation messages
    const newLocalMessages = messages.filter((msg) => {
      // Remove operation messages for completed tools
      if (isOperationMessage(msg)) {
        const toolId = getToolIdFromMessage(msg);
        if (toolId && completedToolIds.has(toolId)) {
          return false;
        }
      }
      // Remove messages that already exist in server data
      const key = getMessageKey(msg);
      return !serverKeys.has(key);
    });

    // Server messages first (correct order), then new local messages
    return [...processedInitial, ...newLocalMessages];
  }, [initialMessages, messages, completedToolIds]);

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

    const dispatcher = createDispatcher({
      setHasTextChunks,
      setStreamingChunks,
      streamingChunksRef,
      hasResponseRef,
      setMessages,
      setIsRegionBlockedModalOpen,
      sessionId,
      setIsStreamingInitiated,
      onOperationStarted,
      onActiveTaskStarted: handleActiveTaskStarted,
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
