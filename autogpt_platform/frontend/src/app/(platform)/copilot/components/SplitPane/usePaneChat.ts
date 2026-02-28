/**
 * Per-pane chat hook. This is a simplified version of useCopilotPage
 * that manages a single chat session for one pane in the split view.
 */

import {
  getGetV2GetSessionQueryKey,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import type { UIMessage } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePaneChatSession } from "./usePaneChatSession";

const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;
const RECONNECT_MAX_ATTEMPTS = 5;

function resolveInProgressTools(
  messages: UIMessage[],
  outcome: "completed" | "cancelled",
): UIMessage[] {
  return messages.map((msg) => ({
    ...msg,
    parts: msg.parts.map((part) =>
      "state" in part &&
      (part.state === "input-streaming" || part.state === "input-available")
        ? outcome === "cancelled"
          ? { ...part, state: "output-error" as const, errorText: "Cancelled" }
          : { ...part, state: "output-available" as const, output: "" }
        : part,
    ),
  }));
}

function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seenIds = new Set<string>();
  return messages.filter((msg) => {
    if (seenIds.has(msg.id)) return false;
    seenIds.add(msg.id);
    return true;
  });
}

interface UsePaneChatArgs {
  paneId: string;
  sessionId: string | null;
  onSessionChange: (paneId: string, sessionId: string | null) => void;
}

export function usePaneChat({
  paneId,
  sessionId: externalSessionId,
  onSessionChange,
}: UsePaneChatArgs) {
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const setSessionId = useCallback(
    (id: string | null) => {
      onSessionChange(paneId, id);
    },
    [paneId, onSessionChange],
  );

  const {
    sessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
  } = usePaneChatSession({
    sessionId: externalSessionId,
    setSessionId,
  });

  const transport = useMemo(
    () =>
      sessionId
        ? new DefaultChatTransport({
            api: `/api/chat/sessions/${sessionId}/stream`,
            prepareSendMessagesRequest: ({ messages }) => {
              const last = messages[messages.length - 1];
              return {
                body: {
                  message: (
                    last.parts?.map((p) => (p.type === "text" ? p.text : "")) ??
                    []
                  ).join(""),
                  is_user_message: last.role === "user",
                  context: null,
                },
              };
            },
            prepareReconnectToStreamRequest: () => ({
              api: `/api/chat/sessions/${sessionId}/stream`,
            }),
          })
        : null,
    [sessionId],
  );

  // Reconnect state
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [isReconnectScheduled, setIsReconnectScheduled] = useState(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const hasShownDisconnectToast = useRef(false);

  function handleReconnect(sid: string) {
    if (isReconnectScheduled || !sid) return;

    const nextAttempt = reconnectAttempts + 1;
    if (nextAttempt > RECONNECT_MAX_ATTEMPTS) {
      toast({
        title: "Connection lost",
        description: "Unable to reconnect. Please refresh the page.",
        variant: "destructive",
      });
      return;
    }

    setIsReconnectScheduled(true);
    setReconnectAttempts(nextAttempt);

    if (!hasShownDisconnectToast.current) {
      hasShownDisconnectToast.current = true;
      toast({
        title: "Connection lost",
        description: "Reconnecting...",
      });
    }

    const delay = Math.min(
      RECONNECT_BASE_DELAY_MS * 2 ** reconnectAttempts,
      RECONNECT_MAX_DELAY_MS,
    );

    reconnectTimerRef.current = setTimeout(() => {
      setIsReconnectScheduled(false);
      resumeStream();
    }, delay);
  }

  const {
    messages: rawMessages,
    sendMessage,
    stop: sdkStop,
    status,
    error,
    setMessages,
    resumeStream,
  } = useChat({
    id: sessionId ? `${paneId}-${sessionId}` : undefined,
    transport: transport ?? undefined,
    onFinish: async ({ isDisconnect, isAbort }) => {
      if (isAbort || !sessionId) return;

      if (isDisconnect) {
        handleReconnect(sessionId);
        return;
      }

      const result = await refetchSession();
      const backendActive =
        result.data?.status === 200 && !!result.data.data.active_stream;

      if (backendActive) {
        handleReconnect(sessionId);
      }
    },
    onError: (error) => {
      if (!sessionId) return;
      const isNetworkError =
        error.name === "TypeError" || error.name === "AbortError";
      if (isNetworkError) {
        handleReconnect(sessionId);
      }
    },
  });

  const messages = useMemo(
    () => deduplicateMessages(rawMessages),
    [rawMessages],
  );

  async function stop() {
    sdkStop();
    setMessages((prev) => resolveInProgressTools(prev, "cancelled"));

    if (!sessionId) return;
    try {
      const res = await postV2CancelSessionTask(sessionId);
      if (
        res.status === 200 &&
        "reason" in res.data &&
        res.data.reason === "cancel_published_not_confirmed"
      ) {
        toast({
          title: "Stop may take a moment",
          description:
            "The cancel was sent but not yet confirmed. The task should stop shortly.",
        });
      }
    } catch {
      toast({
        title: "Could not stop the task",
        description: "The task may still be running in the background.",
        variant: "destructive",
      });
    }
  }

  // Hydrate messages from REST API when not actively streaming
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    if (isReconnectScheduled) return;
    setMessages((prev) => {
      if (prev.length >= hydratedMessages.length) return prev;
      return deduplicateMessages(hydratedMessages);
    });
  }, [hydratedMessages, setMessages, status, isReconnectScheduled]);

  // Track resume state per session
  const hasResumedRef = useRef<Map<string, boolean>>(new Map());

  // Clean up reconnect state on session switch
  useEffect(() => {
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = undefined;
    setReconnectAttempts(0);
    setIsReconnectScheduled(false);
    hasShownDisconnectToast.current = false;
    prevStatusRef.current = status;
  }, [sessionId, status]);

  // Invalidate session cache when stream completes
  const prevStatusRef = useRef(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isIdle = status === "ready" || status === "error";

    if (wasActive && isIdle && sessionId && !isReconnectScheduled) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
      if (status === "ready") {
        setReconnectAttempts(0);
        hasShownDisconnectToast.current = false;
      }
    }
  }, [status, sessionId, queryClient, isReconnectScheduled]);

  // Resume an active stream after hydration
  useEffect(() => {
    if (!sessionId) return;
    if (!hasActiveStream) return;
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    if (hasResumedRef.current.get(sessionId)) return;

    hasResumedRef.current.set(sessionId, true);
    resumeStream();
  }, [sessionId, hasActiveStream, hydratedMessages, status, resumeStream]);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

  // Send pending message once session is created
  useEffect(() => {
    if (!sessionId || !pendingMessage) return;
    const msg = pendingMessage;
    setPendingMessage(null);
    sendMessage({ text: msg });
  }, [sessionId, pendingMessage, sendMessage]);

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;

    if (sessionId) {
      sendMessage({ text: trimmed });
      return;
    }

    setPendingMessage(trimmed);
    await createSession();
  }

  const isReconnecting =
    isReconnectScheduled ||
    (hasActiveStream && status !== "streaming" && status !== "submitted");

  return {
    sessionId,
    messages,
    status,
    error: isReconnecting ? undefined : error,
    stop,
    isReconnecting,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    createSession,
    onSend,
  };
}
