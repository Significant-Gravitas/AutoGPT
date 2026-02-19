import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChatSession } from "./useChatSession";
import { useLongRunningToolPolling } from "./hooks/useLongRunningToolPolling";

const STREAM_START_TIMEOUT_MS = 12_000;

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [sessionToDelete, setSessionToDelete] = useState<{
    id: string;
    title: string | null | undefined;
  } | null>(null);
  const queryClient = useQueryClient();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession,
    createSession,
    isCreatingSession,
  } = useChatSession();

  const { mutate: deleteSessionMutation, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          if (sessionToDelete?.id === sessionId) {
            setSessionId(null);
          }
          setSessionToDelete(null);
        },
        onError: (error) => {
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setSessionToDelete(null);
        },
      },
    });

  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

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
            // Resume: GET goes to the same URL as POST (backend uses
            // method to distinguish).  Override the default formula which
            // would append /{chatId}/stream to the existing path.
            prepareReconnectToStreamRequest: () => ({
              api: `/api/chat/sessions/${sessionId}/stream`,
            }),
          })
        : null,
    [sessionId],
  );

  const {
    messages,
    sendMessage,
    stop,
    status,
    error,
    setMessages,
    resumeStream,
  } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
    // Don't use resume: true — it fires before hydration completes, causing
    // the hydrated messages to overwrite the resumed stream.  Instead we
    // call resumeStream() manually after hydration + active_stream detection.
  });

  // Abort the stream if the backend doesn't start sending data within 12s.
  const stopRef = useRef(stop);
  stopRef.current = stop;
  useEffect(() => {
    if (status !== "submitted") return;

    const timer = setTimeout(() => {
      stopRef.current();
      toast({
        title: "Stream timed out",
        description: "The server took too long to respond. Please try again.",
        variant: "destructive",
      });
    }, STREAM_START_TIMEOUT_MS);

    return () => clearTimeout(timer);
  }, [status]);

  // Hydrate messages from the REST session endpoint.
  // Skip hydration while streaming to avoid overwriting the live stream.
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    setMessages((prev) => {
      if (prev.length >= hydratedMessages.length) return prev;
      return hydratedMessages;
    });
  }, [hydratedMessages, setMessages, status]);

  // Resume an active stream AFTER hydration completes.
  // The backend returns active_stream info when a task is still running.
  // We wait for hydration so the AI SDK has the conversation history
  // before the resumed stream appends the in-progress assistant message.
  const hasResumedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!hasActiveStream || !sessionId) return;
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    // Only resume once per session to avoid re-triggering after stream ends
    if (hasResumedRef.current === sessionId) return;
    hasResumedRef.current = sessionId;
    resumeStream();
  }, [hasActiveStream, sessionId, hydratedMessages, status, resumeStream]);

  // Poll session endpoint when a long-running tool (create_agent, edit_agent)
  // is in progress. When the backend completes, the session data will contain
  // the final tool output — this hook detects the change and updates messages.
  useLongRunningToolPolling(sessionId, messages, setMessages);

  // Clear messages when session is null
  useEffect(() => {
    if (!sessionId) setMessages([]);
  }, [sessionId, setMessages]);

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

  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(
      { limit: 50 },
      { query: { enabled: !isUserLoading && isLoggedIn } },
    );

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  function handleOpenDrawer() {
    setIsDrawerOpen(true);
  }

  function handleCloseDrawer() {
    setIsDrawerOpen(false);
  }

  function handleDrawerOpenChange(open: boolean) {
    setIsDrawerOpen(open);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
    if (isMobile) setIsDrawerOpen(false);
  }

  function handleNewChat() {
    setSessionId(null);
    if (isMobile) setIsDrawerOpen(false);
  }

  const handleDeleteClick = useCallback(
    (id: string, title: string | null | undefined) => {
      if (isDeleting) return;
      setSessionToDelete({ id, title });
    },
    [isDeleting],
  );

  const handleConfirmDelete = useCallback(() => {
    if (sessionToDelete) {
      deleteSessionMutation({ sessionId: sessionToDelete.id });
    }
  }, [sessionToDelete, deleteSessionMutation]);

  const handleCancelDelete = useCallback(() => {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }, [isDeleting]);

  // True while we know the backend has an active stream but haven't
  // reconnected yet.  Used to disable the send button and show stop UI.
  const isReconnecting =
    hasActiveStream && status !== "streaming" && status !== "submitted";

  return {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isLoadingSession,
    isCreatingSession,
    isUserLoading,
    isLoggedIn,
    createSession,
    onSend,
    // Mobile drawer
    isMobile,
    isDrawerOpen,
    sessions,
    isLoadingSessions,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleSelectSession,
    handleNewChat,
    // Delete functionality
    sessionToDelete,
    isDeleting,
    handleDeleteClick,
    handleConfirmDelete,
    handleCancelDelete,
  };
}
