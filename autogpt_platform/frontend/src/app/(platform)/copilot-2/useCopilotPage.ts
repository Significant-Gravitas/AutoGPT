import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useState } from "react";
import { useChatSession } from "./useChatSession";

export function useCopilotPage() {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    isLoadingSession,
    createSession,
    isCreatingSession,
  } = useChatSession();

  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const transport = sessionId
    ? new DefaultChatTransport({
        api: `/api/chat/sessions/${sessionId}/stream`,
        prepareSendMessagesRequest: ({ messages }) => {
          const last = messages[messages.length - 1];
          return {
            body: {
              message: last.parts
                ?.map((p) => (p.type === "text" ? p.text : ""))
                .join(""),
              is_user_message: last.role === "user",
              context: null,
            },
          };
        },
      })
    : null;

  const { messages, sendMessage, status, error, setMessages } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
  });

  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    setMessages((prev) => {
      if (prev.length > hydratedMessages.length) return prev;
      return hydratedMessages;
    });
  }, [hydratedMessages, setMessages]);

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
    useGetV2ListSessions({ limit: 50 });

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  const handleOpenDrawer = useCallback(() => {
    setIsDrawerOpen(true);
  }, []);

  const handleCloseDrawer = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  const handleDrawerOpenChange = useCallback((open: boolean) => {
    setIsDrawerOpen(open);
  }, []);

  const handleSelectSession = useCallback(
    (id: string) => {
      setSessionId(id);
      if (isMobile) setIsDrawerOpen(false);
    },
    [setSessionId, isMobile],
  );

  const handleNewChat = useCallback(() => {
    setSessionId(null);
    if (isMobile) setIsDrawerOpen(false);
  }, [setSessionId, isMobile]);

  return {
    sessionId,
    messages,
    status,
    error,
    isLoadingSession,
    isCreatingSession,
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
  };
}
