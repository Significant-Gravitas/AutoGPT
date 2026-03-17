import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import { getSessionListParams } from "./helpers";
import { useCopilotUIStore } from "./store";
import { useCallbackToken } from "./useCallbackToken";
import { useChatSession } from "./useChatSession";
import { useFileUpload } from "./useFileUpload";
import { useCopilotNotifications } from "./useCopilotNotifications";
import { useCopilotStream } from "./useCopilotStream";
import { useTitlePolling } from "./useTitlePolling";

const noop = () => {};

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const queryClient = useQueryClient();
  const listSessionsParams = getSessionListParams();

  const { sessionToDelete, setSessionToDelete, isDrawerOpen, setDrawerOpen } =
    useCopilotUIStore();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession: isLoadingCurrentSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
  } = useChatSession();

  const {
    messages,
    sendMessage,
    stop,
    status,
    error,
    isReconnecting,
    isUserStoppingRef,
  } = useCopilotStream({
    sessionId,
    hydratedMessages,
    hasActiveStream,
    refetchSession,
  });

  useCopilotNotifications(sessionId);

  // --- Delete session ---
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

  // --- Responsive ---
  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const { isConsumingCallbackToken } = useCallbackToken({
    isLoggedIn,
    onConsumed: setSessionId,
    onClearAutopilot: noop,
  });

  const { isUploadingFiles, onSend } = useFileUpload({
    createSession,
    isUserStoppingRef,
    sendMessage,
    sessionId,
  });

  // --- Session list (for mobile drawer & sidebar) ---
  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(listSessionsParams, {
      query: { enabled: !isUserLoading && isLoggedIn },
    });

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  useTitlePolling({
    isReconnecting,
    sessionId,
    status,
  });

  // --- Mobile drawer handlers ---
  function handleOpenDrawer() {
    setDrawerOpen(true);
  }

  function handleCloseDrawer() {
    setDrawerOpen(false);
  }

  function handleDrawerOpenChange(open: boolean) {
    setDrawerOpen(open);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
    if (isMobile) setDrawerOpen(false);
  }

  function handleNewChat() {
    setSessionId(null);
    if (isMobile) setDrawerOpen(false);
  }

  // --- Delete handlers ---
  function handleDeleteClick(id: string, title: string | null | undefined) {
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function handleConfirmDelete() {
    if (sessionToDelete) {
      deleteSessionMutation({ sessionId: sessionToDelete.id });
    }
  }

  function handleCancelDelete() {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }

  return {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isLoadingSession: isLoadingCurrentSession || isConsumingCallbackToken,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
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
