import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
  type getV2ListSessionsResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import type { FileUIPart } from "ai";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef, useState } from "react";
import { concatWithAssistantMerge } from "./helpers/convertChatSessionToUiMessages";
import { useCopilotUIStore } from "./store";
import { useChatSession } from "./useChatSession";
import { useCopilotNotifications } from "./useCopilotNotifications";
import { useCopilotStream } from "./useCopilotStream";
import { useLoadMoreMessages } from "./useLoadMoreMessages";
import { useWorkflowImportAutoSubmit } from "./useWorkflowImportAutoSubmit";

const TITLE_POLL_INTERVAL_MS = 2_000;
const TITLE_POLL_MAX_ATTEMPTS = 5;

interface UploadedFile {
  file_id: string;
  name: string;
  mime_type: string;
}

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const isModeToggleEnabled = useGetFlag(Flag.CHAT_MODE_OPTION);

  const {
    sessionToDelete,
    setSessionToDelete,
    isDrawerOpen,
    setDrawerOpen,
    copilotChatMode,
    copilotLlmModel,
    isDryRun,
  } = useCopilotUIStore();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    rawSessionMessages,
    historicalDurations,
    hasActiveStream,
    hasMoreMessages,
    oldestSequence,
    isLoadingSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
    sessionDryRun,
  } = useChatSession({ dryRun: isDryRun });

  const {
    messages: currentMessages,
    sendMessage,
    stop,
    status,
    error,
    isReconnecting,
    isSyncing,
    isUserStoppingRef,
    rateLimitMessage,
    dismissRateLimit,
  } = useCopilotStream({
    sessionId,
    hydratedMessages,
    hasActiveStream,
    refetchSession,
    copilotMode: isModeToggleEnabled ? copilotChatMode : undefined,
    copilotModel: isModeToggleEnabled ? copilotLlmModel : undefined,
  });

  const { olderMessages, hasMore, isLoadingMore, loadMore } =
    useLoadMoreMessages({
      sessionId,
      initialOldestSequence: oldestSequence,
      initialHasMore: hasMoreMessages,
      initialPageRawMessages: rawSessionMessages,
    });

  // Combine older (paginated) messages with current page messages,
  // merging consecutive assistant UIMessages at the page boundary so
  // reasoning + response parts stay in a single bubble.
  const messages = concatWithAssistantMerge(olderMessages, currentMessages);

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

  const pendingFilesRef = useRef<File[]>([]);
  // Pre-built file parts from workflow import (already uploaded, skip re-upload)
  const pendingFilePartsRef = useRef<FileUIPart[]>([]);

  // --- Send pending message after session creation ---
  useEffect(() => {
    if (!sessionId || pendingMessage === null) return;
    const msg = pendingMessage;
    const files = pendingFilesRef.current;
    const prebuiltParts = pendingFilePartsRef.current;
    setPendingMessage(null);
    pendingFilesRef.current = [];
    pendingFilePartsRef.current = [];

    if (prebuiltParts.length > 0) {
      // File already uploaded (e.g. workflow import) — send directly
      sendMessage({ text: msg, files: prebuiltParts });
    } else if (files.length > 0) {
      setIsUploadingFiles(true);
      void uploadFiles(files, sessionId)
        .then((uploaded) => {
          if (uploaded.length === 0) {
            toast({
              title: "File upload failed",
              description: "Could not upload any files. Please try again.",
              variant: "destructive",
            });
            return;
          }
          const fileParts = buildFileParts(uploaded);
          sendMessage({
            text: msg,
            files: fileParts.length > 0 ? fileParts : undefined,
          });
        })
        .finally(() => setIsUploadingFiles(false));
    } else {
      sendMessage({ text: msg });
    }
  }, [sessionId, pendingMessage, sendMessage]);

  // --- Extract prompt from URL hash on mount (e.g. /copilot#prompt=Hello) ---
  useWorkflowImportAutoSubmit({
    createSession,
    setPendingMessage,
    pendingFilePartsRef,
  });

  async function uploadFiles(
    files: File[],
    sid: string,
  ): Promise<UploadedFile[]> {
    const results = await Promise.allSettled(
      files.map(async (file) => {
        try {
          const data = await uploadFileDirect(file, sid);
          if (!data.file_id) throw new Error("No file_id returned");
          return {
            file_id: data.file_id,
            name: data.name || file.name,
            mime_type: data.mime_type || "application/octet-stream",
          } as UploadedFile;
        } catch (err) {
          console.error("File upload failed:", err);
          toast({
            title: "File upload failed",
            description: file.name,
            variant: "destructive",
          });
          throw err;
        }
      }),
    );
    return results
      .filter(
        (r): r is PromiseFulfilledResult<UploadedFile> =>
          r.status === "fulfilled",
      )
      .map((r) => r.value);
  }

  function buildFileParts(uploaded: UploadedFile[]): FileUIPart[] {
    return uploaded.map((f) => ({
      type: "file" as const,
      mediaType: f.mime_type,
      filename: f.name,
      url: `/api/proxy/api/workspace/files/${f.file_id}/download`,
    }));
  }

  async function onSend(message: string, files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed && (!files || files.length === 0)) return;

    // Client-side file limits
    if (files && files.length > 0) {
      const MAX_FILES = 10;
      const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB

      if (files.length > MAX_FILES) {
        toast({
          title: "Too many files",
          description: `You can attach up to ${MAX_FILES} files at once.`,
          variant: "destructive",
        });
        return;
      }

      const oversized = files.filter((f) => f.size > MAX_FILE_SIZE_BYTES);
      if (oversized.length > 0) {
        toast({
          title: "File too large",
          description: `${oversized[0].name} exceeds the 100 MB limit.`,
          variant: "destructive",
        });
        return;
      }
    }

    isUserStoppingRef.current = false;

    if (sessionId) {
      if (files && files.length > 0) {
        setIsUploadingFiles(true);
        try {
          const uploaded = await uploadFiles(files, sessionId);
          if (uploaded.length === 0) {
            // All uploads failed — abort send so chips revert to editable
            throw new Error("All file uploads failed");
          }
          const fileParts = buildFileParts(uploaded);
          sendMessage({
            text: trimmed || "",
            files: fileParts.length > 0 ? fileParts : undefined,
          });
        } finally {
          setIsUploadingFiles(false);
        }
      } else {
        sendMessage({ text: trimmed });
      }
      return;
    }

    setPendingMessage(trimmed || "");
    if (files && files.length > 0) {
      pendingFilesRef.current = files;
    }
    await createSession();
  }

  // --- Session list (for mobile drawer & sidebar) ---
  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(
      { limit: 50 },
      { query: { enabled: !isUserLoading && isLoggedIn } },
    );

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  // Start title polling when stream ends cleanly — sidebar title animates in
  const titlePollRef = useRef<ReturnType<typeof setInterval>>();
  const prevStatusRef = useRef(status);

  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isNowReady = status === "ready";

    if (!wasActive || !isNowReady || !sessionId || isReconnecting) return;

    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey({ limit: 50 }),
    });
    const sid = sessionId;
    let attempts = 0;
    clearInterval(titlePollRef.current);
    titlePollRef.current = setInterval(() => {
      const data = queryClient.getQueryData<getV2ListSessionsResponse>(
        getGetV2ListSessionsQueryKey({ limit: 50 }),
      );
      const hasTitle =
        data?.status === 200 &&
        data.data.sessions.some((s) => s.id === sid && s.title);
      if (hasTitle || attempts >= TITLE_POLL_MAX_ATTEMPTS) {
        clearInterval(titlePollRef.current);
        titlePollRef.current = undefined;
        return;
      }
      attempts += 1;
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey({ limit: 50 }),
      });
    }, TITLE_POLL_INTERVAL_MS);
  }, [status, sessionId, isReconnecting, queryClient]);

  // Clean up polling on session change or unmount
  useEffect(() => {
    return () => {
      clearInterval(titlePollRef.current);
      titlePollRef.current = undefined;
    };
  }, [sessionId]);

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
    isSyncing,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
    isUserLoading,
    isLoggedIn,
    createSession,
    onSend,
    // Pagination
    hasMoreMessages: hasMore,
    isLoadingMore,
    loadMore,
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
    // Historical durations for persisted timer stats
    historicalDurations,
    // Rate limit reset
    rateLimitMessage,
    dismissRateLimit,
    // Dry run dev toggle
    // isDryRun = global preference for NEW sessions (from localStorage).
    // sessionDryRun = actual dry_run value of the CURRENT session (from API).
    // Use isDryRun to configure future sessions; use sessionDryRun to display
    // the current session's simulation state (banner, indicators).
    isDryRun,
    sessionDryRun,
  };
}
