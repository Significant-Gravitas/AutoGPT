import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { getWebSocketToken } from "@/lib/supabase/actions";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { useQueryClient } from "@tanstack/react-query";
import type { FileUIPart } from "ai";
import { useEffect, useRef, useState } from "react";
import { useCopilotUIStore } from "./store";
import { useChatSession } from "./useChatSession";
import { useCopilotStream } from "./useCopilotStream";

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

  const { sessionToDelete, setSessionToDelete, isDrawerOpen, setDrawerOpen } =
    useCopilotUIStore();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession,
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

  // --- Send pending message after session creation ---
  useEffect(() => {
    if (!sessionId || pendingMessage === null) return;
    const msg = pendingMessage;
    const files = pendingFilesRef.current;
    setPendingMessage(null);
    pendingFilesRef.current = [];

    if (files.length > 0) {
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

  async function uploadFiles(
    files: File[],
    sid: string,
  ): Promise<UploadedFile[]> {
    // Upload directly to the Python backend, bypassing the Next.js serverless
    // proxy.  Vercel's 4.5 MB function payload limit would reject larger files
    // when routed through /api/workspace/files/upload.
    const { token, error: tokenError } = await getWebSocketToken();
    if (tokenError || !token) {
      toast({
        title: "Authentication error",
        description: "Please sign in again.",
        variant: "destructive",
      });
      return [];
    }

    const backendBase = environment.getAGPTServerBaseUrl();

    const results = await Promise.allSettled(
      files.map(async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        const url = new URL("/api/workspace/files/upload", backendBase);
        url.searchParams.set("session_id", sid);
        const res = await fetch(url.toString(), {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });
        if (!res.ok) {
          const err = await res.text();
          console.error("File upload failed:", err);
          toast({
            title: "File upload failed",
            description: file.name,
            variant: "destructive",
          });
          throw new Error(err);
        }
        const data = await res.json();
        if (!data.file_id) throw new Error("No file_id returned");
        return {
          file_id: data.file_id,
          name: data.name || file.name,
          mime_type: data.mime_type || "application/octet-stream",
        } as UploadedFile;
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
    isLoadingSession,
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
