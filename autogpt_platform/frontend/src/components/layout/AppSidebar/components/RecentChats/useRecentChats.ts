import {
  getV2GetSession,
  usePatchV2UpdateSessionPinned,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useExcludeSkillLearningSource } from "@/app/api/__generated__/endpoints/skill-learning/skill-learning";
import { fetchAndExportChat } from "@/app/(platform)/copilot/helpers/exportChatAsMarkdown";
import {
  SESSION_LIST_QUERY_KEY,
  useSessionList,
} from "@/app/(platform)/copilot/useSessionList";
import { useSessionDeletion } from "@/app/(platform)/copilot/useSessionDeletion";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useStableCallback } from "@/hooks/useStableCallback";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { CHAT_SOURCE_KIND } from "@/services/skill-learning/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useSearchParams } from "next/navigation";
import { useState } from "react";

export function useRecentChats() {
  const queryClient = useQueryClient();
  const { sessions, isLoading, hasMore, isLoadingMore, loadMore } =
    useSessionList();

  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeSessionId =
    pathname === "/home" ? searchParams.get("sessionId") : null;

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [exportingIds, setExportingIds] = useState<Set<string>>(new Set());
  const [sharingSessionId, setSharingSessionId] = useState<string | null>(null);

  const {
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  } = useSessionDeletion();

  const { mutate: setSessionPinned } = usePatchV2UpdateSessionPinned({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
      },
      onError: (error) => {
        toast({
          title: "Failed to update chat",
          description:
            error instanceof Error ? error.message : "An error occurred",
          variant: "destructive",
        });
      },
    },
  });

  const { mutate: renameSession } = usePatchV2UpdateSessionTitle({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
        setEditingSessionId(null);
      },
      onError: (error) => {
        toast({
          title: "Failed to rename chat",
          description:
            error instanceof Error ? error.message : "An error occurred",
          variant: "destructive",
        });
        setEditingSessionId(null);
      },
    },
  });

  const { mutate: excludeSource } = useExcludeSkillLearningSource({
    mutation: {
      onSuccess: () => {
        toast({
          title: "Excluded from learning",
          description:
            "This chat will not be used for future skill learning. Skills " +
            "that depended on it were restored or paused.",
        });
      },
      onError: (error) => {
        toast({
          title: "Could not exclude this chat",
          description:
            error instanceof ApiError
              ? error.message
              : "Nothing has been recorded for this chat yet.",
          variant: "destructive",
        });
      },
    },
  });

  // Every handler handed to the rows keeps one identity across renders, so the
  // memoised `RecentChatItem`s only re-render when their own props change.
  const excludeFromLearning = useStableCallback((id: string) => {
    excludeSource({ sourceKind: CHAT_SOURCE_KIND, sourceRef: id });
  });

  const togglePin = useStableCallback((id: string, isPinned: boolean) => {
    setSessionPinned({ sessionId: id, data: { is_pinned: !isPinned } });
  });

  const startRename = useStableCallback(
    (id: string, title: string | null | undefined) => {
      setEditingSessionId(id);
      setEditingTitle(title || "");
    },
  );

  const submitRename = useStableCallback((id: string) => {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      renameSession({ sessionId: id, data: { title: trimmed } });
    } else {
      setEditingSessionId(null);
    }
  });

  const cancelRename = useStableCallback(() => setEditingSessionId(null));

  const stableRequestDelete = useStableCallback(requestDelete);

  const exportChat = useStableCallback(handleExport);

  async function handleExport(id: string, title: string | null | undefined) {
    if (exportingIds.has(id)) return;
    setExportingIds((prev) => new Set(prev).add(id));
    try {
      await fetchAndExportChat(id, title, getV2GetSession);
      toast({ title: "Chat exported" });
    } catch (error) {
      toast({
        title: "Export failed",
        description:
          error instanceof Error
            ? error.message
            : "Could not export this chat. Please try again.",
        variant: "destructive",
      });
    } finally {
      setExportingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  return {
    sessions,
    isLoading,
    hasMore,
    isLoadingMore,
    loadMore,
    activeSessionId,
    togglePin,
    editingSessionId,
    editingTitle,
    setEditingTitle,
    startRename,
    submitRename,
    cancelRename,
    exportingIds,
    exportChat,
    sharingSessionId,
    setSharingSessionId,
    excludeFromLearning,
    sessionToDelete,
    isDeleting,
    requestDelete: stableRequestDelete,
    confirmDelete,
    cancelDelete,
  };
}
