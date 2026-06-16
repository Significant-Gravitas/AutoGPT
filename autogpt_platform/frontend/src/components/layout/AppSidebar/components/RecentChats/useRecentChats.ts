import {
  getV2GetSession,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { fetchAndExportChat } from "@/app/(platform)/copilot/helpers/exportChatAsMarkdown";
import {
  SESSION_LIST_QUERY_KEY,
  useSessionList,
} from "@/app/(platform)/copilot/useSessionList";
import { useSessionDeletion } from "@/app/(platform)/copilot/useSessionDeletion";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useRecentChats() {
  const queryClient = useQueryClient();
  const { sessions, isLoading } = useSessionList();

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

  function startRename(id: string, title: string | null | undefined) {
    setEditingSessionId(id);
    setEditingTitle(title || "");
  }

  function submitRename(id: string) {
    const trimmed = editingTitle.trim();
    if (trimmed) {
      renameSession({ sessionId: id, data: { title: trimmed } });
    } else {
      setEditingSessionId(null);
    }
  }

  async function exportChat(id: string, title: string | null | undefined) {
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
    editingSessionId,
    editingTitle,
    setEditingTitle,
    startRename,
    submitRename,
    cancelRename: () => setEditingSessionId(null),
    exportingIds,
    exportChat,
    sharingSessionId,
    setSharingSessionId,
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  };
}
