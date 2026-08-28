import {
  getV2GetSession,
  patchV2UpdateSessionPinned,
  patchV2UpdateSessionTitle,
  usePatchV2UpdateSessionPinned,
  usePatchV2UpdateSessionTitle,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { fetchAndExportChat } from "@/app/(platform)/copilot/helpers/exportChatAsMarkdown";
import {
  getSessionListQueryKey,
  useSessionList,
} from "@/app/(platform)/copilot/useSessionList";
import { useSessionDeletion } from "@/app/(platform)/copilot/useSessionDeletion";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useSearchParams } from "next/navigation";
import { useState } from "react";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";

export interface RecentChatSession {
  id: string;
  title?: string | null;
  source_platform?: string | null;
  is_processing?: boolean | null;
  is_pinned?: boolean | null;
  updated_at: string;
  organization_id?: string | null;
  team_id?: string | null;
}

export function useRecentChats() {
  const queryClient = useQueryClient();
  const { sessions, isLoading, hasMore, isLoadingMore, loadMore } =
    useSessionList();

  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeSessionId =
    pathname === "/copilot" ? searchParams.get("sessionId") : null;

  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [exportingIds, setExportingIds] = useState<Set<string>>(new Set());
  const [sharingSession, setSharingSession] =
    useState<RecentChatSession | null>(null);

  function getSession(id: string) {
    const session = sessions.find((candidate) => candidate.id === id);
    if (!session) throw new Error(`Chat session ${id} is no longer available`);
    return session;
  }

  function invalidateSessionList(id: string) {
    const session = getSession(id);
    return queryClient.invalidateQueries({
      queryKey: getSessionListQueryKey(
        session.organization_id,
        session.team_id,
      ),
    });
  }

  const {
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  } = useSessionDeletion();

  const { mutate: setSessionPinned } = usePatchV2UpdateSessionPinned({
    mutation: {
      mutationFn: ({ sessionId, data }) => {
        const session = getSession(sessionId);
        return patchV2UpdateSessionPinned(
          sessionId,
          data,
          getTenantRequestInit(session.organization_id, session.team_id),
        );
      },
      onSuccess: (_response, variables) => {
        invalidateSessionList(variables.sessionId);
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
      mutationFn: ({ sessionId, data }) => {
        const session = getSession(sessionId);
        return patchV2UpdateSessionTitle(
          sessionId,
          data,
          getTenantRequestInit(session.organization_id, session.team_id),
        );
      },
      onSuccess: (_response, variables) => {
        invalidateSessionList(variables.sessionId);
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

  function togglePin(id: string, isPinned: boolean) {
    setSessionPinned({ sessionId: id, data: { is_pinned: !isPinned } });
  }

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
      const session = getSession(id);
      await fetchAndExportChat(
        id,
        title,
        getV2GetSession,
        getTenantRequestInit(session.organization_id, session.team_id),
      );
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
    cancelRename: () => setEditingSessionId(null),
    exportingIds,
    exportChat,
    sharingSession,
    setSharingSession,
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  };
}
