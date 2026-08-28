import { deleteV2DeleteSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotUIStore } from "./store";
import { getSessionListQueryKey } from "./useSessionList";
import type { DeleteTarget } from "./store";

/**
 * Session deletion flow: reads the pending `sessionToDelete` from the store,
 * fires the delete mutation, clears the active session if it was the one
 * deleted, and toasts on error.
 *
 * Consumed by both `ChatSidebar` and `MobileDrawer` so each can render its
 * own `DeleteChatDialog` without duplicating the mutation wiring.
 */
export function useSessionDeletion() {
  const queryClient = useQueryClient();
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const { sessionToDelete, setSessionToDelete } = useCopilotUIStore();

  const { mutate: deleteSession, isPending: isDeleting } = useMutation({
    mutationFn: (target: DeleteTarget) =>
      deleteV2DeleteSession(
        target.id,
        getTenantRequestInit(target.organizationId, target.teamId),
      ),
    onSuccess: (_data, target) => {
      queryClient.invalidateQueries({
        queryKey: getSessionListQueryKey(target.organizationId, target.teamId),
      });
      // Use the mutation's own `variables` — not the closed-over store
      // value — so a rapid open/cancel/open-different sequence can't
      // accidentally clear the wrong active session after the network
      // round-trip.
      if (target.id === sessionId) {
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
  });

  function requestDelete(target: DeleteTarget) {
    if (isDeleting) return;
    setSessionToDelete(target);
  }

  function confirmDelete() {
    if (sessionToDelete) {
      deleteSession(sessionToDelete);
    }
  }

  function cancelDelete() {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }

  return {
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  };
}
