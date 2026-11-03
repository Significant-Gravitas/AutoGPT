import { deleteV2DeleteSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { useCopilotUIStore, type DeleteTarget } from "./store";
import {
  getSessionListQueryKey,
  SESSION_DELETE_MUTATION_KEY,
  type SessionListInfiniteData,
  removeSessionFromList,
} from "./useSessionList";

/**
 * Session deletion flow: reads the pending `sessionToDelete` from the store,
 * closes the dialog as soon as the user confirms, leaves the chat if it is
 * the open one (with its `expertId`, so the expert's latest-thread adoption
 * cannot select it again), and fires the delete mutation. `useSessionList`
 * hides the chat while the mutation is pending. On success it is dropped from
 * the cached list; on error the chat shows again and a toast is shown.
 *
 * Consumed by the app sidebar's recent-chats list so it can render its own
 * `DeleteChatDialog` without duplicating the mutation wiring.
 */
export function useSessionDeletion() {
  const queryClient = useQueryClient();
  const [{ sessionId }, setChatParams] = useQueryStates({
    sessionId: parseAsString,
    expertId: parseAsString,
  });
  const sessionToDelete = useCopilotUIStore((s) => s.sessionToDelete);
  const setSessionToDelete = useCopilotUIStore((s) => s.setSessionToDelete);

  const { mutate: deleteSession, isPending: isDeleting } = useMutation({
    mutationKey: SESSION_DELETE_MUTATION_KEY,
    mutationFn: (target: DeleteTarget) =>
      deleteV2DeleteSession(
        target.id,
        getTenantRequestInit(target.organizationId, target.teamId),
      ),
    onSuccess: (_data, target) => {
      queryClient.setQueriesData<SessionListInfiniteData>(
        {
          queryKey: getSessionListQueryKey(
            target.organizationId,
            target.teamId,
          ),
        },
        (data) => removeSessionFromList(data, target.id),
      );
      if (target.id === sessionId) leaveChat();
    },
    onError: (error) => {
      toast({
        title: "Failed to delete chat",
        description:
          error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    },
    onSettled: (_data, _error, target) => {
      queryClient.invalidateQueries({
        queryKey: getSessionListQueryKey(target.organizationId, target.teamId),
      });
    },
  });

  function leaveChat() {
    setChatParams({ sessionId: null, expertId: null });
  }

  function requestDelete(target: DeleteTarget) {
    if (!isDeleting) setSessionToDelete(target);
  }

  function confirmDelete() {
    if (!sessionToDelete) return;
    deleteSession(sessionToDelete);
    if (sessionToDelete.id === sessionId) leaveChat();
    setSessionToDelete(null);
  }

  function cancelDelete() {
    setSessionToDelete(null);
  }

  return {
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  };
}
