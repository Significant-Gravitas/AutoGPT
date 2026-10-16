import { useDeleteV2DeleteSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotUIStore } from "./store";
import {
  SESSION_LIST_QUERY_KEY,
  type SessionListInfiniteData,
  removeSessionFromList,
} from "./useSessionList";

/**
 * Session deletion flow: reads the pending `sessionToDelete` from the store,
 * closes the dialog and removes the chat from the list as soon as the user
 * confirms, then fires the delete mutation. On error the list is restored and
 * a toast is shown; on success the active session is cleared if it was the
 * one deleted.
 *
 * Consumed by `ChatSidebar`, `MobileDrawer` and `RecentChats` so each can
 * render its own `DeleteChatDialog` without duplicating the mutation wiring.
 */
export function useSessionDeletion() {
  const queryClient = useQueryClient();
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const { sessionToDelete, setSessionToDelete } = useCopilotUIStore();

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onMutate: async ({ sessionId: id }) => {
          await queryClient.cancelQueries({ queryKey: SESSION_LIST_QUERY_KEY });
          const previous = queryClient.getQueryData<SessionListInfiniteData>(
            SESSION_LIST_QUERY_KEY,
          );
          queryClient.setQueryData<SessionListInfiniteData>(
            SESSION_LIST_QUERY_KEY,
            (data) => removeSessionFromList(data, id),
          );
          return { previous };
        },
        onSuccess: (_data, variables) => {
          // Use the mutation's own `variables` — not the closed-over store
          // value — so a rapid open/cancel/open-different sequence can't
          // accidentally clear the wrong active session after the network
          // round-trip.
          if (variables.sessionId === sessionId) {
            setSessionId(null);
          }
        },
        onError: (error, _variables, context) => {
          if (context?.previous) {
            queryClient.setQueryData(SESSION_LIST_QUERY_KEY, context.previous);
          }
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
        },
        onSettled: () => {
          queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
        },
      },
    });

  function requestDelete(id: string, title: string | null | undefined) {
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function confirmDelete() {
    if (!sessionToDelete) return;
    deleteSession({ sessionId: sessionToDelete.id });
    setSessionToDelete(null);
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
