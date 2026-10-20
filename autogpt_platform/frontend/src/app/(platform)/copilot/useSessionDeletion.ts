import { useDeleteV2DeleteSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { useCopilotUIStore } from "./store";
import {
  SESSION_DELETE_MUTATION_KEY,
  SESSION_LIST_QUERY_KEY,
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

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        mutationKey: SESSION_DELETE_MUTATION_KEY,
        onSuccess: (_data, variables) => {
          queryClient.setQueryData<SessionListInfiniteData>(
            SESSION_LIST_QUERY_KEY,
            (data) => removeSessionFromList(data, variables.sessionId),
          );
          // Use the mutation's own `variables` — not the closed-over store
          // value — so a rapid open/cancel/open-different sequence can't
          // accidentally clear the wrong active session after the network
          // round-trip.
          if (variables.sessionId === sessionId) {
            leaveChat();
          }
        },
        onError: (error) => {
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

  function leaveChat() {
    setChatParams({ sessionId: null, expertId: null });
  }

  function requestDelete(id: string, title: string | null | undefined) {
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function confirmDelete() {
    if (!sessionToDelete) return;
    deleteSession({ sessionId: sessionToDelete.id });
    // Leaving now keeps the composer from sending into a chat that is
    // already on its way out.
    if (sessionToDelete.id === sessionId) {
      leaveChat();
    }
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
