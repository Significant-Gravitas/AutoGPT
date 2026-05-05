import {
  getGetV2ListSessionsQueryKey,
  useDeleteV2DeleteSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotUIStore } from "./store";

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

  const { mutate: deleteSession, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: (_data, variables) => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          // Use the mutation's own `variables` — not the closed-over store
          // value — so a rapid open/cancel/open-different sequence can't
          // accidentally clear the wrong active session after the network
          // round-trip.
          if (variables.sessionId === sessionId) {
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

  function requestDelete(id: string, title: string | null | undefined) {
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function confirmDelete() {
    if (sessionToDelete) {
      deleteSession({ sessionId: sessionToDelete.id });
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
