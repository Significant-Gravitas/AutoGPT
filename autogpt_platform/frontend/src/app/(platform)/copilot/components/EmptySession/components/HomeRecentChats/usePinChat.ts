import { usePatchV2UpdateSessionPinned } from "@/app/api/__generated__/endpoints/chat/chat";
import { SESSION_GROUP_PAGES_QUERY_KEY } from "@/app/(platform)/copilot/sessionGroups";
import { SESSION_LIST_QUERY_KEY } from "@/app/(platform)/copilot/useSessionList";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

/** Pins a chat to Home's pinned box, or takes it back out. Both chat lists
 *  are read again so the chat moves between the boxes. */
export function usePinChat() {
  const queryClient = useQueryClient();
  const { mutate, isPending, variables } = usePatchV2UpdateSessionPinned({
    mutation: {
      onSuccess: () =>
        Promise.all([
          queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY }),
          queryClient.invalidateQueries({
            queryKey: SESSION_GROUP_PAGES_QUERY_KEY,
          }),
        ]),
      onError: (error) =>
        toast({
          title: "The chat could not be updated",
          description: error instanceof Error ? error.message : undefined,
          variant: "destructive",
        }),
    },
  });

  function togglePin(sessionID: string, isPinned: boolean) {
    mutate({ sessionId: sessionID, data: { is_pinned: !isPinned } });
  }

  return {
    togglePin,
    /** The chat whose pin is being changed right now. */
    pendingID: isPending ? (variables?.sessionId ?? null) : null,
  };
}
