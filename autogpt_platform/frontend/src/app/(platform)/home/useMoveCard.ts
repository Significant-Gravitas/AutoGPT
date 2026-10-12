import { usePatchV2UpdateSessionBoard } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  BOARD_MOVE_MUTATION_KEY,
  moveCard,
  moveCardToTop,
  refusalReason,
} from "./board-helpers";
import { type BoardSnapshot, useBoardCache } from "./useBoardCache";

function wasStarted(snapshot: BoardSnapshot | undefined, cardId: string) {
  return (snapshot ?? []).some(
    ([, response]) =>
      response?.status === 200 &&
      response.data.sessions.some((card) => card.id === cardId && card.started),
  );
}

export function useMoveCard(boardRef: string) {
  const { applyOptimistic, rollback, settleBoard } = useBoardCache(boardRef);

  const { mutate } = usePatchV2UpdateSessionBoard({
    mutation: {
      mutationKey: BOARD_MOVE_MUTATION_KEY,
      onMutate: ({ sessionId, data }) =>
        applyOptimistic((board) =>
          data.position == null
            ? moveCardToTop(board, sessionId, data.column ?? "")
            : moveCard(board, sessionId, data.column ?? "", data.position),
        ),
      onSuccess: (response, { sessionId }, snapshot) => {
        if (response.status !== 200 || wasStarted(snapshot, sessionId)) return;
        if (response.data.started || response.data.is_processing) {
          toast({ title: "Started" });
        }
      },
      onError: (error, _variables, snapshot) => {
        rollback(snapshot);
        toast({
          title: refusalReason(error) ?? "Couldn't move the task",
          variant: "destructive",
        });
      },
      onSettled: () => settleBoard(),
    },
  });

  // Without a position the card goes to the top of the column.
  function move(sessionId: string, column: string, position?: number) {
    mutate({ sessionId, data: { column, position } });
  }

  return { move };
}
