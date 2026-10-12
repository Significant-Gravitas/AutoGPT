import type { getV2GetBoardResponse } from "@/app/api/__generated__/endpoints/chat/chat";
import type { BoardResponse } from "@/app/api/__generated__/models/boardResponse";
import { type QueryKey, useQueryClient } from "@tanstack/react-query";
import { BOARD_MUTATION_KEY, boardScopeKey } from "./board-helpers";

export type BoardSnapshot = [QueryKey, getV2GetBoardResponse | undefined][];

// Every read and write stays inside one board's cache entries, so a change
// on one board never touches another board's cached cards.
export function useBoardCache(boardRef: string) {
  const queryClient = useQueryClient();
  const queryKey = boardScopeKey(boardRef);

  async function applyOptimistic(
    transform: (data: BoardResponse) => BoardResponse,
  ): Promise<BoardSnapshot> {
    await queryClient.cancelQueries({ queryKey });
    const snapshot = queryClient.getQueriesData<getV2GetBoardResponse>({
      queryKey,
    });
    queryClient.setQueriesData<getV2GetBoardResponse>({ queryKey }, (old) =>
      old?.status === 200 ? { ...old, data: transform(old.data) } : old,
    );
    return snapshot;
  }

  function rollback(snapshot: BoardSnapshot | undefined) {
    for (const [key, data] of snapshot ?? []) {
      queryClient.setQueryData(key, data);
    }
  }

  function invalidateBoard() {
    return queryClient.invalidateQueries({ queryKey });
  }

  // Called from onSettled, where the settling mutation still counts: while
  // another board change is in flight, its refetch would undo that change's
  // optimistic update, so leave the refetch to whichever settles last.
  function settleBoard() {
    if (queryClient.isMutating({ mutationKey: BOARD_MUTATION_KEY }) > 1) return;
    return invalidateBoard();
  }

  return { applyOptimistic, rollback, invalidateBoard, settleBoard };
}
