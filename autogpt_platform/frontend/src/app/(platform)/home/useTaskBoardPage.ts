import { useGetV2GetBoard } from "@/app/api/__generated__/endpoints/chat/chat";
import { keepPreviousData, useIsMutating } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { usePlatformChrome } from "../PlatformChrome/usePlatformChrome";
import {
  BOARD_MOVE_MUTATION_KEY,
  BOARD_REFETCH_INTERVAL_MS,
  boardQueryKey,
  type Card,
  groupCards,
} from "./board-helpers";

export function useTaskBoardPage() {
  const [includeOlder, setIncludeOlder] = useState(false);
  const [reassignTarget, setReassignTarget] = useState<Card | null>(null);
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  // A poll landing mid-move would snap the card back until the next one.
  const movesInFlight = useIsMutating({ mutationKey: BOARD_MOVE_MUTATION_KEY });
  const { showNewLayout } = usePlatformChrome();
  const params = { include_older: includeOlder };

  const boardQuery = useGetV2GetBoard(params, {
    query: {
      queryKey: boardQueryKey(params),
      refetchInterval: movesInFlight > 0 ? false : BOARD_REFETCH_INTERVAL_MS,
      placeholderData: keepPreviousData,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });
  const board = boardQuery.data;
  const isPanelReady = !!sessionId && (!!board || boardQuery.isError);

  return {
    columns: board ? groupCards(board.columns, board.sessions) : [],
    isEmpty: !!board && board.sessions.length === 0,
    isLoading: boardQuery.isLoading,
    isError: boardQuery.isError && !board,
    errorMessage:
      boardQuery.error instanceof Error
        ? boardQuery.error.message
        : "Something went wrong.",
    includeOlder,
    showNewLayout,
    retry: () => void boardQuery.refetch(),
    toggleOlder: () => setIncludeOlder((value) => !value),
    openCard: (id: string) => void setSessionId(id),
    panel: isPanelReady
      ? {
          sessionId,
          card: board?.sessions.find((card) => card.id === sessionId) ?? null,
          columns: board?.columns ?? [],
        }
      : null,
    closePanel: () => void setSessionId(null),
    reassignTarget,
    startReassign: setReassignTarget,
  };
}
