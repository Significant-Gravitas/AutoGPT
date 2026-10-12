import { useGetV2GetBoard } from "@/app/api/__generated__/endpoints/chat/chat";
import { keepPreviousData } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { usePlatformChrome } from "../PlatformChrome/usePlatformChrome";
import {
  BOARD_REFETCH_INTERVAL_MS,
  boardQueryKey,
  groupCards,
} from "./board-helpers";

export function useTaskBoardPage() {
  const [includeOlder, setIncludeOlder] = useState(false);
  const [, setSessionId] = useQueryState("sessionId", parseAsString);
  const { showNewLayout } = usePlatformChrome();
  const params = { include_older: includeOlder };

  const boardQuery = useGetV2GetBoard(params, {
    query: {
      queryKey: boardQueryKey(params),
      refetchInterval: BOARD_REFETCH_INTERVAL_MS,
      placeholderData: keepPreviousData,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });
  const board = boardQuery.data;

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
    openCard: (sessionId: string) => void setSessionId(sessionId),
  };
}
