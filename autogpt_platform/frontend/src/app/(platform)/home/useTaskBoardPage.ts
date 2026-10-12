import {
  useGetV2GetBoard,
  useGetV2GetSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { keepPreviousData, useIsMutating } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { usePlatformChrome } from "../PlatformChrome/usePlatformChrome";
import {
  BOARD_MUTATION_KEY,
  BOARD_REFETCH_INTERVAL_MS,
  boardHref,
  boardQueryKey,
  type Card,
  DEFAULT_BOARD_REF,
  groupCards,
  isBoardRef,
  isNotFoundError,
  isSessionId,
  retryUnlessNotFound,
} from "./board-helpers";
import { useBoardDrag } from "./useBoardDrag";

export function useTaskBoardPage(boardRef: string) {
  const [includeOlder, setIncludeOlder] = useState(false);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [reassignTarget, setReassignTarget] = useState<Card | null>(null);
  const [sessionParam, setSessionId] = useQueryState(
    "sessionId",
    parseAsString,
  );
  // The id ends up in API paths; anything but a UUID opens nothing.
  const sessionId = isSessionId(sessionParam) ? sessionParam : null;
  const isValidRef = isBoardRef(boardRef);
  // A poll landing mid-drag or mid-change would snap the card back.
  const changesInFlight = useIsMutating({ mutationKey: BOARD_MUTATION_KEY });
  const isPollPaused = draggingId !== null || changesInFlight > 0;
  const { showNewLayout } = usePlatformChrome();
  const params = { include_older: includeOlder };

  const boardQuery = useGetV2GetBoard(boardRef, params, {
    query: {
      queryKey: boardQueryKey(boardRef, params),
      enabled: isValidRef,
      refetchInterval: isPollPaused ? false : BOARD_REFETCH_INTERVAL_MS,
      refetchOnWindowFocus: !isPollPaused,
      placeholderData: keepPreviousData,
      retry: retryUnlessNotFound,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });
  // A refetch that finds the board gone keeps the last data, so drop it.
  const isNotFound = !isValidRef || isNotFoundError(boardQuery.error);
  const board = isNotFound ? undefined : boardQuery.data;
  const isDefault = boardRef === DEFAULT_BOARD_REF || !!board?.board.is_default;
  const columns = board ? groupCards(board.columns, board.sessions) : [];
  const openCard = board?.sessions.find((card) => card.id === sessionId);
  const isOffBoard =
    !!sessionId && !openCard && !isNotFound && (!!board || boardQuery.isError);

  // A link to a session on another board goes to that board.
  const sessionBoardQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: isOffBoard,
      select: (res) =>
        res.status === 200 ? (res.data.board_id ?? null) : null,
    },
  });
  const sessionBoardId = sessionBoardQuery.data ?? null;
  const isOtherBoard =
    sessionBoardQuery.isSuccess &&
    sessionBoardId !== (isDefault ? null : boardRef);
  const isPanelReady =
    !!openCard ||
    (isOffBoard &&
      (sessionBoardQuery.isError ||
        (sessionBoardQuery.isSuccess && !isOtherBoard)));

  const drag = useBoardDrag({
    boardRef,
    columns,
    onDraggingChange: setDraggingId,
  });

  return {
    boardContext: { boardRef, isDefault },
    name: isDefault ? "Home" : (board?.board.name ?? null),
    redirectHref:
      isOffBoard && isOtherBoard && sessionId
        ? boardHref(sessionBoardId, sessionId)
        : null,
    columns,
    isEmpty: !!board && board.sessions.length === 0,
    isLoading: boardQuery.isLoading,
    isNotFound,
    isError: boardQuery.isError && !board && !isNotFound,
    errorMessage:
      boardQuery.error instanceof Error
        ? boardQuery.error.message
        : "Something went wrong.",
    includeOlder,
    showNewLayout,
    retry: () => void boardQuery.refetch(),
    toggleOlder: () => setIncludeOlder((value) => !value),
    openCard: (id: string) => void setSessionId(id),
    drag,
    draggingCard:
      board?.sessions.find((card) => card.id === draggingId) ?? null,
    panel:
      sessionId && isPanelReady
        ? {
            sessionId,
            card: openCard ?? null,
            columns: board?.columns ?? [],
          }
        : null,
    closePanel: () => void setSessionId(null),
    reassignTarget,
    startReassign: setReassignTarget,
    closeReassign: () => setReassignTarget(null),
    openReassigned: (id: string) => void setSessionId(id),
  };
}
