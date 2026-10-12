import {
  getGetV2GetSessionQueryKey,
  useGetV2ListBoards,
  usePatchV2UpdateSessionBoard,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { BOARDS_LIST_QUERY_KEY } from "@/components/layout/AppSidebar/components/ProjectsNavSection/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useState } from "react";
import { toast as sonnerToast } from "sonner";
import {
  BOARD_PROJECT_MUTATION_KEY,
  boardHref,
  boardScopeKey,
  DEFAULT_BOARD_REF,
  refusalReason,
  removeCard,
} from "../../board-helpers";
import { useBoardRef } from "../../BoardContext";
import { useBoardCache } from "../../useBoardCache";
import { type MoveTarget, moveTargets } from "./helpers";

export function useMoveToProject(cardId: string) {
  const queryClient = useQueryClient();
  const { boardRef, isDefault } = useBoardRef();
  const { applyOptimistic, rollback, settleBoard } = useBoardCache(boardRef);
  const [isOpen, setIsOpen] = useState(false);
  const [openSessionId, setOpenSessionId] = useQueryState(
    "sessionId",
    parseAsString,
  );

  const boardsQuery = useGetV2ListBoards({
    query: {
      queryKey: BOARDS_LIST_QUERY_KEY,
      enabled: isOpen,
      select: (res) => (res.status === 200 ? res.data.boards : []),
    },
  });
  const targets = moveTargets(boardsQuery.data ?? [], boardRef, isDefault);

  function showMovedToast(target: MoveTarget) {
    const key = `moved-${cardId}`;
    toast({
      title: `Moved to ${target.name}`,
      replaceKey: key,
      action: (
        <Button
          as="NextLink"
          href={boardHref(target.boardId, cardId)}
          size="small"
          variant="secondary"
          onClick={() => sonnerToast.dismiss(key)}
        >
          View
        </Button>
      ),
    });
  }

  const { mutate } = usePatchV2UpdateSessionBoard({
    mutation: {
      mutationKey: BOARD_PROJECT_MUTATION_KEY,
      onMutate: () => {
        // The card leaves this board, so its panel goes with it.
        if (openSessionId === cardId) void setOpenSessionId(null);
        return applyOptimistic((board) => removeCard(board, cardId));
      },
      onSuccess: (_response, { data }) => {
        const ref = data.board_id ?? DEFAULT_BOARD_REF;
        void queryClient.invalidateQueries({ queryKey: boardScopeKey(ref) });
        // The deep-link redirect reads the session's board from here.
        void queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(cardId),
        });
        const target = targets.find((item) => item.ref === ref);
        if (target) showMovedToast(target);
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

  function moveTo(target: MoveTarget) {
    mutate({ sessionId: cardId, data: { board_id: target.ref } });
  }

  return {
    isOpen,
    setIsOpen,
    targets,
    isLoading: boardsQuery.isLoading,
    moveTo,
  };
}
