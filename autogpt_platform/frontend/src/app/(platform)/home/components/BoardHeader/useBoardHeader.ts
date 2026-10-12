import {
  getGetV2ListBoardsQueryKey,
  useDeleteV2DeleteBoard,
  usePatchV2UpdateBoard,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useBoardRef } from "../../BoardContext";
import {
  BOARD_MUTATION_KEY,
  boardScopeKey,
  DEFAULT_BOARD_REF,
  MAX_BOARD_NAME,
} from "../../board-helpers";
import { useBoardCache } from "../../useBoardCache";

type Action = "rename" | "delete";

export function useBoardHeader(name: string) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { boardRef } = useBoardRef();
  const { applyOptimistic, rollback, settleBoard } = useBoardCache(boardRef);
  const [isRenaming, setIsRenaming] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [chosen, setChosen] = useState<Action | null>(null);

  function invalidateBoardList() {
    return queryClient.invalidateQueries({
      queryKey: getGetV2ListBoardsQueryKey(),
    });
  }

  const { mutate: renameBoard } = usePatchV2UpdateBoard({
    mutation: {
      mutationKey: [...BOARD_MUTATION_KEY, "rename-board"],
      onMutate: ({ data }) =>
        applyOptimistic((board) => ({
          ...board,
          board: { ...board.board, name: data.name ?? board.board.name },
        })),
      onError: (_error, _variables, snapshot) => {
        rollback(snapshot);
        toast({ title: "Couldn't rename the project", variant: "destructive" });
      },
      onSettled: () => {
        void invalidateBoardList();
        void settleBoard();
      },
    },
  });

  const { mutate: deleteBoard, isPending: isDeleting } = useDeleteV2DeleteBoard(
    {
      mutation: {
        mutationKey: [...BOARD_MUTATION_KEY, "delete-board"],
        onSuccess: () => {
          setIsDeleteOpen(false);
          void invalidateBoardList();
          void queryClient.invalidateQueries({
            queryKey: boardScopeKey(DEFAULT_BOARD_REF),
          });
          router.replace("/home");
        },
        onError: () => {
          toast({
            title: "Couldn't delete the project",
            variant: "destructive",
          });
        },
      },
    },
  );

  function startRename() {
    setDraft(name);
    setError(null);
    setIsRenaming(true);
  }

  function stopRename() {
    setIsRenaming(false);
    setError(null);
  }

  function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    setDraft(event.target.value);
    setError(null);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (isKey(event, "Escape")) {
      event.preventDefault();
      stopRename();
      return;
    }
    if (!isKey(event, "Enter")) return;
    event.preventDefault();
    const next = draft.trim().slice(0, MAX_BOARD_NAME);
    if (!next) {
      setError("Give the project a name.");
      return;
    }
    stopRename();
    if (next !== name) renameBoard({ boardId: boardRef, data: { name: next } });
  }

  // Runs once the menu has closed and let go of focus, so the rename input
  // keeps it; Delete lets focus go back to the menu button first.
  function handleMenuCloseAutoFocus(event: Event) {
    if (!chosen) return;
    setChosen(null);
    if (chosen === "delete") {
      setIsDeleteOpen(true);
      return;
    }
    event.preventDefault();
    startRename();
  }

  function confirmDelete() {
    if (!isDeleting) deleteBoard({ boardId: boardRef });
  }

  function cancelDelete() {
    if (!isDeleting) setIsDeleteOpen(false);
  }

  return {
    isRenaming,
    draft,
    error,
    isDeleteOpen,
    isDeleting,
    chooseRename: () => setChosen("rename"),
    chooseDelete: () => setChosen("delete"),
    handleMenuCloseAutoFocus,
    handleChange,
    handleKeyDown,
    cancelRename: stopRename,
    confirmDelete,
    cancelDelete,
  };
}
