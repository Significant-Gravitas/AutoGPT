import {
  getGetV2GetSessionQueryKey,
  usePatchV2UpdateSessionBoard,
  usePatchV2UpdateSessionExpert,
  usePatchV2UpdateSessionTitle,
  useGetV2GetSession,
  usePostV2StartSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  BOARD_MOVE_MUTATION_KEY,
  BOARD_MUTATION_KEY,
  type Card,
  cardColumnKey,
  moveCardToTop,
  patchCard,
} from "../../board-helpers";
import { useBoardCache } from "../../useBoardCache";

interface Args {
  card: Card | null;
  sessionId: string;
  columns: BoardColumn[];
}

function isConflict(error: unknown) {
  return error instanceof ApiError && error.status === 409;
}

const EDITABLE_SELECTOR = "input, textarea, [contenteditable='true']";

export function useTaskPanel({ card, sessionId, columns }: Args) {
  const queryClient = useQueryClient();
  const { applyOptimistic, rollback, settleBoard } = useBoardCache();
  const [showFiles, setShowFiles] = useState(false);
  const [startedLocally, setStartedLocally] = useState(false);
  const [renamedTitle, setRenamedTitle] = useState<string | null>(null);

  const { data: sessionTitle } = useGetV2GetSession(sessionId, undefined, {
    query: {
      enabled: !card,
      select: (res) => (res.status === 200 ? res.data.title : null),
    },
  });

  const { mutate: moveTask } = usePatchV2UpdateSessionBoard({
    mutation: {
      mutationKey: BOARD_MOVE_MUTATION_KEY,
      onMutate: ({ data }) =>
        applyOptimistic((board) =>
          moveCardToTop(board, sessionId, data.column ?? ""),
        ),
      onError: (_error, _variables, snapshot) => {
        rollback(snapshot);
        toast({ title: "Couldn't move the task", variant: "destructive" });
      },
      onSettled: () => settleBoard(),
    },
  });

  const { mutate: assignTask } = usePatchV2UpdateSessionExpert({
    mutation: {
      mutationKey: [...BOARD_MUTATION_KEY, "assign"],
      onMutate: ({ data }) =>
        applyOptimistic((board) =>
          patchCard(board, sessionId, { expert_id: data.expert_id }),
        ),
      onError: (error, _variables, snapshot) => {
        rollback(snapshot);
        toast({
          title: isConflict(error)
            ? "This task already started. Use Reassign instead."
            : "Couldn't change who works on this task",
          variant: "destructive",
        });
      },
      onSettled: () => settleBoard(),
    },
  });

  const { mutate: renameTask } = usePatchV2UpdateSessionTitle({
    mutation: {
      mutationKey: [...BOARD_MUTATION_KEY, "rename"],
      onMutate: ({ data }) => setRenamedTitle(data.title),
      onError: () => {
        setRenamedTitle(null);
        toast({ title: "Couldn't rename the task", variant: "destructive" });
      },
      onSettled: () => settleBoard(),
    },
  });

  const { mutate: startTask, isPending: isStarting } = usePostV2StartSession({
    mutation: {
      mutationKey: [...BOARD_MUTATION_KEY, "start"],
      onSuccess: () => setStartedLocally(true),
      onError: (error) => {
        if (isConflict(error)) {
          setStartedLocally(true);
          return;
        }
        toast({ title: "Couldn't start the task", variant: "destructive" });
      },
      onSettled: () => {
        void settleBoard();
        void queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(sessionId),
        });
      },
    },
  });

  const columnKey = card
    ? cardColumnKey(card, new Set(columns.map((column) => column.key)))
    : null;

  function handleRename(title: string) {
    renameTask({ sessionId, data: { title } });
  }

  function handleAssign(expertId: string | null) {
    if (expertId === (card?.expert_id ?? null)) return;
    assignTask({ sessionId, data: { expert_id: expertId } });
  }

  function handleMove(column: string) {
    if (column === columnKey) return;
    moveTask({ sessionId, data: { column } });
  }

  function handleStart(message: string) {
    startTask({ sessionId, data: { message } });
  }

  // Radix closes the sheet from a document-level capture listener, so
  // Escape meant for an input (a mention menu, an IME, the title editor)
  // would close the whole panel.
  function handleEscapeKeyDown(event: KeyboardEvent) {
    const target = event.target;
    if (
      event.isComposing ||
      (target instanceof Element && target.closest(EDITABLE_SELECTOR))
    ) {
      event.preventDefault();
    }
  }

  return {
    title:
      renamedTitle ??
      (card ? card.title || "Untitled task" : sessionTitle || "Untitled chat"),
    columnKey,
    isDraft: !!card && !card.started && !startedLocally,
    isStarting,
    showFiles,
    toggleFiles: () => setShowFiles((value) => !value),
    handleRename,
    handleAssign,
    handleMove,
    handleStart,
    handleEscapeKeyDown,
  };
}
