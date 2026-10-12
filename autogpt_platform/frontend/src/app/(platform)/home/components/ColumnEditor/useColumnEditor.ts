import { usePutV2UpdateBoardColumns } from "@/app/api/__generated__/endpoints/chat/chat";
import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import {
  BOARD_COLUMNS_MUTATION_KEY,
  MAX_COLUMNS,
  newColumnKey,
  nextColumnName,
} from "../../board-helpers";
import { useBoardCache } from "../../useBoardCache";

export function useColumnEditor(columns: BoardColumn[]) {
  const { applyOptimistic, rollback, settleBoard } = useBoardCache();
  const [renamingKey, setRenamingKey] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<BoardColumn | null>(null);
  // The column whose actions button takes focus once it is on screen again.
  const [focusKey, setFocusKey] = useState<string | null>(null);

  const { mutate: saveColumns } = usePutV2UpdateBoardColumns({
    mutation: {
      mutationKey: BOARD_COLUMNS_MUTATION_KEY,
      onMutate: ({ data }) =>
        applyOptimistic((board) => ({ ...board, columns: data.columns })),
      onError: (_error, _variables, snapshot) => {
        rollback(snapshot);
        toast({ title: "Couldn't save the columns", variant: "destructive" });
      },
      onSettled: () => settleBoard(),
    },
  });

  function save(next: BoardColumn[]) {
    saveColumns({ data: { columns: next } });
  }

  function finishRename(key: string) {
    setRenamingKey(null);
    setFocusKey(key);
  }

  function rename(key: string, name: string) {
    finishRename(key);
    const current = columns.find((column) => column.key === key);
    if (!current || current.name === name) return;
    save(
      columns.map((column) =>
        column.key === key ? { ...column, name } : column,
      ),
    );
  }

  function addColumnAfter(key: string) {
    if (columns.length >= MAX_COLUMNS) return;
    const index = columns.findIndex((column) => column.key === key);
    const column: BoardColumn = {
      key: newColumnKey(),
      name: nextColumnName(columns),
      builtin: false,
    };
    save([...columns.slice(0, index + 1), column, ...columns.slice(index + 1)]);
    setRenamingKey(column.key);
  }

  function confirmDelete() {
    if (!pendingDelete) return;
    save(columns.filter((column) => column.key !== pendingDelete.key));
    setPendingDelete(null);
    setFocusKey("backlog");
  }

  function cancelDelete() {
    if (pendingDelete) setFocusKey(pendingDelete.key);
    setPendingDelete(null);
  }

  function menuRef(key: string) {
    return (element: HTMLElement | null) => {
      if (!element || focusKey !== key) return;
      element.querySelector("button")?.focus();
      setFocusKey(null);
    };
  }

  return {
    renamingKey,
    canAddColumn: columns.length < MAX_COLUMNS,
    startRename: setRenamingKey,
    cancelRename: finishRename,
    dismissRename: () => setRenamingKey(null),
    rename,
    addColumnAfter,
    addColumn: () => addColumnAfter(columns[columns.length - 1]?.key ?? ""),
    pendingDelete,
    askDelete: setPendingDelete,
    cancelDelete,
    confirmDelete,
    menuRef,
  };
}
