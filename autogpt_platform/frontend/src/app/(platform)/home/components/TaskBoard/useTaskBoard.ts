import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { Key, storage } from "@/services/storage/local-storage";
import { useState } from "react";
import {
  type BoardState,
  groupSessionsIntoColumns,
  isDefaultColumn,
  makeColumnId,
  parseBoardState,
} from "./helpers";

export function useTaskBoard() {
  const { sessions, isLoading } = useSessionList();
  const { expertsById } = useExpertMap();
  const [board, setBoard] = useState(() =>
    parseBoardState(storage.get(Key.EXPERT_TASK_BOARD)),
  );

  function updateBoard(update: (prev: BoardState) => BoardState) {
    setBoard((prev) => {
      const next = update(prev);
      storage.set(Key.EXPERT_TASK_BOARD, JSON.stringify(next));
      return next;
    });
  }

  function moveCard(sessionId: string, columnId: string) {
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) return;
    updateBoard((prev) => ({
      ...prev,
      placements: {
        ...prev.placements,
        [sessionId]: { columnId, updatedAt: session.updated_at },
      },
    }));
  }

  function addColumn(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    updateBoard((prev) => ({
      ...prev,
      columns: [...prev.columns, { id: makeColumnId(trimmed), name: trimmed }],
    }));
  }

  function removeColumn(columnId: string) {
    if (isDefaultColumn(columnId)) return;
    updateBoard((prev) => ({
      ...prev,
      columns: prev.columns.filter((column) => column.id !== columnId),
    }));
  }

  return {
    isLoading,
    columns: groupSessionsIntoColumns(sessions, board),
    allColumns: board.columns,
    expertsById,
    moveCard,
    addColumn,
    removeColumn,
  };
}
