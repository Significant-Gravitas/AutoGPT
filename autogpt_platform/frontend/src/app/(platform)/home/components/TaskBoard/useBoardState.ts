import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Key } from "@/services/storage/local-storage";
import { createSafeStorage } from "@/services/storage/safe-storage";
import { useState } from "react";
import { type BoardState, parseBoardState } from "./helpers";

const boardStorage =
  createSafeStorage<`${Key.EXPERT_TASK_BOARD}:${string}`>("local");

function readBoard(userId: string | undefined) {
  return {
    userId,
    board: parseBoardState(
      userId ? boardStorage.get(`${Key.EXPERT_TASK_BOARD}:${userId}`) : null,
    ),
  };
}

export function useBoardState() {
  const { user } = useAuth();
  const userId = user?.id;
  const [saved, setSaved] = useState(() => readBoard(userId));
  const current = saved.userId === userId ? saved : readBoard(userId);
  if (current !== saved) setSaved(current);

  function updateBoard(update: (prev: BoardState) => BoardState) {
    const board = update(current.board);
    if (userId) {
      boardStorage.set(
        `${Key.EXPERT_TASK_BOARD}:${userId}`,
        JSON.stringify(board),
      );
    }
    setSaved({ userId, board });
  }

  return { board: current.board, updateBoard };
}
