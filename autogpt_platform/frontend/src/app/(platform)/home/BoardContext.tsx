"use client";

import { createContext, type ReactNode, useContext } from "react";

export interface BoardContextValue {
  /** The id used in API paths and cache keys: "default" or a board uuid. */
  boardRef: string;
  /** The board's real id, once it has loaded. */
  boardId: string | null;
  isDefault: boolean;
}

const BoardContext = createContext<BoardContextValue | null>(null);

interface Props {
  value: BoardContextValue;
  children: ReactNode;
}

export function BoardProvider({ value, children }: Props) {
  return (
    <BoardContext.Provider value={value}>{children}</BoardContext.Provider>
  );
}

export function useBoardRef() {
  const value = useContext(BoardContext);
  if (!value) throw new Error("useBoardRef must be used inside BoardProvider");
  return value;
}
