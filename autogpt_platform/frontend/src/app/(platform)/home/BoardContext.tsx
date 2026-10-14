"use client";

import { createContext, type ReactNode, useContext } from "react";

export interface BoardContextValue {
  boardRef: string;
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
