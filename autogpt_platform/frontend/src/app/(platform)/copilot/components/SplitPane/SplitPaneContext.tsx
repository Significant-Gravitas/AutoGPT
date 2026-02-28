"use client";

import { createContext, ReactNode, useContext } from "react";
import type { SplitDirection } from "./types";
import { usePaneTree } from "./usePaneTree";

interface SplitPaneContextValue {
  splitPane: (paneId: string, direction: SplitDirection) => void;
  closePane: (paneId: string) => void;
  setPaneSession: (paneId: string, sessionId: string | null) => void;
  focusedPaneId: string;
  setFocusedPaneId: (id: string) => void;
  leafCount: number;
}

const SplitPaneContext = createContext<SplitPaneContextValue | null>(null);

export function useSplitPaneContext() {
  const ctx = useContext(SplitPaneContext);
  if (!ctx) {
    throw new Error(
      "useSplitPaneContext must be used within a SplitPaneProvider",
    );
  }
  return ctx;
}

interface Props {
  children: (tree: ReturnType<typeof usePaneTree>["tree"]) => ReactNode;
}

export function SplitPaneProvider({ children }: Props) {
  const {
    tree,
    focusedPaneId,
    setFocusedPaneId,
    splitPane,
    closePane,
    setPaneSession,
    leafCount,
  } = usePaneTree();

  return (
    <SplitPaneContext.Provider
      value={{
        splitPane,
        closePane,
        setPaneSession,
        focusedPaneId,
        setFocusedPaneId,
        leafCount,
      }}
    >
      {children(tree)}
    </SplitPaneContext.Provider>
  );
}
