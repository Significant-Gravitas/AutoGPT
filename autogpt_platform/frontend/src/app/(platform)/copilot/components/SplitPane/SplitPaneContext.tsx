"use client";

import { createContext, ReactNode, useContext } from "react";
import type { PaneNode, SplitDirection } from "./types";
import { usePaneTree } from "./usePaneTree";

interface SplitPaneContextValue {
  splitPane: (
    paneId: string,
    direction: SplitDirection,
    sessionIdForNewPane?: string | null,
  ) => void;
  closePane: (paneId: string) => void;
  setPaneSession: (paneId: string, sessionId: string | null) => void;
  focusedPaneId: string;
  setFocusedPaneId: (id: string) => void;
  leafCount: number;
  tree: PaneNode;
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

/** Safe version that returns null when outside the provider (e.g. mobile). */
export function useSplitPaneContextOptional() {
  return useContext(SplitPaneContext);
}

interface Props {
  children: ReactNode;
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
        tree,
      }}
    >
      {children}
    </SplitPaneContext.Provider>
  );
}
