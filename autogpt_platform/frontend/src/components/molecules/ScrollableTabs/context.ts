import * as React from "react";
import { createContext, useContext } from "react";

interface ScrollableTabsContextValue {
  activeValue: string | null;
  setActiveValue: React.Dispatch<React.SetStateAction<string | null>>;
  registerContent: (value: string, element: HTMLElement | null) => void;
  scrollToSection: (value: string) => void;
  scrollContainer: HTMLElement | null;
}

export const ScrollableTabsContext = createContext<
  ScrollableTabsContextValue | undefined
>(undefined);

export function useScrollableTabs() {
  const context = useContext(ScrollableTabsContext);
  if (!context) {
    throw new Error("useScrollableTabs must be used within a ScrollableTabs");
  }
  return context;
}
