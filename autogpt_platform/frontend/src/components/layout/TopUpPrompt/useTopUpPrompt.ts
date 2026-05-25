import { createContext, useContext } from "react";

interface TopUpPromptContextValue {
  isOutOfCredits: boolean;
  openTopUp: () => void;
}

export const TopUpPromptContext = createContext<TopUpPromptContextValue | null>(
  null,
);

export function useTopUpPrompt() {
  const context = useContext(TopUpPromptContext);
  if (!context) {
    throw new Error("useTopUpPrompt must be used within a TopUpPromptProvider");
  }
  return context;
}
