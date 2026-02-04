"use client";

import { createContext, useContext } from "react";

interface CopilotChatActions {
  onSend: (message: string) => void | Promise<void>;
}

const CopilotChatActionsContext = createContext<CopilotChatActions | null>(
  null,
);

export function useCopilotChatActions(): CopilotChatActions {
  const ctx = useContext(CopilotChatActionsContext);
  if (!ctx) {
    throw new Error(
      "useCopilotChatActions must be used within CopilotChatActionsProvider",
    );
  }
  return ctx;
}

export { CopilotChatActionsContext };
