"use client";

import { CopilotChatActionsContext } from "./useCopilotChatActions";

interface Props {
  onSend: (message: string) => void | Promise<void>;
  children: React.ReactNode;
}

export function CopilotChatActionsProvider({ onSend, children }: Props) {
  return (
    <CopilotChatActionsContext.Provider value={{ onSend }}>
      {children}
    </CopilotChatActionsContext.Provider>
  );
}
