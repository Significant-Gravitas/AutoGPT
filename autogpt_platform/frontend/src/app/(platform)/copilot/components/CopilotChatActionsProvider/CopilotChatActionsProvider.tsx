"use client";

import {
  CopilotChatActionsContext,
  type CopilotChatSurface,
} from "./useCopilotChatActions";

interface Props {
  onSend: (message: string) => void | Promise<void>;
  /** Defaults to "copilot" — the standalone page. */
  chatSurface?: CopilotChatSurface;
  children: React.ReactNode;
}

export function CopilotChatActionsProvider({
  onSend,
  chatSurface = "copilot",
  children,
}: Props) {
  return (
    <CopilotChatActionsContext.Provider value={{ onSend, chatSurface }}>
      {children}
    </CopilotChatActionsContext.Provider>
  );
}
