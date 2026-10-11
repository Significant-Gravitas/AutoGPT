"use client";

import {
  CopilotChatActionsContext,
  type CopilotChatSurface,
} from "./useCopilotChatActions";

interface Props {
  onSend: (message: string) => void | Promise<void>;
  onBackendTurn?: () => void;
  /** Defaults to "copilot" — the standalone page. */
  chatSurface?: CopilotChatSurface;
  getExecutionShareToken?: (executionId: string) => string | null | undefined;
  sessionId?: string | null;
  assistantName?: string;
  children: React.ReactNode;
}

export function CopilotChatActionsProvider({
  onSend,
  onBackendTurn,
  chatSurface = "copilot",
  getExecutionShareToken,
  sessionId,
  assistantName,
  children,
}: Props) {
  return (
    <CopilotChatActionsContext.Provider
      value={{
        onSend,
        onBackendTurn,
        chatSurface,
        getExecutionShareToken,
        sessionId,
        assistantName,
      }}
    >
      {children}
    </CopilotChatActionsContext.Provider>
  );
}
