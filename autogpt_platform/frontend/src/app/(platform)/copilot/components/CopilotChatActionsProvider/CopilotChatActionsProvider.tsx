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
  children: React.ReactNode;
}

export function CopilotChatActionsProvider({
  onSend,
  onBackendTurn,
  chatSurface = "copilot",
  getExecutionShareToken,
  children,
}: Props) {
  return (
    <CopilotChatActionsContext.Provider
      value={{ onSend, onBackendTurn, chatSurface, getExecutionShareToken }}
    >
      {children}
    </CopilotChatActionsContext.Provider>
  );
}
