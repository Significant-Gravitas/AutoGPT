"use client";

import { useStableCallback } from "@/hooks/useStableCallback";
import { useMemo } from "react";
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
  // The copilot page hands a new `onBackendTurn` every render (it only reads
  // refs), which would change the actions on every streamed flush. Cards get
  // one function that calls the latest; absent stays absent.
  const followBackendTurn = useStableCallback(() => onBackendTurn?.());
  const hasBackendTurn = Boolean(onBackendTurn);
  const actions = useMemo(
    () => ({
      onSend,
      onBackendTurn: hasBackendTurn ? followBackendTurn : undefined,
      chatSurface,
      getExecutionShareToken,
      sessionId,
      assistantName,
    }),
    [
      onSend,
      hasBackendTurn,
      followBackendTurn,
      chatSurface,
      getExecutionShareToken,
      sessionId,
      assistantName,
    ],
  );
  return (
    <CopilotChatActionsContext.Provider value={actions}>
      {children}
    </CopilotChatActionsContext.Provider>
  );
}
