"use client";

import { createContext, useContext } from "react";

/**
 * Which chat surface this message list is rendered in.
 *
 * `"copilot"` — the standalone `/copilot` page; tool cards should show
 *   navigation CTAs (Open in library, Open in builder, View Execution)
 *   so the user can jump to the referenced resource.
 * `"builder"` — the in-builder chat panel (`BuilderChatPanel`); the user
 *   is already looking at the builder and the panel auto-switches URL on
 *   edit_agent / run_agent completion, so the navigation CTAs are
 *   redundant and open duplicate tabs.
 */
export type CopilotChatSurface = "copilot" | "builder";

interface CopilotChatActions {
  onSend: (message: string) => void | Promise<void>;
  chatSurface: CopilotChatSurface;
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
