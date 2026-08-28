"use client";

import { useCopilotStreamStore } from "../../copilotStreamStore";
import { useCopilotTenantScope } from "../../CopilotTenantScopeContext";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const scope = useCopilotTenantScope();
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  return getSessionActivity(messages ?? [], scope);
}
