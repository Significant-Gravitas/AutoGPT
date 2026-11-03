"use client";

import { useCopilotStreamStore } from "../../copilotStreamStore";
import { useCopilotTenantScope } from "../../CopilotTenantScopeContext";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const { organizationId, teamId } = useCopilotTenantScope();
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  // The thread chip reads this on every render; the compiler caches the scan
  // over every message part, so its identity stays stable for consumers. The
  // scope is rebuilt from its ids so a new context object does not rescan.
  return getSessionActivity(messages ?? [], { organizationId, teamId });
}
