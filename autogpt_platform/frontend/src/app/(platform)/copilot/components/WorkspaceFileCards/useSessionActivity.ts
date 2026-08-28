"use client";

import { useCopilotStreamStore } from "../../copilotStreamStore";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  return getSessionActivity(messages ?? []);
}
