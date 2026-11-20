"use client";

import { useCopilotStreamStore } from "../../copilotStreamStore";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  // The thread chip reads this on every render; the compiler caches the scan
  // over every message part, so its identity stays stable for consumers.
  return getSessionActivity(messages ?? []);
}
