"use client";

import { useMemo } from "react";
import { useCopilotStreamStore } from "../../copilotStreamStore";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  // The thread chip reads this on every render, so the scan over every
  // message part is memoised and its identity kept stable for consumers.
  return useMemo(() => getSessionActivity(messages ?? []), [messages]);
}
