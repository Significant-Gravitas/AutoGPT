"use client";

import { useMemo } from "react";
import { useCopilotStreamStore } from "../../copilotStreamStore";
import { getSessionActivity } from "./helpers";

export function useSessionActivity(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId ? s.messageSnapshots[sessionId] : undefined,
  );
  // The thread chip reads this on every render, and `messageSnapshots` is a
  // fresh array per streamed token — so the scan over every message part has
  // to be memoised, and the identity kept stable for consumers.
  return useMemo(() => getSessionActivity(messages ?? []), [messages]);
}
