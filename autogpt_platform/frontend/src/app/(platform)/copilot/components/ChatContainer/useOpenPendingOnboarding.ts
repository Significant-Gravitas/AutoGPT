import { parseAsString, useQueryState } from "nuqs";
import { useEffect } from "react";
import { usePendingOnboardingSession } from "../ExpertOnboardingCard/usePendingOnboardingSession";

interface Args {
  expertId: string | null;
  sessionId: string | null;
  enabled: boolean;
}

/** An empty expert chat moves to the chat whose setup questions still wait,
 *  so the answers reach the expert that asked them. */
export function useOpenPendingOnboarding({
  expertId,
  sessionId,
  enabled,
}: Args) {
  const [, setSessionId] = useQueryState("sessionId", parseAsString);
  const pending = usePendingOnboardingSession(expertId, enabled && !!sessionId);
  useEffect(() => {
    if (!pending.sessionId || pending.sessionId === sessionId) return;
    void setSessionId(pending.sessionId);
  }, [pending.sessionId, sessionId, setSessionId]);
}
