"use client";

import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "../../store";

interface UseAutoOpenArtifactsOptions {
  sessionId: string | null;
}

export function useAutoOpenArtifacts({
  sessionId,
}: UseAutoOpenArtifactsOptions) {
  const resetArtifactPanel = useCopilotUIStore(
    (state) => state.resetArtifactPanel,
  );
  const prevSessionIdRef = useRef(sessionId);

  useEffect(() => {
    const isSessionChange = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;

    // Artifact previews should open only from an explicit user click.
    // When the session changes, fully clear the panel state so stale
    // active artifacts and back-stack entries never bleed into the next chat.
    if (isSessionChange) {
      resetArtifactPanel();
    }
  }, [sessionId, resetArtifactPanel]);

  // Reset on unmount so navigating away from /copilot (to /profile, /home,
  // etc.) can't leave the panel open in the Zustand store, which would then
  // render the panel re-open when the user returns. See SECRT-2254/2220.
  useEffect(() => {
    return () => {
      resetArtifactPanel();
    };
  }, [resetArtifactPanel]);
}
