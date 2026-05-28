"use client";

import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "../../store";
import { useSessionFiles } from "./components/FilesTab/useSessionFiles";

/**
 * Opens the context panel to the Files tab the first time a session has files,
 * unless the user has explicitly closed it. Mirrors useAutoOpenArtifacts.
 */
export function useAutoOpenForFiles(sessionId: string | null) {
  const openContextPanelForFiles = useCopilotUIStore(
    (s) => s.openContextPanelForFiles,
  );
  const resetAutoOpenState = useCopilotUIStore((s) => s.resetAutoOpenState);
  const { uploaded, generated } = useSessionFiles(sessionId);
  const hasFiles = uploaded.length + generated.length > 0;
  const triggered = useRef(false);
  const prevSessionIdRef = useRef(sessionId);

  // Session change: reset the per-session "already triggered" guard AND the
  // global "user closed" flag so the next session can auto-open independently.
  // Without this, closing the panel once disables files auto-open for every
  // subsequent session in the same tab until reload.
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      triggered.current = false;
      resetAutoOpenState();
      prevSessionIdRef.current = sessionId;
    }
  }, [sessionId, resetAutoOpenState]);

  useEffect(() => {
    if (hasFiles && !triggered.current) {
      triggered.current = true;
      openContextPanelForFiles();
    }
  }, [hasFiles, openContextPanelForFiles]);
}
