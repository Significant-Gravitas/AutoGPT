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
  const { uploaded, generated } = useSessionFiles(sessionId);
  const hasFiles = uploaded.length + generated.length > 0;
  const triggered = useRef(false);
  const prevSessionIdRef = useRef(sessionId);

  // Session change: clear the per-session "already triggered" guard so the
  // next session can auto-open independently. The shared `_autoOpenReady` /
  // `_autoOpenUserClosed` flags are reset by `useAutoOpenArtifacts` (the
  // single owner of that global state); calling `resetAutoOpenState()` here
  // too races with that hook and can clobber the ready flag depending on
  // mount order.
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      triggered.current = false;
      prevSessionIdRef.current = sessionId;
    }
  }, [sessionId]);

  useEffect(() => {
    if (hasFiles && !triggered.current) {
      triggered.current = true;
      openContextPanelForFiles();
    }
  }, [hasFiles, openContextPanelForFiles]);
}
