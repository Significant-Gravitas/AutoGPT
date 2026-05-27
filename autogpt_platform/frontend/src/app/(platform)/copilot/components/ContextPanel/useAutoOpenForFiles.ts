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

  useEffect(() => {
    if (hasFiles && !triggered.current) {
      triggered.current = true;
      openContextPanelForFiles();
    }
  }, [hasFiles, openContextPanelForFiles]);
}
