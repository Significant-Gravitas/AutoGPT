"use client";

import { useEffect } from "react";
import { useCopilotUIStore } from "../../store";

/**
 * Collapses the context panel on session entry. ContextPanelAutoOpen is keyed
 * by sessionId in CopilotPage, so this runs on every session change. The
 * useAutoOpenFor{Files,Progress} hooks then reopen the panel on the same
 * render when the new session has files or an active task list — sessions
 * with nothing to surface stay collapsed.
 */
export function useCollapseContextPanelOnSession() {
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  useEffect(() => {
    closeArtifactPanel();
  }, [closeArtifactPanel]);
}
