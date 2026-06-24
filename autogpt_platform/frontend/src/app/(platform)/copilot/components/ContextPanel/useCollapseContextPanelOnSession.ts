"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { useCopilotUIStore } from "../../store";

/**
 * Collapses the context panel on session entry. ContextPanelAutoOpen is keyed
 * by sessionId in CopilotPage, so this runs on every session change. The
 * useAutoOpenFor{Files,Progress} hooks then reopen the panel on the same
 * render when the new session has files or an active task list — sessions
 * with nothing to surface stay collapsed.
 *
 * Skips when sessionId is null so the very first mount (no session yet)
 * doesn't overwrite the user's persisted open/closed preference.
 */
export function useCollapseContextPanelOnSession(sessionId: string | null) {
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  useMountEffect(() => {
    if (!sessionId) return;
    closeArtifactPanel();
  });
}
