"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useAutoOpenForFiles } from "./useAutoOpenForFiles";
import { useAutoOpenForProgress } from "./useAutoOpenForProgress";
import { useCollapseContextPanelOnSession } from "./useCollapseContextPanelOnSession";

interface Props {
  sessionId: string | null;
  /** Auto-opening is desktop-only — on mobile the panel is a full-screen
   *  sheet, so opening it on session entry would bury the chat. The
   *  session-entry collapse below must still run on mobile: it is what
   *  forgets the previous chat's artifact, and the new tool UI mounts its
   *  artifacts button on every viewport. */
  canAutoOpen?: boolean;
}

export function ContextPanelAutoOpen({ sessionId, canAutoOpen = true }: Props) {
  // The sidebar auto-opens on progress only when the task bar is off.
  const taskBarEnabled = useGetFlag(Flag.TASK_PROGRESS_BAR);
  useCollapseContextPanelOnSession(sessionId);
  useAutoOpenForFiles(sessionId, canAutoOpen);
  useAutoOpenForProgress(sessionId, canAutoOpen && !taskBarEnabled);
  return null;
}
