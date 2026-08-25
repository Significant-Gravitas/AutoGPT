"use client";

import { useAutoOpenForFiles } from "./useAutoOpenForFiles";
import { useCollapseContextPanelOnSession } from "./useCollapseContextPanelOnSession";

interface Props {
  sessionId: string | null;
  /** Auto-opening is desktop-only — on mobile the panel is a full-screen
   *  sheet, so opening it on session entry would bury the chat. The
   *  session-entry collapse below must still run on mobile: it is what
   *  forgets the previous chat's artifact, and the chat column mounts its
   *  artifacts button on every viewport. */
  canAutoOpen?: boolean;
}

export function ContextPanelAutoOpen({ sessionId, canAutoOpen = true }: Props) {
  useCollapseContextPanelOnSession(sessionId);
  useAutoOpenForFiles(sessionId, canAutoOpen);
  return null;
}
