"use client";

import { useAutoOpenForFiles } from "./useAutoOpenForFiles";
import { useAutoOpenForProgress } from "./useAutoOpenForProgress";
import { useCollapseContextPanelOnSession } from "./useCollapseContextPanelOnSession";

interface Props {
  sessionId: string | null;
}

export function ContextPanelAutoOpen({ sessionId }: Props) {
  useCollapseContextPanelOnSession(sessionId);
  useAutoOpenForFiles(sessionId);
  useAutoOpenForProgress(sessionId);
  return null;
}
