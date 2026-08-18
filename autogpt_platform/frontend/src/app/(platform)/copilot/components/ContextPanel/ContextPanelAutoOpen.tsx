"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useAutoOpenForFiles } from "./useAutoOpenForFiles";
import { useAutoOpenForProgress } from "./useAutoOpenForProgress";
import { useCollapseContextPanelOnSession } from "./useCollapseContextPanelOnSession";

interface Props {
  sessionId: string | null;
}

export function ContextPanelAutoOpen({ sessionId }: Props) {
  // The sidebar auto-opens on progress only when the task bar is off.
  const taskBarEnabled = useGetFlag(Flag.TASK_PROGRESS_BAR);
  useCollapseContextPanelOnSession(sessionId);
  useAutoOpenForFiles(sessionId);
  useAutoOpenForProgress(sessionId, !taskBarEnabled);
  return null;
}
