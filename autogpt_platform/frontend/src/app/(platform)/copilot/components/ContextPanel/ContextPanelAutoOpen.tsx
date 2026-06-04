"use client";

import { useAutoOpenForFiles } from "./useAutoOpenForFiles";
import { useAutoOpenForProgress } from "./useAutoOpenForProgress";

interface Props {
  sessionId: string | null;
}

export function ContextPanelAutoOpen({ sessionId }: Props) {
  useAutoOpenForFiles(sessionId);
  useAutoOpenForProgress(sessionId);
  return null;
}
