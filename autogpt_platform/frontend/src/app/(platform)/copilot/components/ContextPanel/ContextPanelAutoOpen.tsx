"use client";

import { useAutoOpenForFiles } from "./useAutoOpenForFiles";

interface Props {
  sessionId: string | null;
}

export function ContextPanelAutoOpen({ sessionId }: Props) {
  useAutoOpenForFiles(sessionId);
  return null;
}
