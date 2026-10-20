"use client";

import { useCollapseContextPanelOnSession } from "./useCollapseContextPanelOnSession";

interface Props {
  sessionId: string | null;
}

/** Session-entry reset: forgets the previous chat's artifact so it can't
 *  bleed into the next chat. The panel no longer auto-opens on entry — the
 *  top-right artifacts button carries the latest filename instead, so one
 *  click reaches the same file without hijacking the layout. */
export function ContextPanelAutoOpen({ sessionId }: Props) {
  useCollapseContextPanelOnSession(sessionId);
  return null;
}
