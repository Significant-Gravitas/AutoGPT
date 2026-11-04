import type { MouseEvent } from "react";

// Streamdown's href for a link whose markdown is still streaming in.
export const INCOMPLETE_LINK_HREF = "streamdown:incomplete-link";

type LinkClick = "plain" | "modified" | "middle" | "other";

// The clicks on a link to another site that ask first. Middle-clicks and
// Ctrl/Cmd/Shift/Alt-clicks open it at once, as the browser's "Open in new
// tab" menu does (SECRT-2096).
const ASKS_BEFORE_LEAVING: Record<LinkClick, boolean> = {
  plain: true,
  modified: false,
  middle: false,
  other: false,
};

function kindOfClick(event: MouseEvent): LinkClick {
  if (event.button === 1) return "middle";
  if (event.button !== 0) return "other";
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
    return "modified";
  return "plain";
}

export function isSameOriginLink(url: string) {
  try {
    return (
      new URL(url, window.location.origin).origin === window.location.origin
    );
  } catch {
    return false;
  }
}

export function asksBeforeOpening(event: MouseEvent, url: string) {
  return ASKS_BEFORE_LEAVING[kindOfClick(event)] && !isSameOriginLink(url);
}
