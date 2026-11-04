import type { MouseEvent } from "react";

// Streamdown's href for a link whose markdown is still streaming in.
export const INCOMPLETE_LINK_HREF = "streamdown:incomplete-link";

const LEFT_BUTTON = 0;
const MIDDLE_BUTTON = 1;

export function isSameOriginLink(url: string) {
  try {
    return (
      new URL(url, window.location.origin).origin === window.location.origin
    );
  } catch {
    return false;
  }
}

// Every way of opening a link to another site asks first (SECRT-2096): a left
// click, with or without Ctrl, Cmd, Shift or Alt, and a middle-click. The right
// button only opens the context menu, which offers no "Open in new tab" for a
// button. On macOS a Ctrl-click is a right-click, so it opens that menu too.
export function asksBeforeOpening(event: MouseEvent) {
  return event.button === LEFT_BUTTON || event.button === MIDDLE_BUTTON;
}

// A middle press starts the browser's autoscroll on some systems, which can
// swallow the auxclick that asks.
export function isMiddlePress(event: MouseEvent) {
  return event.button === MIDDLE_BUTTON;
}
