import type { MouseEvent } from "react";

// A modified click opens the link somewhere else and leaves this page as it was.
export function isPlainLeftClick(event: MouseEvent) {
  return (
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey
  );
}
