import { act } from "@testing-library/react";

// What the browser fires when Back brings the page out of its back/forward
// cache: the same document, with every React state exactly as it was left.
// happy-dom's PageTransitionEvent is a plain Event, so `persisted` is set here.
export function restoreFromBackForwardCache() {
  const event = new Event("pageshow");
  Object.defineProperty(event, "persisted", { value: true });
  act(() => {
    window.dispatchEvent(event);
  });
}
