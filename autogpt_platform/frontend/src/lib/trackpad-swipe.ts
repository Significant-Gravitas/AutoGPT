const THRESHOLD = 60;
const SETTLE_MS = 320;

export type SwipeStep = 1 | -1;

/**
 * Turns horizontal trackpad scrolling over `element` into discrete page
 * steps. `onStep` fires once per gesture; `canPage` is asked before each
 * gesture so the caller can opt out while a view cannot turn. Returns the
 * cleanup.
 */
export function listenForSwipes(
  element: HTMLElement,
  onStep: (step: SwipeStep) => void,
  canPage: () => boolean = () => true,
) {
  let travelled = 0;
  let fired = false;
  let settle: ReturnType<typeof setTimeout> | undefined;

  const reset = () => {
    travelled = 0;
    fired = false;
  };

  const onWheel = (event: WheelEvent) => {
    if (Math.abs(event.deltaX) <= Math.abs(event.deltaY)) return;
    if (!canPage()) return;
    event.preventDefault();
    clearTimeout(settle);
    settle = setTimeout(reset, SETTLE_MS);
    if (fired) return;
    travelled += event.deltaX;
    if (Math.abs(travelled) < THRESHOLD) return;
    fired = true;
    onStep(travelled > 0 ? 1 : -1);
  };

  element.addEventListener("wheel", onWheel, { passive: false });
  return () => {
    clearTimeout(settle);
    element.removeEventListener("wheel", onWheel);
  };
}
