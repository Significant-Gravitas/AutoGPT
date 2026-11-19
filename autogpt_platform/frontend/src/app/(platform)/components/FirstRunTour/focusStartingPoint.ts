import { isTouchDevice } from "@/lib/platform";
import { TOUR_STARTING_POINT_ATTRIBUTE } from "@/lib/tour-targets";
import { findTourTarget } from "./helpers";

export function focusStartingPoint() {
  focusAfterTour(() => firstTask() ?? composer());
}

// The composer mounted under the walkthrough's focus trap, so nothing else
// would give it focus afterwards.
export function focusComposer() {
  focusAfterTour(composer);
}

// Two frames, so the overlay's focus trap has handed focus back first.
function focusAfterTour(find: () => HTMLElement | null) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const destination = find();
      if (!destination) return;
      destination.scrollIntoView({
        block: "center",
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });
      destination.focus({ preventScroll: true });
    });
  });
}

// A task's own button, never a "Hide" beside it; a busy row has none enabled.
function firstTask() {
  return (
    findTourTarget("first-tasks")?.querySelector<HTMLElement>(
      `[${TOUR_STARTING_POINT_ATTRIBUTE}]:not([disabled])`,
    ) ?? null
  );
}

// Not on a touch screen, where focus opens the keyboard.
function composer() {
  if (isTouchDevice()) return null;
  return (
    findTourTarget("composer")?.querySelector<HTMLElement>(
      "textarea:not([disabled])",
    ) ?? null
  );
}

function prefersReducedMotion() {
  return (
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false
  );
}
