import { findTourTarget } from "./helpers";

// "Start here" lands the user on the first thing they can act on: the first
// suggested task if there is one, the composer otherwise.
export function focusStartingPoint() {
  focusAfterTour(() => firstTask() ?? composer());
}

// Skipping hands focus to the composer, as Home does on any other visit. It
// mounted under the walkthrough, whose focus trap kept it from taking focus,
// and nothing else would give it focus afterwards.
export function focusComposer() {
  focusAfterTour(composer);
}

// Deferred two frames so the walkthrough's overlay is gone and its focus
// trap has already handed focus back behind it.
function focusAfterTour(find: () => HTMLElement | null) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const destination = find();
      if (!destination) return;
      destination.scrollIntoView({ block: "center", behavior: "smooth" });
      destination.focus({ preventScroll: true });
    });
  });
}

function firstTask() {
  return (
    findTourTarget("first-tasks")?.querySelector<HTMLElement>(
      "button:not([disabled])",
    ) ?? null
  );
}

function composer() {
  return (
    findTourTarget("composer")?.querySelector<HTMLElement>(
      "textarea:not([disabled])",
    ) ?? null
  );
}
