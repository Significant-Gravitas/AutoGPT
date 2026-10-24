import { findTourTarget } from "./helpers";

// "Start here" lands the user on the first thing they can act on: the first
// suggested task if there is one, the composer otherwise. Deferred a frame so
// the walkthrough's overlay is gone and nothing hands focus back behind it.
export function focusStartingPoint() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const tasks = findTourTarget("first-tasks");
      const task = tasks?.querySelector<HTMLElement>("button:not([disabled])");
      const composer = findTourTarget("composer")?.querySelector<HTMLElement>(
        "textarea:not([disabled])",
      );
      const destination = task ?? composer;
      if (!destination) return;
      destination.scrollIntoView({ block: "center", behavior: "smooth" });
      destination.focus({ preventScroll: true });
    });
  });
}
