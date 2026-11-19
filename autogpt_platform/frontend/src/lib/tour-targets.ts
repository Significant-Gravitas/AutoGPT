// The real UI the first-run walkthrough and its tips point at. Each value is set as
// `data-tour-target` on the element it names, so a step never depends on a
// class name or a DOM path that a redesign could move.
export type TourTarget =
  | "chats"
  | "team"
  | "files"
  | "workspace"
  | "wallet"
  | "account"
  | "first-tasks"
  | "composer"
  | "menu";

export const TOUR_TARGET_ATTRIBUTE = "data-tour-target";

export function tourTarget(target: TourTarget) {
  return { [TOUR_TARGET_ATTRIBUTE]: target };
}

// What "Start here" focuses inside a target: a task's own button, not the
// controls beside it (a row's "Hide" button).
export const TOUR_STARTING_POINT_ATTRIBUTE = "data-tour-starting-point";

export function tourStartingPoint() {
  return { [TOUR_STARTING_POINT_ATTRIBUTE]: "" };
}
