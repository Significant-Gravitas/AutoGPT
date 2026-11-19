import type { Task } from "@/app/api/__generated__/models/task";

/** The cards to show: each live suggestion not set aside, a started one in
 *  its place as the running task, and started ones the list has since
 *  dropped after them, so a card never jumps away once started. */
export function visibleCards(
  suggestions: Task[],
  started: Task[],
  hiddenIDs: string[],
) {
  const startedByID = new Map(started.map((task) => [task.id, task]));
  const live = suggestions.filter((task) => !hiddenIDs.includes(task.id));
  const liveIDs = new Set(live.map((task) => task.id));
  return [
    ...live.map((task) => startedByID.get(task.id) ?? task),
    ...started.filter((task) => !liveIDs.has(task.id)),
  ];
}
