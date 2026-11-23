import type { BoardColumn } from "../../helpers";

export const MOVE_KEYS = [
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Home",
  "End",
] as const;
export type MoveKey = (typeof MOVE_KEYS)[number];

export type MoveFn = (
  columns: BoardColumn[],
  id: string,
  key: MoveKey,
) => string | null;

function locate(columns: BoardColumn[], id: string) {
  for (const [lane, column] of columns.entries()) {
    const index = column.tasks.findIndex((task) => task.id === id);
    if (index !== -1) return { lane, index };
  }
  return null;
}

/** The nearest lane with cards, left or right of `lane`, and the card
 *  level with `index` in it (or its last). */
function sideways(
  columns: BoardColumn[],
  lane: number,
  index: number,
  step: number,
) {
  for (let at = lane + step; at >= 0 && at < columns.length; at += step) {
    const tasks = columns[at].tasks;
    if (tasks.length > 0) return tasks[Math.min(index, tasks.length - 1)].id;
  }
  return null;
}

/** Where a key moves focus on the board: up and down a lane, Home and End
 *  to its ends, left and right to the next lane with cards. */
export const moveOnBoard: MoveFn = (columns, id, key) => {
  const at = locate(columns, id);
  if (!at) return null;
  const tasks = columns[at.lane].tasks;
  switch (key) {
    case "ArrowUp":
      return tasks[at.index - 1]?.id ?? null;
    case "ArrowDown":
      return tasks[at.index + 1]?.id ?? null;
    case "Home":
      return tasks[0].id;
    case "End":
      return tasks[tasks.length - 1].id;
    case "ArrowLeft":
      return sideways(columns, at.lane, at.index, -1);
    case "ArrowRight":
      return sideways(columns, at.lane, at.index, 1);
  }
};

/** Where a key moves focus in the list: up and down across the groups,
 *  Home and End to the first and last task. */
export const moveInList: MoveFn = (columns, id, key) => {
  const ids = columns.flatMap((column) => column.tasks.map((task) => task.id));
  const index = ids.indexOf(id);
  if (index === -1) return null;
  switch (key) {
    case "ArrowUp":
      return ids[index - 1] ?? null;
    case "ArrowDown":
      return ids[index + 1] ?? null;
    case "Home":
      return ids[0];
    case "End":
      return ids[ids.length - 1];
    default:
      return null;
  }
};

/** What a card says of where it is: "Done, 1 of 3". */
export function positionLabel(column: BoardColumn, index: number) {
  return `${column.label}, ${index + 1} of ${column.tasks.length}`;
}
