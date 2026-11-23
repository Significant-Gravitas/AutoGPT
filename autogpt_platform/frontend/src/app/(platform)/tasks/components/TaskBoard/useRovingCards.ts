import { isKey } from "@/lib/keyboard";
import { useState, type KeyboardEvent } from "react";
import { taskRowID, type BoardColumn } from "../../helpers";
import { MOVE_KEYS, type MoveFn } from "./boardKeys";

interface Args {
  columns: BoardColumn[];
  move: MoveFn;
  /** The card Tab lands on first: the open task's, when it is here. */
  preferredID: string | null;
}

/** One card in the tab order at a time; the arrow keys move between them
 *  and Enter opens the one in focus. */
export function useRovingCards({ columns, move, preferredID }: Args) {
  const [activeID, setActiveID] = useState<string | null>(null);
  const ids = columns.flatMap((column) => column.tasks.map((task) => task.id));
  const tabStop =
    [activeID, preferredID].find((id) => id !== null && ids.includes(id)) ??
    ids[0] ??
    null;

  function onKeyDown(e: KeyboardEvent<HTMLElement>, id: string) {
    const key = MOVE_KEYS.find((name) => isKey(e, name));
    if (!key) return;
    e.preventDefault();
    const next = move(columns, id, key);
    if (!next) return;
    setActiveID(next);
    document.getElementById(taskRowID(next))?.focus();
  }

  return { tabStop, focusCard: setActiveID, onKeyDown };
}

export type RovingCards = ReturnType<typeof useRovingCards>;

/** What one card needs to take part. */
export interface CardNav {
  position: string;
  isTabStop: boolean;
  onFocus: () => void;
  onKeyDown: (e: KeyboardEvent<HTMLElement>) => void;
}

export function cardNav(
  roving: RovingCards,
  id: string,
  position: string,
): CardNav {
  return {
    position,
    isTabStop: roving.tabStop === id,
    onFocus: () => roving.focusCard(id),
    onKeyDown: (e) => roving.onKeyDown(e, id),
  };
}
