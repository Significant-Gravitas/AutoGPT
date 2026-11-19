import { Key, storage } from "@/services/storage/local-storage";
import { useState } from "react";

export type BoardView = "board" | "list";

function storedView(): BoardView {
  return storage.get(Key.HOME_TASKS_VIEW) === "list" ? "list" : "board";
}

/** Board or list, kept across reloads; the board until a list is picked. */
export function useBoardView() {
  const [view, setViewState] = useState<BoardView>(storedView);

  function setView(next: BoardView) {
    setViewState(next);
    storage.set(Key.HOME_TASKS_VIEW, next);
  }

  return { view, setView };
}
