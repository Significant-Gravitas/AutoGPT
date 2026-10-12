import type { BoardSummary } from "@/app/api/__generated__/models/boardSummary";
import { DEFAULT_BOARD_REF } from "../../board-helpers";

export interface MoveTarget {
  ref: string;
  boardId: string | null;
  name: string;
  isHome: boolean;
}

const HOME_TARGET: MoveTarget = {
  ref: DEFAULT_BOARD_REF,
  boardId: null,
  name: "Home",
  isHome: true,
};

// Home always exists, so a project offers it before the list has loaded.
export function moveTargets(
  boards: BoardSummary[],
  currentRef: string,
  isCurrentDefault: boolean,
): MoveTarget[] {
  const projects = boards.flatMap((board) =>
    board.is_default || board.id === currentRef
      ? []
      : [{ ref: board.id, boardId: board.id, name: board.name, isHome: false }],
  );
  return isCurrentDefault ? projects : [HOME_TARGET, ...projects];
}
