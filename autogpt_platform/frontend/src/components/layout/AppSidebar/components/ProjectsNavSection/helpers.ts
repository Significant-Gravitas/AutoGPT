import { boardHref } from "@/app/(platform)/home/board-helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const MAX_BOARDS = 20;
export const PROJECT_LIMIT_MESSAGE = `You can have up to ${MAX_BOARDS} projects`;

const UNTITLED_PROJECT = "Untitled project";

export function nextProjectName(names: string[]): string {
  const taken = new Set(names);
  if (!taken.has(UNTITLED_PROJECT)) return UNTITLED_PROJECT;
  let n = 2;
  while (taken.has(`${UNTITLED_PROJECT} ${n}`)) n += 1;
  return `${UNTITLED_PROJECT} ${n}`;
}

export function newProjectHref(boardId: string): string {
  return `${boardHref(boardId)}?rename=1`;
}

export function isBoardPath(pathname: string | null, boardId?: string) {
  if (!pathname) return false;
  const path = boardId ? boardHref(boardId) : "/boards";
  return pathname === path || pathname.startsWith(`${path}/`);
}

export function errorDetail(error: unknown): string | undefined {
  return error instanceof ApiError ? error.message || undefined : undefined;
}
