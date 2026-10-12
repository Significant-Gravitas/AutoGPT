import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import type { BoardResponse } from "@/app/api/__generated__/models/boardResponse";
import type { GetV2GetBoardParams } from "@/app/api/__generated__/models/getV2GetBoardParams";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const BOARD_QUERY_KEY = ["copilot", "board"] as const;
export const BOARD_REFETCH_INTERVAL_MS = 10_000;
export const BOARD_MUTATION_KEY = ["copilot", "board", "mutation"];
export const BOARD_MOVE_MUTATION_KEY = [...BOARD_MUTATION_KEY, "move"];
export const BOARD_COLUMNS_MUTATION_KEY = [...BOARD_MUTATION_KEY, "columns"];
export const BOARD_CREATE_MUTATION_KEY = [...BOARD_MUTATION_KEY, "create"];
export const MAX_COLUMNS = 6;
export const MAX_COLUMN_NAME = 40;
export const MAX_BOARD_NAME = 40;
export const DEFAULT_BOARD_REF = "default";
export const BUILTIN_KEYS = ["backlog", "in_progress", "done"] as const;

const SESSION_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const RECENT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const POSITION_GAP = 1024;
const DEFAULT_RETRIES = 3;
// Parameters only the chat page acts on: a link carrying one must not land
// on the board.
const CHAT_ONLY_PARAMS = ["expertId", "kickoff", "new", "autosubmit", "source"];

export type Card = SessionSummaryResponse;

export interface ColumnWithCards {
  column: BoardColumn;
  cards: Card[];
}

export function isSessionId(value: string | null): value is string {
  return value !== null && SESSION_ID_PATTERN.test(value);
}

export function isBoardRef(value: string): boolean {
  return value === DEFAULT_BOARD_REF || isSessionId(value);
}

export function boardScopeKey(boardRef: string) {
  return [...BOARD_QUERY_KEY, boardRef] as const;
}

export function boardQueryKey(boardRef: string, params: GetV2GetBoardParams) {
  return [...boardScopeKey(boardRef), params] as const;
}

export function boardHref(boardId: string | null, sessionId?: string): string {
  const path = boardId ? `/boards/${boardId}` : "/home";
  return sessionId ? `${path}?sessionId=${sessionId}` : path;
}

export function isNotFoundError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

// Retries as the app's query client would, but a missing board stays missing.
export function retryUnlessNotFound(defaultRetry: unknown) {
  const limit =
    defaultRetry === false
      ? 0
      : typeof defaultRetry === "number"
        ? defaultRetry
        : DEFAULT_RETRIES;
  return (failureCount: number, error: unknown) =>
    !isNotFoundError(error) && failureCount < limit;
}

function byPosition(a: Card, b: Card) {
  return (a.board_position ?? 0) - (b.board_position ?? 0);
}

function byNewest(a: Card, b: Card) {
  return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
}

export function cardColumnKey(
  card: Card,
  columnKeys: Set<string>,
  now = new Date(),
): string {
  if (card.board_column) {
    return columnKeys.has(card.board_column) ? card.board_column : "backlog";
  }
  const isRecent =
    now.getTime() - new Date(card.updated_at).getTime() <= RECENT_WINDOW_MS;
  return isRecent ? "in_progress" : "done";
}

export function groupCards(
  columns: BoardColumn[],
  sessions: Card[],
  now = new Date(),
): ColumnWithCards[] {
  const keys = new Set(columns.map((column) => column.key));
  const placed = new Map<string, Card[]>();
  const unplaced = new Map<string, Card[]>();

  for (const card of sessions) {
    const target = card.board_column ? placed : unplaced;
    const key = cardColumnKey(card, keys, now);
    const list = target.get(key);
    if (list) list.push(card);
    else target.set(key, [card]);
  }

  return columns.map((column) => ({
    column,
    cards: [
      ...(placed.get(column.key) ?? []).sort(byPosition),
      ...(unplaced.get(column.key) ?? []).sort(byNewest),
    ],
  }));
}

export function positionBetween(
  before: number | null,
  after: number | null,
): number {
  if (before === null && after === null) return 0;
  if (before === null) return (after as number) - POSITION_GAP;
  if (after === null) return before + POSITION_GAP;
  return (before + after) / 2;
}

export function dropPosition(
  cards: Card[],
  targetIndex: number,
  movingId: string,
): number {
  const others = cards.filter((card) => card.id !== movingId);
  // Placed cards come first, so a card dropped among the unplaced ones lands
  // right after the last placed card above it.
  const before = others
    .slice(0, targetIndex)
    .findLast((card) => card.board_position != null);
  return positionBetween(
    before?.board_position ?? null,
    others[targetIndex]?.board_position ?? null,
  );
}

export interface DropTarget {
  column: string;
  index: number;
}

// `below`: the dragged card's top is past the middle of the card it is over.
// Within its own column a card takes the index of the card it is over, as
// arrayMove does; from another column it goes after that card when below.
export function resolveDrop(
  grouped: ColumnWithCards[],
  activeId: string,
  overId: string,
  below = false,
): DropTarget | null {
  for (const { column, cards } of grouped) {
    if (column.key === overId) {
      return {
        column: column.key,
        index: cards.filter((card) => card.id !== activeId).length,
      };
    }
    const index = cards.findIndex((card) => card.id === overId);
    if (index === -1) continue;
    const isSameColumn = cards.some((card) => card.id === activeId);
    return {
      column: column.key,
      index: !isSameColumn && below ? index + 1 : index,
    };
  }
  return null;
}

export function planDrop(
  grouped: ColumnWithCards[],
  activeId: string,
  overId: string,
  below = false,
): { column: string; position: number } | null {
  const target = resolveDrop(grouped, activeId, overId, below);
  const source = grouped.find(({ cards }) =>
    cards.some((card) => card.id === activeId),
  );
  if (!target || !source) return null;
  const fromIndex = source.cards.findIndex((card) => card.id === activeId);
  if (source.column.key === target.column && fromIndex === target.index) {
    return null;
  }
  const cards =
    grouped.find(({ column }) => column.key === target.column)?.cards ?? [];
  return {
    column: target.column,
    position: dropPosition(cards, target.index, activeId),
  };
}

export function dragLabel(grouped: ColumnWithCards[], id: string): string {
  for (const { column, cards } of grouped) {
    if (column.key === id) return column.name;
    const card = cards.find((item) => item.id === id);
    if (card) return card.title || "Untitled task";
  }
  return "the board";
}

export function nextColumnName(columns: BoardColumn[]): string {
  const names = new Set(columns.map((column) => column.name));
  if (!names.has("New column")) return "New column";
  let n = 2;
  while (names.has(`New column ${n}`)) n += 1;
  return `New column ${n}`;
}

export function newColumnKey(): string {
  return `col_${crypto.randomUUID()}`;
}

export function patchCard(
  data: BoardResponse,
  cardId: string,
  patch: Partial<Card>,
): BoardResponse {
  return {
    ...data,
    sessions: data.sessions.map((card) =>
      card.id === cardId ? { ...card, ...patch } : card,
    ),
  };
}

export function moveCard(
  data: BoardResponse,
  cardId: string,
  column: string,
  position: number,
): BoardResponse {
  return patchCard(data, cardId, {
    board_column: column,
    board_position: position,
  });
}

export function moveCardToTop(
  data: BoardResponse,
  cardId: string,
  column: string,
): BoardResponse {
  const positions = data.sessions.flatMap((card) =>
    card.id !== cardId &&
    card.board_column === column &&
    card.board_position != null
      ? [card.board_position]
      : [],
  );
  const top = positions.length > 0 ? Math.min(...positions) : null;
  return moveCard(data, cardId, column, positionBetween(null, top));
}

export function applyReassign(
  data: BoardResponse,
  oldSession: Card,
  newSession: Card,
): BoardResponse {
  return {
    ...data,
    sessions: [
      newSession,
      ...data.sessions.flatMap((card) => {
        if (card.id === newSession.id) return [];
        if (card.id !== oldSession.id) return [card];
        return [{ ...oldSession, reassigned_to_session_id: newSession.id }];
      }),
    ],
  };
}

export type CardState =
  | { kind: "running" }
  | { kind: "queued" }
  | { kind: "needs_you" }
  | { kind: "not_started" }
  | { kind: "line"; text: string }
  | { kind: "empty" };

export function cardState(card: Card): CardState {
  if (card.is_processing) return { kind: "running" };
  if (card.chat_status === "queued") return { kind: "queued" };
  if (card.needs_you) return { kind: "needs_you" };
  if (!card.started) return { kind: "not_started" };
  if (card.last_line) return { kind: "line", text: card.last_line };
  return { kind: "empty" };
}

export function stateLineText(state: CardState): string | null {
  switch (state.kind) {
    case "running":
      return "Working…";
    case "queued":
      return "Queued";
    case "needs_you":
      return "Waiting for your answer";
    case "not_started":
      return "Not started";
    case "line":
      return state.text;
    case "empty":
      return null;
  }
}

export function parseMention(text: string): {
  text: string;
  mentionOpen: boolean;
} {
  if (!/(^|\s)@$/.test(text)) return { text, mentionOpen: false };
  return { text: text.slice(0, -1), mentionOpen: true };
}

// A paywall (402) or a usage limit (429) says what to do next, so show it.
export function refusalReason(error: unknown): string | undefined {
  if (!(error instanceof ApiError)) return undefined;
  if (error.status !== 402 && error.status !== 429) return undefined;
  return error.message || undefined;
}

export function chatLinkTarget(search: string, hash: string): string | null {
  const params = new URLSearchParams(search);
  const hasChatParam = CHAT_ONLY_PARAMS.some((name) => params.has(name));
  const hasPrompt = new URLSearchParams(hash.replace(/^#/, "")).has("prompt");
  return hasChatParam || hasPrompt ? `/chat${search}${hash}` : null;
}
