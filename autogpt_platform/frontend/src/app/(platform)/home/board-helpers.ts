import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import type { BoardResponse } from "@/app/api/__generated__/models/boardResponse";
import type { GetV2GetBoardParams } from "@/app/api/__generated__/models/getV2GetBoardParams";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";

export const BOARD_QUERY_KEY = ["copilot", "board"] as const;
export const BOARD_REFETCH_INTERVAL_MS = 10_000;
export const BOARD_MOVE_MUTATION_KEY = ["patchV2UpdateSessionBoard"];
export const MAX_COLUMNS = 6;
export const BUILTIN_KEYS = ["backlog", "in_progress", "done"] as const;

const RECENT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const POSITION_GAP = 1024;

export type Card = SessionSummaryResponse;

export interface ColumnWithCards {
  column: BoardColumn;
  cards: Card[];
}

export function boardQueryKey(params: GetV2GetBoardParams) {
  return [...BOARD_QUERY_KEY, params] as const;
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
  return positionBetween(
    others[targetIndex - 1]?.board_position ?? null,
    others[targetIndex]?.board_position ?? null,
  );
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

export type CardState =
  | { kind: "running" }
  | { kind: "needs_you" }
  | { kind: "not_started" }
  | { kind: "line"; text: string }
  | { kind: "empty" };

export function cardState(card: Card): CardState {
  if (card.is_processing) return { kind: "running" };
  if (card.needs_you) return { kind: "needs_you" };
  if (!card.started) return { kind: "not_started" };
  if (card.last_line) return { kind: "line", text: card.last_line };
  return { kind: "empty" };
}

export function stateLineText(state: CardState): string | null {
  switch (state.kind) {
    case "running":
      return "Working…";
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
