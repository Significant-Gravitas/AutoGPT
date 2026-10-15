import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";

export const TODO = "todo";
export const IN_PROGRESS = "in-progress";
export const DONE = "done";
export const TASK_CARD_MIME = "application/x-task-card";

const STALE_AFTER_MS = 3 * 24 * 60 * 60 * 1000;

export interface BoardColumnDef {
  id: string;
  name: string;
}

export interface Placement {
  columnId: string;
  // The session's updated_at when it was moved. Newer activity drops the
  // placement so the card follows the conversation again.
  updatedAt: string;
}

export interface BoardState {
  columns: BoardColumnDef[];
  placements: Record<string, Placement>;
}

export const DEFAULT_COLUMNS: BoardColumnDef[] = [
  { id: TODO, name: "To do" },
  { id: IN_PROGRESS, name: "In progress" },
  { id: DONE, name: "Done" },
];

export function isDefaultColumn(columnId: string) {
  return DEFAULT_COLUMNS.some((column) => column.id === columnId);
}

export function parseBoardState(raw: string | null): BoardState {
  const fallback = { columns: DEFAULT_COLUMNS, placements: {} };
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as Partial<BoardState>;
    const custom = (parsed.columns ?? []).filter(
      (column) => !isDefaultColumn(column.id),
    );
    return {
      columns: [...DEFAULT_COLUMNS, ...custom],
      placements: parsed.placements ?? {},
    };
  } catch {
    return fallback;
  }
}

export function isSessionRunning(session: SessionSummaryResponse) {
  return (
    session.is_processing ||
    session.chat_status === "running" ||
    session.chat_status === "queued"
  );
}

export function getCardColumnId(
  session: SessionSummaryResponse,
  placement: Placement | undefined,
  columnIds: ReadonlySet<string>,
  now = Date.now(),
) {
  if (isSessionRunning(session)) return IN_PROGRESS;
  if (
    placement &&
    columnIds.has(placement.columnId) &&
    placement.updatedAt === session.updated_at
  ) {
    return placement.columnId;
  }
  if (!session.last_message) return TODO;
  if (now - new Date(session.updated_at).getTime() > STALE_AFTER_MS) {
    return DONE;
  }
  return IN_PROGRESS;
}

export function groupSessionsIntoColumns(
  sessions: SessionSummaryResponse[],
  state: BoardState,
) {
  const columnIds = new Set(state.columns.map((column) => column.id));
  const byColumn = new Map<string, SessionSummaryResponse[]>(
    state.columns.map((column) => [column.id, []]),
  );
  const sorted = [...sessions].sort(
    (a, b) =>
      new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
  );
  for (const session of sorted) {
    const columnId = getCardColumnId(
      session,
      state.placements[session.id],
      columnIds,
    );
    byColumn.get(columnId)?.push(session);
  }
  return state.columns.map((column) => ({
    ...column,
    sessions: byColumn.get(column.id) ?? [],
  }));
}

export function makeColumnId(name: string) {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${slug || "column"}-${Date.now().toString(36)}`;
}

export function getPreviewLine(text: string) {
  return text.replace(/^[#>*\-\s]+/, "").replace(/[*_`]/g, "");
}
