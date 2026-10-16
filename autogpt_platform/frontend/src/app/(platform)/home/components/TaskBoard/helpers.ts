import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";

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
    new Date(placement.updatedAt).getTime() ===
      new Date(session.updated_at).getTime()
  ) {
    return placement.columnId;
  }
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

export const SCHEDULED = "scheduled";

export interface ScheduleCardData {
  id: string;
  kind: "copilot_turn" | "graph";
  title: string;
  expertId: string | null;
  recurrence: string;
  nextRunTime: string | null;
  href: string;
}

const FOLLOWUPS_PAGE = "/library/followups";

function nextRunMs(nextRunTime: string | null) {
  if (!nextRunTime) return Number.POSITIVE_INFINITY;
  const ms = new Date(nextRunTime).getTime();
  return Number.isNaN(ms) ? Number.POSITIVE_INFINITY : ms;
}

function describeRecurrence(cron: string | null | undefined) {
  return cron ? safeHumanizeCronExpression(cron) : "Runs once";
}

export function buildScheduleCards(
  followups: CopilotTurnJobInfo[],
  workflows: GraphExecutionJobInfo[],
): ScheduleCardData[] {
  const cards: ScheduleCardData[] = [
    ...followups.map((followup) => ({
      id: followup.id,
      kind: "copilot_turn" as const,
      title: getPreviewLine(followup.message ?? "") || followup.name,
      expertId: followup.expert_id ?? null,
      recurrence: describeRecurrence(followup.cron),
      nextRunTime: followup.next_run_time || null,
      href: followup.session_id
        ? `/home?sessionId=${encodeURIComponent(followup.session_id)}`
        : FOLLOWUPS_PAGE,
    })),
    ...workflows.map((workflow) => ({
      id: workflow.id,
      kind: "graph" as const,
      title: workflow.agent_name || workflow.name || "Scheduled workflow",
      expertId: workflow.expert_id ?? null,
      recurrence: describeRecurrence(workflow.cron),
      nextRunTime: workflow.next_run_time || null,
      href: FOLLOWUPS_PAGE,
    })),
  ];
  return cards.sort(
    (a, b) => nextRunMs(a.nextRunTime) - nextRunMs(b.nextRunTime),
  );
}

// Earliest pending follow-up per chat, for the clock badge on its card.
export function getNextFollowupBySession(followups: CopilotTurnJobInfo[]) {
  const bySession = new Map<string, string>();
  for (const followup of followups) {
    const { session_id: sessionId, next_run_time: nextRun } = followup;
    if (!sessionId || !nextRun) continue;
    const current = bySession.get(sessionId);
    if (!current || nextRunMs(nextRun) < nextRunMs(current)) {
      bySession.set(sessionId, nextRun);
    }
  }
  return bySession;
}

export function formatTimeUntil(iso: string | null, now = Date.now()) {
  if (!iso) return null;
  const ms = new Date(iso).getTime() - now;
  if (Number.isNaN(ms)) return null;
  const minutes = Math.round(ms / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `in ${hours}h`;
  return `in ${Math.round(hours / 24)}d`;
}

export function formatRunMoment(iso: string, locale?: string) {
  return new Intl.DateTimeFormat(locale, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(iso));
}
