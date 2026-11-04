import { STATUS_LABELS } from "../../../../data/helpers";
import type { TaskActivity } from "../../../../data/types";

export type CheckpointStatus = "running" | "done" | "failed" | "stopped";

export interface Checkpoint {
  id: string;
  title: string;
  actor: TaskActivity["actor"];
  status: CheckpointStatus;
  /** When it happened, ISO */
  at: string;
  cost?: number;
}

const TITLE_CHARS = 90;

/** The task's log as one line of checkpoints, oldest first. */
export function toCheckpoints(
  items: TaskActivity[],
  running: boolean,
): Checkpoint[] {
  const ordered = items.toSorted((a, b) =>
    a.created_at.localeCompare(b.created_at),
  );
  return ordered.map((item, i) => ({
    id: item.id,
    title: clip(upperFirst(titleOf(item))),
    actor: item.actor,
    status: statusOf(item, running && i === ordered.length - 1),
    at: item.created_at,
    cost:
      item.kind === "turn" && item.metrics.cost_usd > 0
        ? item.metrics.cost_usd
        : undefined,
  }));
}

function titleOf(item: TaskActivity) {
  const who = item.actor?.name ?? "Otto";
  switch (item.kind) {
    case "turn": {
      if (item.summary) return item.summary;
      const calls = item.tool_calls.length;
      return calls
        ? `${who} worked · ${calls} tool call${calls === 1 ? "" : "s"}`
        : `${who} worked`;
    }
    case "comment":
      return `${who}: ${item.body}`;
    case "status_change":
      return `Moved to ${STATUS_LABELS[item.to_status]}`;
    case "question":
      return `${who} asked: ${item.body}`;
    case "answer":
      return `${who} answered: ${item.body}`;
    default:
      return item.body || "Updated";
  }
}

function statusOf(item: TaskActivity, isLive: boolean): CheckpointStatus {
  if (isLive) return "running";
  if (item.kind === "status_change" && item.to_status === "cancelled")
    return "stopped";
  if (
    item.kind === "turn" &&
    item.tool_calls.some((call) => call.status === "error")
  )
    return "failed";
  return "done";
}

function clip(text: string) {
  const line = text.replace(/\s+/g, " ").trim();
  return line.length > TITLE_CHARS
    ? `${line.slice(0, TITLE_CHARS - 1).trimEnd()}…`
    : line;
}

function upperFirst(text: string) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
