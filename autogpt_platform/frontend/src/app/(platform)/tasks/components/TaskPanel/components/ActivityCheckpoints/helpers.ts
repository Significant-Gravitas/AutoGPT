import { STATUS_LABELS } from "../../../../data/helpers";
import type { TaskActivity } from "../../../../data/types";
import { firstLine, sentence, stripTaskKeys, whoOf } from "./lines";

export type CheckpointStatus = "running" | "done" | "failed" | "stopped";

export interface Checkpoint {
  id: string;
  /** One line of markdown. */
  title: string;
  actor: TaskActivity["actor"];
  status: CheckpointStatus;
  /** When it happened, ISO */
  at: string;
  cost?: number;
}

interface Options {
  /** Every entry, tool turns and bookkeeping included (Show work). */
  all: boolean;
  /** The result the Outcome card shows, so the feed doesn't repeat it. */
  resultSummary?: string | null;
}

/** The task's log, oldest first. By default only the checkpoints a person
 *  would mention: asked, question and answer, files, result, failure. */
export function toCheckpoints(
  items: TaskActivity[],
  running: boolean,
  { all, resultSummary }: Options,
): Checkpoint[] {
  const ordered = items.toSorted((a, b) =>
    a.created_at.localeCompare(b.created_at),
  );
  if (all)
    return ordered.map((item, i) =>
      everyStep(item, running && i === ordered.length - 1),
    );

  const kept = collapse(
    ordered.flatMap((item) => {
      const line = checkpointLine(item, resultSummary);
      return line
        ? [
            {
              ...line,
              title: stripTaskKeys(line.title),
              id: item.id,
              actor: item.actor,
              at: item.created_at,
            },
          ]
        : [];
    }),
  );
  const last = ordered.at(-1);
  if (!running || !last) return kept;
  if (kept.at(-1)?.id === last.id)
    return [
      ...kept.slice(0, -1),
      { ...kept[kept.length - 1], status: "running" },
    ];
  return [
    ...kept,
    {
      id: `live-${last.id}`,
      title: `${whoOf(last.actor)} is working on it`,
      actor: last.actor,
      status: "running",
      at: last.created_at,
    },
  ];
}

type Line = { title: string; status: CheckpointStatus };

function checkpointLine(
  item: TaskActivity,
  resultSummary?: string | null,
): Line | null {
  const who = whoOf(item.actor);
  const done = (title: string): Line => ({ title, status: "done" });
  switch (item.kind) {
    case "created":
      return done(`${who} created the task`);
    case "comment":
      return done(`${who} commented: ${firstLine(item.body)}`);
    case "question":
      return done(`${who} asked: ${firstLine(item.body)}`);
    case "answer":
      return done(`${who} answered: ${firstLine(item.body)}`);
    case "turn":
      if (!item.summary || sameText(item.summary, resultSummary)) return null;
      return done(sentence(who, firstLine(item.summary)));
    case "artifact_linked":
      return done(
        `${who} ${firstLine(item.body).replace(/^linked\s+/i, "added ")}`,
      );
    case "note":
      return item.body ? done(firstLine(item.body)) : null;
    case "status_change":
      return item.to_status === "cancelled"
        ? { title: `${who} cancelled the task`, status: "stopped" }
        : null;
    case "system":
      return systemLine(who, item.body);
    default:
      return null;
  }
}

/** Contract bookkeeping arrives as "system" lines; only the result and a
 *  failure are checkpoints. */
function systemLine(who: string, body: string): Line | null {
  if (body === "Finished" || body.startsWith("Outcome:"))
    return { title: `${who} finished the task`, status: "done" };
  const stuck = body.match(/^got stuck:\s*(.*)$/s);
  if (stuck)
    return {
      title: `${who} couldn't finish: ${firstLine(stuck[1])}`,
      status: "failed",
    };
  if (body === "Plan approved")
    return { title: `${who} approved the plan`, status: "done" };
  return null;
}

/** The same line twice in a row reads once, at its latest time. */
function collapse(lines: Checkpoint[]) {
  return lines.reduce<Checkpoint[]>((out, line) => {
    const prev = out.at(-1);
    if (prev && prev.title === line.title) return [...out.slice(0, -1), line];
    return [...out, line];
  }, []);
}

function sameText(a: string, b?: string | null) {
  if (!b) return false;
  const norm = (t: string) =>
    t
      .replace(/[\s*_#`>]+/g, " ")
      .trim()
      .toLowerCase();
  const x = norm(a);
  const y = norm(b);
  return x === y || x.startsWith(y) || y.startsWith(x);
}

function everyStep(item: TaskActivity, isLive: boolean): Checkpoint {
  return {
    id: item.id,
    title: upperFirst(stripTaskKeys(firstLine(rawTitle(item)))),
    actor: item.actor,
    status: isLive ? "running" : rawStatus(item),
    at: item.created_at,
    cost:
      item.kind === "turn" && item.metrics.cost_usd > 0
        ? item.metrics.cost_usd
        : undefined,
  };
}

function rawTitle(item: TaskActivity) {
  const who = whoOf(item.actor);
  switch (item.kind) {
    case "turn": {
      if (item.summary) return item.summary;
      const tools = [...new Set(item.tool_calls.map((c) => c.name))];
      return tools.length ? `${who}: ${tools.join(", ")}` : `${who} worked`;
    }
    case "comment":
      return `${who}: ${item.body}`;
    case "status_change":
      return `Moved to ${STATUS_LABELS[item.to_status]}`;
    case "question":
      return `${who} asked: ${item.body}`;
    case "answer":
      return `${who} answered: ${item.body}`;
    case "system":
      return /^[a-z]/.test(item.body) ? `${who} ${item.body}` : item.body;
    default:
      return item.body || "Updated";
  }
}

function rawStatus(item: TaskActivity): CheckpointStatus {
  if (item.kind === "status_change" && item.to_status === "cancelled")
    return "stopped";
  if (
    item.kind === "turn" &&
    item.tool_calls.some((call) => call.status === "error")
  )
    return "failed";
  return "done";
}

function upperFirst(text: string) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
