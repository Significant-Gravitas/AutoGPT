import type { Task } from "@/app/api/__generated__/models/task";
import { SessionKey, sessionStorage } from "@/services/storage/session-storage";
import { OTTO_VALUE } from "./helpers";
import type { RepeatChoice, RepeatKind } from "./repeat";
import type { StartChoice } from "./startChoice";
import { NO_OPTIONS, type TaskOptions } from "./taskOptions";

/** What the create dialog holds before it is sent. */
export interface TaskDraft {
  text: string;
  owner: string;
  options: TaskOptions;
}

interface Stored {
  userID: string;
  text: string;
  owner: string;
  doneWhen: string;
  goal: string;
  start: StartChoice | null;
  repeat: RepeatChoice | null;
}

const REPEAT_KINDS: RepeatKind[] = ["hourly", "daily", "weekdays", "weekly"];

function asString(value: unknown) {
  return typeof value === "string" ? value : "";
}

function asStart(value: unknown): StartChoice | null {
  if (!value || typeof value !== "object" || !("kind" in value)) return null;
  if (value.kind === "picking") return { kind: "picking" };
  if (value.kind === "quick" && "label" in value) {
    return { kind: "quick", label: asString(value.label) };
  }
  if (value.kind === "at" && "at" in value) {
    const at = new Date(asString(value.at));
    return Number.isNaN(at.getTime()) ? null : { kind: "at", at };
  }
  return null;
}

function asRepeat(value: unknown): RepeatChoice | null {
  if (!value || typeof value !== "object" || !("kind" in value)) return null;
  const kind = REPEAT_KINDS.find((entry) => entry === value.kind);
  if (!kind) return null;
  const time = "time" in value ? asString(value.time) : "";
  const day = "day" in value && typeof value.day === "number" ? value.day : 1;
  return { kind, time, day };
}

/** This user's unsent draft from earlier in the browser session. */
export function readDraft(userID: string | undefined): TaskDraft | null {
  if (!userID) return null;
  try {
    const stored: unknown = JSON.parse(
      sessionStorage.get(SessionKey.TASK_DRAFT) ?? "null",
    );
    if (!stored || typeof stored !== "object") return null;
    if (!("userID" in stored) || stored.userID !== userID) return null;
    const read = (key: string) =>
      key in stored ? (stored as Record<string, unknown>)[key] : undefined;
    return {
      text: asString(read("text")),
      owner: asString(read("owner")) || OTTO_VALUE,
      options: {
        ...NO_OPTIONS,
        doneWhen: asString(read("doneWhen")),
        goal: asString(read("goal")),
        start: asStart(read("start")),
        repeat: asRepeat(read("repeat")),
      },
    };
  } catch {
    return null;
  }
}

/** Keep the draft for this user, or forget it (null). */
export function writeDraft(
  userID: string | undefined,
  draft: TaskDraft | null,
) {
  if (!userID) return;
  if (!draft) {
    sessionStorage.clean(SessionKey.TASK_DRAFT);
    return;
  }
  const stored: Stored = {
    userID,
    text: draft.text,
    owner: draft.owner,
    doneWhen: draft.options.doneWhen,
    goal: draft.options.goal,
    start: draft.options.start,
    repeat: draft.options.repeat,
  };
  sessionStorage.set(SessionKey.TASK_DRAFT, JSON.stringify(stored));
}

/** A new task like this one: its words, goal, finish line and owner, but
 *  not its schedule. */
export function draftFromTask(task: Task): TaskDraft {
  const description = task.description?.trim() ?? "";
  const text = !description
    ? task.title
    : description.startsWith(task.title)
      ? description
      : `${task.title}\n${description}`;
  return {
    text,
    owner: task.owner_expert_id ?? OTTO_VALUE,
    options: {
      ...NO_OPTIONS,
      doneWhen: task.done_when ?? "",
      goal: task.goal ?? "",
    },
  };
}
