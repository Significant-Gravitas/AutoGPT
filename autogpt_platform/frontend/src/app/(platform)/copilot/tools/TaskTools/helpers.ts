import { statusOf } from "@/app/(platform)/tasks/helpers";
import { isQueued, QUEUED_WORD } from "@/app/(platform)/tasks/queue";
import {
  formatStartTime,
  SCHEDULE_WORDS,
  scheduleStateOf,
  type RepeatLike,
} from "@/app/(platform)/tasks/schedule";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import {
  asItems,
  asObject,
  str,
} from "../../components/ToolChain/resultHelpers";

/** One task as a chat step shows it: no ids, links or flags on screen. */
export interface TaskLine {
  id: string;
  title: string;
  status: TaskStatus;
  ownerExpertID: string | null;
  startAt: string | null;
  missedStart: boolean;
  /** Started, but waiting for a free turn slot. */
  isQueued: boolean;
  repeat: RepeatLike | null;
}

export type TaskToolView =
  | { kind: "tasks"; tasks: TaskLine[] }
  | { kind: "archived" }
  | {
      kind: "finished";
      document: string | null;
      /** Why it was saved as not done; null when it was delivered. */
      notDone: string | null;
    };

/** finish_task's words when a worker saves its task as not done. */
const NOT_DONE_PREFIX = "Saved on the task as not done";

const ONE_TASK_TYPES = new Set([
  "task_created",
  "task_started",
  "task_updated",
  "task_details",
]);

/** What a task tool's result says, in words a person reads; null for an
 *  error or a result this view does not know, which then shows as is. */
export function taskToolViewOf(
  output: unknown,
  input: unknown = null,
): TaskToolView | null {
  const data = asObject(output);
  if (!data) return null;
  const type = str(data, "type");
  if (type === "task_archived") return { kind: "archived" };
  if (type === "task_finished") {
    return {
      kind: "finished",
      document: str(data, "document"),
      notDone: notDoneReason(data, input),
    };
  }
  if (type === "task_list") {
    const tasks = (asItems(data.tasks) ?? []).flatMap((item) => lineOf(item));
    return { kind: "tasks", tasks };
  }
  if (!type || !ONE_TASK_TYPES.has(type)) return null;
  const task = lineOf(data.task, str(data, "start_at"));
  return task.length > 0 ? { kind: "tasks", tasks: task } : null;
}

/** The worker's reason, when it saved the task as blocked or failed. */
function notDoneReason(data: Record<string, unknown>, input: unknown) {
  const args = asObject(input);
  const outcome = args && str(args, "outcome")?.trim().toLowerCase();
  const saidNotDone =
    (outcome && outcome !== "done") ||
    (str(data, "message") ?? "").startsWith(NOT_DONE_PREFIX);
  if (!saidNotDone) return null;
  return (args && str(args, "summary")?.trim()) || "";
}

function lineOf(value: unknown, startAt: string | null = null): TaskLine[] {
  const task = asObject(value);
  const id = task && str(task, "id");
  if (!task || !id) return [];
  const status = str(task, "status");
  return [
    {
      id,
      title: str(task, "title")?.trim() || "Untitled task",
      status: isTaskStatus(status) ? status : TaskStatus.TODO,
      ownerExpertID: str(task, "owner_expert_id"),
      startAt: str(task, "start_at") ?? startAt,
      missedStart: task.missed_start === true,
      isQueued: task.is_queued === true,
      repeat: repeatOf(task.repeat),
    },
  ];
}

/** Enough of a repeating task's rhythm to say where it stands. */
function repeatOf(value: unknown): RepeatLike | null {
  const repeat = asObject(value);
  const label = repeat && str(repeat, "label");
  if (!repeat || !label) return null;
  const watch = asObject(repeat.watch);
  return {
    label,
    paused: repeat.paused === true,
    watch: watch ? { line: str(watch, "line") ?? "" } : null,
  };
}

function isTaskStatus(value: string | null): value is TaskStatus {
  return Object.values(TaskStatus).some((status) => status === value);
}

/** Where the task stands in the board's words, or when it starts. */
export function taskStateWords(task: TaskLine, now: number) {
  const schedule = scheduleStateOf({
    status: task.status,
    start_at: task.startAt ? new Date(task.startAt) : null,
    missed_start: task.missedStart,
    repeat: task.repeat,
  });
  if (schedule === "scheduled" && task.startAt) {
    const at = formatStartTime(task.startAt, now);
    if (at) return `Starts ${at}`;
  }
  if (schedule) return SCHEDULE_WORDS[schedule];
  if (isQueued({ status: task.status, is_queued: task.isQueued })) {
    return QUEUED_WORD;
  }
  return statusOf(task.status).word;
}
