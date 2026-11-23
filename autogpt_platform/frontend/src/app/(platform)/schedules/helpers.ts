import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskRun } from "@/app/api/__generated__/models/taskRun";
import type { UpcomingRuns } from "@/app/api/__generated__/models/upcomingRuns";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";

import type {
  RunOutcome,
  Schedule,
  ScheduleLabels,
} from "./affect/agent-schedules";

export type ScheduleKind = "task" | "workflow";

const MIN = 60_000;
/** A task at work shows as running on the line for at least this long. */
const RUN_SPAN = 10 * MIN;

/** The block's words, in ours: workflows and Otto, never agents. */
/** What the empty page offers to start with: Otto opens with it typed. */
export const EXAMPLE_SCHEDULE =
  "Every Monday at 9am, send me a summary of last week";

/** Otto's new chat (Home) with a message ready in the composer, not sent. */
export function askOttoHref(text: string): string {
  return `/home#prompt=${encodeURIComponent(text)}`;
}

export const LABELS: ScheduleLabels = {
  activity: "Runs around now",
  perTen: "",
  perTenOne: "",
  noScheduleText:
    "Pick a schedule to see its next runs and how the last ones went.",
  empty: "No schedules yet",
  emptyText: `Ask Otto to do something every day, week or month, or schedule a workflow. For example: “${EXAMPLE_SCHEDULE}.”`,
  emptyAction: "Ask Otto this",
};

/** The empty state's words when the schedules could not be loaded. */
export const ERROR_LABELS: ScheduleLabels = {
  ...LABELS,
  empty: "We couldn't load your schedules",
  emptyText: "Something went wrong while loading them. Try again in a moment.",
  emptyAction: "Try again",
};

/** The id a row carries: kind and id, so a task and a schedule never clash. */
export function rowID(kind: ScheduleKind, id: string): string {
  return `${kind}:${id}`;
}

export function parseRowID(
  row: string,
): { kind: ScheduleKind; id: string } | null {
  const [kind, ...rest] = row.split(":");
  if (kind !== "task" && kind !== "workflow") return null;
  return { kind, id: rest.join(":") };
}

/** The tasks the page lists: repeating, watching, or waiting for one start. */
export function isScheduledTask(task: Task): boolean {
  if (task.is_suggested || task.is_archived) return false;
  if (task.repeat) return true;
  return Boolean(task.start_at) && task.status === "TODO";
}

function runsByID(upcoming: UpcomingRuns[], kind: ScheduleKind) {
  return new Map(
    upcoming
      .filter((entry) => entry.kind === kind)
      .map((entry) => [entry.id, (entry.runs ?? []).map(toTime)]),
  );
}

function toTime(value: Date | string): number {
  return new Date(value).getTime();
}

export function taskSchedule(
  task: Task,
  coming: number[],
  now: number,
): Schedule {
  const repeat = task.repeat;
  const started = task.started_at ? toTime(task.started_at) : null;
  const working = task.status === "IN_PROGRESS" && started !== null;
  const startAt = task.start_at ? [toTime(task.start_at)] : [];
  const past = started !== null ? [started] : [];
  // Without the coming times (the backend not restarted yet), a repeating
  // task still shows its next run: start_at holds it.
  const next = repeat && coming.length ? coming : startAt;
  const runs = repeat?.watch ? past : [...past, ...next];
  return {
    id: rowID("task", task.id),
    name: task.title,
    project: "Task",
    tone: "purple",
    prompt: task.goal || task.description || undefined,
    cadence: taskCadence(task),
    runs: Array.from(new Set(runs)),
    duration: working ? Math.max(RUN_SPAN, now - started + 5 * MIN) : undefined,
    paused: repeat?.paused ?? false,
    history: lastRunOf(task),
    canPause: Boolean(repeat),
    canRunNow: !working,
    canSkip: false,
  };
}

function taskCadence(task: Task): string {
  const repeat = task.repeat;
  if (repeat?.watch) return `When ${lowerFirst(repeat.watch.line)}`;
  if (repeat) return repeat.label;
  if (task.missed_start) return "Once – missed its start";
  return "Once";
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

function lastRunOf(task: Task): Schedule["history"] {
  const last = task.repeat?.last_run;
  if (!last || !task.started_at) return [];
  const at = toTime(task.completed_at ?? task.started_at);
  if (last === "done") {
    return [{ at, outcome: "done", summary: task.result || "Done" }];
  }
  const summary =
    last === "cancelled" ? "Stopped" : task.failure_reason || "Failed";
  return [{ at, outcome: "failed", summary }];
}

export function workflowSchedule(
  schedule: GraphExecutionJobInfo,
  coming: number[] | undefined,
): Schedule {
  const next = schedule.next_run_time ? [toTime(schedule.next_run_time)] : [];
  const zone = schedule.timezone ? ` (${schedule.timezone})` : "";
  return {
    id: rowID("workflow", schedule.id),
    name: schedule.name,
    project: "Workflow",
    tone: "teal",
    prompt: schedule.agent_name
      ? `Runs the workflow ${schedule.agent_name}.`
      : undefined,
    cadence: `${safeHumanizeCronExpression(schedule.cron)}${zone}`,
    runs: coming?.length ? coming : next,
    canPause: false,
    canRunNow: true,
    canSkip: false,
  };
}

interface BuildArgs {
  tasks: Task[];
  workflows: GraphExecutionJobInfo[];
  upcoming: UpcomingRuns[];
  now: number;
}

export function buildSchedules({
  tasks,
  workflows,
  upcoming,
  now,
}: BuildArgs): Schedule[] {
  const taskRuns = runsByID(upcoming, "task");
  const workflowRuns = runsByID(upcoming, "workflow");
  return [
    ...tasks
      .filter(isScheduledTask)
      .map((task) => taskSchedule(task, taskRuns.get(task.id) ?? [], now)),
    ...workflows.map((schedule) =>
      workflowSchedule(schedule, workflowRuns.get(schedule.id)),
    ),
  ];
}

/** A task's runs, newest first, as the panel's Last runs. */
export function taskRunHistory(runs: TaskRun[]): Schedule["history"] {
  return runs
    .filter((run) => run.outcome)
    .slice(0, 5)
    .map((run) => ({
      at: toTime(run.ended_at ?? run.started_at),
      outcome: run.outcome === "done" ? "done" : "failed",
      summary:
        run.outcome === "done"
          ? run.result || "Done"
          : run.outcome === "cancelled"
            ? "Stopped"
            : run.failure_reason || "Failed",
    }));
}

const EXECUTION_OUTCOMES: Partial<
  Record<GraphExecutionMeta["status"], RunOutcome>
> = {
  COMPLETED: "done",
  FAILED: "failed",
  TERMINATED: "failed",
  REVIEW: "needs",
};

const EXECUTION_WORDS: Record<RunOutcome, string> = {
  done: "Finished",
  failed: "Failed",
  needs: "Waiting for your review",
};

/** A workflow's last runs, newest first, as the panel's Last runs. */
export function executionHistory(
  executions: GraphExecutionMeta[],
): Schedule["history"] {
  return executions
    .flatMap((execution) => {
      const outcome = EXECUTION_OUTCOMES[execution.status];
      const at = execution.ended_at ?? execution.started_at;
      if (!outcome || !at) return [];
      return [{ at: toTime(at), outcome, summary: EXECUTION_WORDS[outcome] }];
    })
    .sort((a, b) => b.at - a.at)
    .slice(0, 5);
}
