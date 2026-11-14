import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import { isSettled } from "./taskBlocks";

const LINES_WITHOUT_STEPS = 5;
const SECOND = 1000;

export type TrailState =
  | "done"
  | "running"
  | "waiting"
  | "failed"
  | "stopped"
  | "next";

export interface TrailStep {
  id: string;
  title: string;
  state: TrailState;
  /** The line that named the step; null for one the trail adds itself. */
  line: TaskActivity | null;
  startedAt: number | null;
  /** How long a finished step took. */
  tookMs: number | null;
  /** What was done inside the step, oldest first. */
  lines: TaskActivity[];
}

interface Group {
  head: TaskActivity | null;
  lines: TaskActivity[];
}

/** A task's activity as a trail, oldest step first: each step the worker
 *  named with the tool calls made under it, the last one in the task's own
 *  state, and what comes next as a step still to walk. A task with no named
 *  steps yet shows its latest lines as the steps. */
export function trailOf(task: Task, activity: TaskActivity[]): TrailStep[] {
  const lines = [...activity].reverse();
  const groups = lines.some((line) => line.is_milestone)
    ? groupsOf(lines)
    : lines
        .slice(-LINES_WITHOUT_STEPS)
        .map((line) => ({ head: line, lines: [] }));

  const steps = groups.map((group, index): TrailStep => {
    const isLast = index === groups.length - 1;
    const startedAt = startOf(group);
    const endedAt = isLast ? endOf(task, group) : startOf(groups[index + 1]);
    const state = isLast ? lastState(task.status) : "done";
    return {
      id: group.head?.id ?? "start",
      title: group.head?.text ?? "Getting started",
      state,
      line: group.head,
      startedAt,
      tookMs: state === "done" ? span(startedAt, endedAt) : null,
      lines: group.lines,
    };
  });

  if (steps.length === 0 || isSettled(task.status) || !task.next_step) {
    return steps;
  }
  return [
    ...steps,
    {
      id: "next",
      title: task.next_step,
      state: "next",
      line: null,
      startedAt: null,
      tookMs: null,
      lines: [],
    },
  ];
}

function groupsOf(lines: TaskActivity[]) {
  return lines.reduce<Group[]>((groups, line) => {
    if (line.is_milestone) return [...groups, { head: line, lines: [] }];
    const open = groups.at(-1) ?? { head: null, lines: [] };
    return [...groups.slice(0, -1), { ...open, lines: [...open.lines, line] }];
  }, []);
}

function lastState(status: Task["status"]): TrailState {
  if (status === "IN_PROGRESS") return "running";
  if (status === "NEEDS_INPUT") return "waiting";
  if (status === "FAILED") return "failed";
  if (status === "CANCELLED") return "stopped";
  return "done";
}

function startOf(group: Group) {
  const first = group.head ?? group.lines[0];
  return first ? timeOf(first.created_at) : null;
}

function endOf(task: Task, group: Group) {
  if (task.completed_at) return timeOf(task.completed_at);
  const last = group.lines.at(-1);
  return last ? timeOf(last.created_at) : null;
}

function timeOf(date: Date | string) {
  const time = new Date(date).getTime();
  return Number.isFinite(time) ? time : null;
}

function span(from: number | null, to: number | null) {
  if (from === null || to === null || to < from) return null;
  return to - from;
}

/** "12s", "1m 4s", "1h 2m": how long a step took, or has been going. */
export function formatDuration(ms: number | null) {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return "";
  const seconds = Math.round(ms / SECOND);
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return seconds % 60 ? `${minutes}m ${seconds % 60}s` : `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  return minutes % 60 ? `${hours}h ${minutes % 60}m` : `${hours}h`;
}
