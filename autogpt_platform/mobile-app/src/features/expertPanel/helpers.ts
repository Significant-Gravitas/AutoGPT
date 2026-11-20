// Relative, type-only imports: jest has no `@/` alias.
import type {
  ComputerInfo,
  ExpertDetail,
  ExpertRunStatus,
  ScheduleInfo,
} from "../../lib/api/expertActivity";

export const UPCOMING_LIMIT = 3;
export const RECENT_LIMIT = 3;

const DAY = 24 * 60 * 60 * 1000;

/** The word under "<Name>'s computer"; null leaves the section out. */
export function computerState(info: ComputerInfo) {
  if (!info.e2b_active) return null;
  if (!info.box) return "Not started";
  if (info.box.state === "paused") return "Asleep";
  return info.screen_on ? "Awake, screen on" : "Awake";
}

export type RunTone = "done" | "failed" | "running" | "other";

export function runTone(status: ExpertRunStatus): RunTone {
  if (status === "completed") return "done";
  if (status === "failed") return "failed";
  if (status === "running" || status === "queued") return "running";
  return "other";
}

/** "Next: Mon 9:00 AM", or "Paused" for a job with no next run. */
export function nextRunText(nextRunTime: string, now = new Date()) {
  const then = new Date(nextRunTime);
  if (!nextRunTime || Number.isNaN(then.getTime())) return "Paused";
  const time = then.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const isToday = then.toDateString() === now.toDateString();
  if (isToday) return `Next: today ${time}`;
  if (then.getTime() - now.getTime() < 6 * DAY) {
    return `Next: ${then.toLocaleDateString([], { weekday: "short" })} ${time}`;
  }
  return `Next: ${then.getDate()} ${then.toLocaleDateString([], { month: "short" })}`;
}

interface ExpertSchedulesArgs {
  expertID: string;
  detail: ExpertDetail | undefined;
  graphSchedules: ScheduleInfo[];
  followupSchedules: ScheduleInfo[];
}

/**
 * The web's rule for workflow schedules (stamped with the expert, or the
 * schedule an installed workflow made), plus the expert's scheduled chats.
 * Soonest first; paused ones last.
 */
export function expertSchedules({
  expertID,
  detail,
  graphSchedules,
  followupSchedules,
}: ExpertSchedulesArgs) {
  const workflowScheduleIDs = new Set(
    (detail?.workflows ?? [])
      .map((workflow) => workflow.schedule_id)
      .filter((id): id is string => Boolean(id)),
  );
  const graphs = graphSchedules.filter(
    (schedule) =>
      schedule.expert_id === expertID || workflowScheduleIDs.has(schedule.id),
  );
  const chats = followupSchedules.filter(
    (schedule) => schedule.expert_id === expertID,
  );
  return [...graphs, ...chats].sort((a, b) => nextRunMs(a) - nextRunMs(b));
}

function nextRunMs(schedule: ScheduleInfo) {
  const time = new Date(schedule.next_run_time).getTime();
  return Number.isNaN(time) ? Number.POSITIVE_INFINITY : time;
}
