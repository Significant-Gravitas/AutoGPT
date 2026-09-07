import { ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { formatDistanceToNow } from "date-fns";

export const WORK_FILTERS = [
  { value: "all", label: "All work" },
  { value: "needs-review", label: "Needs review" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
  { value: "running", label: "In progress" },
] as const;

export type WorkFilter = (typeof WORK_FILTERS)[number]["value"];

export function filterExpertRuns(runs: ExpertRun[], filter: WorkFilter) {
  if (filter === "all") return runs;
  if (filter === "needs-review") return runs.filter((run) => run.needs_review);
  return runs.filter((run) => {
    const status = run.status.toLowerCase();
    if (filter === "completed") return status === "completed";
    if (filter === "failed")
      return status === "failed" || status === "terminated";
    return status === "running" || status === "queued";
  });
}

export function getWorkEmptyMessage(filter: WorkFilter) {
  if (filter === "all")
    return "No completed work yet. Finished runs will show up here.";
  if (filter === "needs-review") return "Nothing is waiting on your review.";
  return "No work matches.";
}

const SOURCE_LABELS: Record<string, string> = {
  scheduled: "Scheduled",
  trigger: "Triggered",
  manual: "Run manually",
};

function toDate(value: unknown): Date | null {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(String(value));
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatRunDuration(startedAt: unknown, endedAt: unknown) {
  const start = toDate(startedAt);
  const end = toDate(endedAt);
  if (!start || !end) return null;
  const totalSeconds = Math.max(
    0,
    Math.round((end.getTime() - start.getTime()) / 1000),
  );
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

/** Older backends send no `source`; the row simply omits that part. */
export function getRunMeta(run: ExpertRun) {
  const parts: string[] = [];
  const source = run.source ? SOURCE_LABELS[run.source] : null;
  if (source) parts.push(source);
  const started = toDate(run.started_at);
  if (started) {
    parts.push(`${formatDistanceToNow(started, { addSuffix: true })}`);
  }
  const duration = formatRunDuration(run.started_at, run.ended_at);
  if (duration) parts.push(duration);
  return { parts, startedAt: started };
}
