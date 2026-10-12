import type { DelegationCounts } from "@/app/api/__generated__/models/delegationCounts";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import type { DelegationSummaryStatus } from "@/app/api/__generated__/models/delegationSummaryStatus";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";

export type DelegationGroup = "needs_you" | "working" | "completed" | "failed";
export type DelegationFilter = "all" | DelegationGroup;

export interface DelegationFilterOption {
  value: DelegationFilter;
  label: string;
}

export const OTTO_DELEGATION_FILTERS: DelegationFilterOption[] = [
  { value: "all", label: "All" },
  { value: "needs_you", label: "Needs you" },
  { value: "working", label: "Working" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
];

export const EXPERT_DELEGATION_FILTERS: DelegationFilterOption[] = [
  { value: "all", label: "All" },
  { value: "working", label: "In progress" },
  { value: "needs_you", label: "Needs review" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
];

const GROUP_OF: Record<DelegationSummaryStatus, DelegationGroup> = {
  proposed: "needs_you",
  needs_input: "needs_you",
  queued: "working",
  running: "working",
  completed: "completed",
  failed: "failed",
  cancelled: "failed",
};

export function getDelegationGroup(status: DelegationSummaryStatus) {
  return GROUP_OF[status];
}

export function filterDelegations(
  delegations: DelegationSummary[],
  filter: DelegationFilter,
) {
  if (filter === "all") return delegations;
  return delegations.filter((d) => GROUP_OF[d.status] === filter);
}

export type BadgeTone = "warning" | "working" | "info" | "success" | "error";

const BADGES: Record<
  DelegationSummaryStatus,
  { label: string; tone: BadgeTone }
> = {
  proposed: { label: "Waiting for you", tone: "warning" },
  needs_input: { label: "Needs you", tone: "warning" },
  queued: { label: "Queued", tone: "info" },
  running: { label: "Working", tone: "working" },
  completed: { label: "Done", tone: "success" },
  failed: { label: "Failed", tone: "error" },
  cancelled: { label: "Stopped", tone: "info" },
};

export function getDelegationBadge(status: DelegationSummaryStatus) {
  return BADGES[status];
}

export function formatUsd(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  return `$${value.toFixed(2)}`;
}

export function formatDuration(
  seconds: number | null | undefined,
): string | null {
  if (seconds === null || seconds === undefined) return null;
  const whole = Math.max(0, Math.round(seconds));
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export function formatFiles(count: number): string | null {
  if (count <= 0) return null;
  return `${count} file${count === 1 ? "" : "s"}`;
}

function startOfDay(at: Date) {
  return new Date(at.getFullYear(), at.getMonth(), at.getDate()).getTime();
}

function daysAgo(at: Date, now: Date) {
  return Math.round((startOfDay(now) - startOfDay(at)) / 86_400_000);
}

export function formatClock(at: Date) {
  return at.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
}

/** "10:48" today, "Yesterday", a weekday within the week, else the date. */
export function formatDelegationTime(value: Date | string, now = new Date()) {
  const at = new Date(value);
  const days = daysAgo(at, now);
  if (days <= 0) return formatClock(at);
  if (days === 1) return "Yesterday";
  if (days < 7) return at.toLocaleDateString([], { weekday: "short" });
  return at.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** "today 10:41", "yesterday", "Thu": when in the run's own meta line. */
export function formatDelegationDay(value: Date | string, now = new Date()) {
  const at = new Date(value);
  const days = daysAgo(at, now);
  if (days <= 0) return `today ${formatClock(at)}`;
  if (days === 1) return "yesterday";
  return formatDelegationTime(at, now);
}

export function isToday(value: Date | string, now = new Date()) {
  return daysAgo(new Date(value), now) === 0;
}

export function isThisWeek(value: Date | string, now = new Date()) {
  return daysAgo(new Date(value), now) < 7;
}

const STATE_WORDS: Partial<Record<DelegationSummaryStatus, string>> = {
  proposed: "waiting for your approval",
  needs_input: "asked a question",
  queued: "queued",
  running: "working",
  cancelled: "stopped",
};

function outcomeParts(delegation: DelegationSummary) {
  const finished =
    delegation.status === "completed" || delegation.status === "failed";
  return [
    finished ? formatDuration(delegation.elapsed_seconds) : null,
    formatUsd(delegation.cost_usd),
    formatFiles(delegation.files_count),
  ];
}

/** Otto's list: "Otto → Devon · working · $0.04". */
export function getDelegationMeta(delegation: DelegationSummary) {
  const route = `${AUTOPILOT_NAME} → ${delegation.expert?.name ?? "an expert"}`;
  return [route, STATE_WORDS[delegation.status], ...outcomeParts(delegation)]
    .filter(Boolean)
    .join(" · ");
}

/** An expert's list: "From Otto · today 10:41 · 6m 40s · $0.31 · 1 file". */
export function getHandoffMeta(
  delegation: DelegationSummary,
  now = new Date(),
) {
  const from = delegation.delegated_by_expert_id
    ? "From a teammate"
    : `From ${AUTOPILOT_NAME}`;
  return [
    from,
    formatDelegationDay(delegation.created_at, now),
    STATE_WORDS[delegation.status],
    ...outcomeParts(delegation),
  ]
    .filter(Boolean)
    .join(" · ");
}

export function getDelegationHref(delegation: DelegationSummary) {
  const sessionId = delegation.parent_session_id ?? delegation.sub_session_id;
  if (!sessionId) return null;
  return `/copilot?sessionId=${encodeURIComponent(sessionId)}`;
}

export function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function getDelegationKey(delegation: DelegationSummary) {
  return (
    delegation.sub_session_id ??
    delegation.review_id ??
    `${delegation.title}-${new Date(delegation.created_at).getTime()}`
  );
}

const MODE_LABELS: Record<string, string> = {
  ask_first: "Ask first",
  auto: "Auto",
  unsupervised: "Unsupervised",
};

export function getModeLabel(mode: string | null | undefined) {
  return mode ? (MODE_LABELS[mode] ?? mode) : null;
}

/** "3 delegations today · 2 working · 1 needs you · $0.40 spent". */
export function getOttoSummaryLine(
  todayCount: number,
  summary: DelegationCounts,
) {
  return [
    `${plural(todayCount, "delegation")} today`,
    summary.working > 0 ? `${summary.working} working` : null,
    summary.needs_you > 0
      ? `${summary.needs_you} need${summary.needs_you === 1 ? "s" : ""} you`
      : null,
    `${formatUsd(summary.spent_today_usd)} spent`,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** "5 delegations this week · 4 done · 1 failed · $1.67 spent". */
export function getExpertSummaryLine(
  delegations: DelegationSummary[],
  now = new Date(),
) {
  const week = delegations.filter((d) => isThisWeek(d.created_at, now));
  const done = week.filter((d) => d.status === "completed").length;
  const failed = week.filter((d) => d.status === "failed").length;
  const spent = week.reduce((sum, d) => sum + (d.cost_usd ?? 0), 0);
  return [
    `${plural(week.length, "delegation")} this week`,
    done > 0 ? `${done} done` : null,
    failed > 0 ? `${failed} failed` : null,
    `${formatUsd(spent)} spent`,
  ]
    .filter(Boolean)
    .join(" · ");
}
