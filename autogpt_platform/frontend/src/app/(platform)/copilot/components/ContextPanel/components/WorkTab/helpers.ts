import {
  type ChatDelegation,
  type LiveDelegationStatus,
  formatElapsed,
} from "../../../../delegations";
import { type DelegationTone, formatCost } from "../../../../delegationViews";

export const BADGE_VARIANT: Record<
  DelegationTone,
  "success" | "error" | "warning" | "info"
> = {
  working: "info",
  waiting: "warning",
  done: "success",
  failed: "error",
  muted: "info",
};

export const DOT_CLASS: Record<DelegationTone, string> = {
  working: "bg-purple-500",
  waiting: "bg-amber-500",
  done: "bg-emerald-500",
  failed: "bg-red-500",
  muted: "bg-zinc-400",
};

/** The muted line under a row: what the teammate is doing or what came of it. */
export function delegationLine(
  delegation: ChatDelegation,
  status: LiveDelegationStatus,
  question: string | null,
  latestText: string | null,
): string {
  switch (status) {
    case "proposed":
      return "Waiting for your approval";
    case "queued":
      return "Starts as soon as they are free";
    case "running":
      return latestText ?? "Working on it";
    case "needs-input":
      return question ?? "Asked you a question";
    case "completed": {
      const files = delegation.files.length;
      return files > 0
        ? `Reported back · ${files} file${files === 1 ? "" : "s"}`
        : "Reported back";
    }
    case "transferred":
      return "Now owns this task and reports to you directly";
    case "failed":
      return delegation.error ?? "Stopped before finishing";
    case "cancelled":
      return "Stopped";
    case "unknown":
      return "Live updates stopped";
  }
}

export function workTotals(
  delegations: ChatDelegation[],
  statusOf: (delegation: ChatDelegation) => LiveDelegationStatus,
) {
  const count = (statuses: LiveDelegationStatus[]) =>
    delegations.filter((d) => statuses.includes(statusOf(d))).length;
  const costs = delegations.flatMap((d) =>
    d.costUsd === null ? [] : [d.costUsd],
  );
  return [
    { label: "Working", value: String(count(["running", "queued"])) },
    { label: "Waiting", value: String(count(["proposed", "needs-input"])) },
    {
      label: "Spent",
      value:
        formatCost(costs.length ? costs.reduce((a, b) => a + b, 0) : null) ??
        "—",
    },
  ];
}

export function countExperts(delegations: ChatDelegation[]): number {
  return new Set(
    delegations.map((d) => d.expert?.id ?? d.expertId ?? d.toolCallId),
  ).size;
}

function clock(iso: string | null): string | null {
  const at = Date.parse(iso ?? "");
  if (!Number.isFinite(at)) return null;
  return new Date(at).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function joined(parts: (string | null | false)[]): string {
  return parts.filter((part): part is string => !!part).join(" · ");
}

interface SubtitleInput {
  status: LiveDelegationStatus;
  elapsedSeconds: number | null;
  askedAt: number | null;
  resumed: boolean;
}

/** The muted line under the detail's title: state, time, cost. */
export function delegationSubtitle(
  delegation: ChatDelegation,
  { status, elapsedSeconds, askedAt, resumed }: SubtitleInput,
): string | null {
  const elapsed = formatElapsed(elapsedSeconds);
  const cost = formatCost(delegation.costUsd);
  switch (status) {
    case "queued":
      return "Queued · starts as soon as they are free";
    case "running":
      return joined([resumed ? "Resumed" : "Working", elapsed, cost]);
    case "needs-input":
      return joined([
        "Paused with a question",
        askedAt !== null && formatElapsed((Date.now() - askedAt) / 1000),
      ]);
    case "completed": {
      const span =
        clock(delegation.startedAt) && clock(delegation.finishedAt)
          ? `${clock(delegation.startedAt)} → ${clock(delegation.finishedAt)}`
          : null;
      return joined([span ?? "Done", elapsed, cost]);
    }
    case "failed":
      return joined([
        `Stopped${clock(delegation.finishedAt) ? ` ${clock(delegation.finishedAt)}` : ""}`,
        cost,
      ]);
    case "cancelled":
      return "You stopped this hand-off";
    default:
      return null;
  }
}

/** Only a budget or cap failure can be fixed by raising it. */
export function isBudgetError(error: string | null): boolean {
  return !!error && /budget|cap\b|spend|limit/i.test(error);
}
