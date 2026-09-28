import {
  type ChatDelegation,
  type DelegationTone,
  type LiveDelegationStatus,
  formatElapsed,
} from "../../../../delegations";

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

/** The task's name in the panel: the first line of what Otto asked for. */
export function delegationTitle(delegation: ChatDelegation): string {
  const line = delegation.prompt?.split("\n")[0].trim();
  if (!line) return "Task for " + (delegation.expert?.name ?? "an expert");
  return line.length > 80 ? `${line.slice(0, 77)}…` : line;
}

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

export function delegationSubtitle(
  status: LiveDelegationStatus,
  elapsedSeconds: number | null,
): string | null {
  const elapsed = formatElapsed(elapsedSeconds);
  switch (status) {
    case "running":
      return elapsed ? `Working · ${elapsed}` : "Working";
    case "completed":
      return elapsed ? `Done in ${elapsed}` : "Done";
    case "needs-input":
      return "Paused with a question";
    case "failed":
      return "Stopped";
    default:
      return null;
  }
}

export function threadHref(delegation: ChatDelegation): string | null {
  if (delegation.link) return delegation.link;
  return delegation.subSessionId
    ? `/copilot?sessionId=${delegation.subSessionId}`
    : null;
}
