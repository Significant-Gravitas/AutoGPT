import {
  type ChatDelegation,
  type LiveDelegationStatus,
  formatElapsed,
} from "../../../../delegations";
import type { DelegationTone } from "../../../../delegationViews";

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
