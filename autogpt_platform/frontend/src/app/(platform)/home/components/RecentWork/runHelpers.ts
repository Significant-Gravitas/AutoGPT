import type { SitrepItemData } from "@/app/(platform)/library/types";

export function buildRunPrompt(run: SitrepItemData): string {
  if (run.priority === "success") {
    return `${run.agentName} just finished a run — can you summarize what it did?`;
  }
  switch (run.status) {
    case "error":
      return `What happened with ${run.agentName}? It has an error — can you check?`;
    case "running":
      return `Give me a status update on ${run.agentName} — what has it done so far?`;
    case "idle":
      return `${run.agentName} hasn't run recently. Should I keep it or update and re-run it?`;
    default:
      return `Tell me about ${run.agentName} — what's its current status?`;
  }
}

export function buildAskHref(run: SitrepItemData): string {
  return `/copilot?autosubmit=true#prompt=${encodeURIComponent(buildRunPrompt(run))}`;
}

interface RunStatus {
  label: string;
  dot: string;
  pulse: boolean;
}

export function getRunStatus(run: SitrepItemData): RunStatus {
  if (run.priority === "success") {
    return { label: "Completed", dot: "bg-emerald-500", pulse: false };
  }
  switch (run.status) {
    case "running":
      return { label: "Running", dot: "bg-blue-500", pulse: true };
    case "error":
      return { label: "Error", dot: "bg-red-500", pulse: false };
    case "listening":
      return { label: "Listening", dot: "bg-purple-500", pulse: true };
    case "scheduled":
      return { label: "Scheduled", dot: "bg-amber-500", pulse: false };
    case "idle":
      return { label: "Idle", dot: "bg-zinc-400", pulse: false };
    default:
      // The union is exhaustive today; a status the feed adds later reads as
      // idle rather than crashing the row.
      return { label: "Idle", dot: "bg-zinc-400", pulse: false };
  }
}
