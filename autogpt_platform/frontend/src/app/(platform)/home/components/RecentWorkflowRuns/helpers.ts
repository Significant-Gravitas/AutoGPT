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
