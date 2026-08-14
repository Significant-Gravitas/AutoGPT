import type { HomeAgentStatusStatus } from "@/app/api/__generated__/models/homeAgentStatusStatus";

// Presence palette requested by design: working is amber, paused/needs-setup
// read as unavailable (gray), everything else is a live green dot.
export function getPresenceColor(status: HomeAgentStatusStatus): string {
  switch (status) {
    case "working":
      return "bg-amber-500";
    case "paused":
    case "needs_setup":
      return "bg-zinc-300";
    default:
      return "bg-emerald-500";
  }
}

export function getPresenceLabel(status: HomeAgentStatusStatus): string {
  switch (status) {
    case "working":
      return "Working";
    case "paused":
      return "Paused";
    case "needs_setup":
      return "Needs setup";
    case "failed":
      return "Needs attention";
    default:
      return "Ready";
  }
}

// Opening an expert via `?expertId=` adopts that expert's latest thread (or
// starts a fresh one), matching how the rest of the app links into copilot.
export function getExpertChatHref(expertId: string): string {
  return `/copilot?expertId=${expertId}`;
}
