import type { HomeAgentStatusStatus } from "@/app/api/__generated__/models/homeAgentStatusStatus";

// Single source of truth so colour and label stay in lockstep and a new enum
// member fails compilation instead of silently defaulting.
const PRESENCE: Record<
  HomeAgentStatusStatus,
  { color: string; label: string }
> = {
  ready: { color: "bg-emerald-500", label: "Ready" },
  working: { color: "bg-amber-500", label: "Working" },
  needs_setup: { color: "bg-zinc-300", label: "Needs setup" },
  paused: { color: "bg-zinc-300", label: "Paused" },
  failed: { color: "bg-red-500", label: "Needs attention" },
};

export function getPresenceColor(status: HomeAgentStatusStatus): string {
  return PRESENCE[status].color;
}

export function getPresenceLabel(status: HomeAgentStatusStatus): string {
  return PRESENCE[status].label;
}

// Opening an expert via `?expertId=` adopts that expert's latest thread (or
// starts a fresh one), matching how the rest of the app links into copilot.
export function getExpertChatHref(expertId: string): string {
  return `/copilot?expertId=${encodeURIComponent(expertId)}`;
}
