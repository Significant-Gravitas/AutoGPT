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

const UNKNOWN_PRESENCE = { color: "bg-zinc-300", label: "Unknown" };

export function getPresenceColor(status: HomeAgentStatusStatus): string {
  return (PRESENCE[status] ?? UNKNOWN_PRESENCE).color;
}

export function getPresenceLabel(status: HomeAgentStatusStatus): string {
  return (PRESENCE[status] ?? UNKNOWN_PRESENCE).label;
}

export function getExpertHref(expertID: string): string {
  return `/team/${encodeURIComponent(expertID)}`;
}
