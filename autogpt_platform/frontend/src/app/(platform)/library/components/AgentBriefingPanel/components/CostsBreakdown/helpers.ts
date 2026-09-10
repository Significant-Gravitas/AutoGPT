import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export interface AgentLookupEntry {
  libraryAgentId: string;
  name: string;
  imageUrl?: string | null;
}

export function buildAgentLookup(
  agents: LibraryAgent[],
): Map<string, AgentLookupEntry> {
  const map = new Map<string, AgentLookupEntry>();
  for (const agent of agents) {
    map.set(agent.graph_id, {
      libraryAgentId: agent.id,
      name: agent.name,
      imageUrl: agent.image_url,
    });
  }
  return map;
}

export function formatRelativeDate(input: string | Date): string {
  const date = input instanceof Date ? input : new Date(input);
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
