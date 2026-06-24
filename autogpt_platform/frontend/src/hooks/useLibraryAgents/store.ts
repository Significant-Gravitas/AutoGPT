import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export type AgentInfo = LibraryAgent;

export function buildAgentInfoMap(agents: AgentInfo[]) {
  const map = new Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >();
  agents.forEach((a) => {
    if (a.graph_id && a.id) {
      map.set(a.graph_id, {
        name:
          a.name || (a.graph_id ? `Agent ${a.graph_id.slice(0, 8)}` : "Agent"),
        description: a.description || "",
        library_agent_id: a.id,
      });
    }
  });
  return map;
}
