import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

export function featuredFirst(
  agents: StoreAgent[],
  featured: StoreAgent[],
): StoreAgent[] {
  const featuredIDs = new Set(featured.map((agent) => agent.agent_graph_id));
  return [
    ...featured,
    ...agents.filter((agent) => !featuredIDs.has(agent.agent_graph_id)),
  ];
}
