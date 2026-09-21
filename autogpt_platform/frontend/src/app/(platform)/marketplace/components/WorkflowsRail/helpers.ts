import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

export const RAIL_SIZE = 12;

export function featuredFirst(
  agents: StoreAgent[],
  featured: StoreAgent[],
): StoreAgent[] {
  const featuredSlugs = new Set(featured.map((agent) => agent.slug));
  return [
    ...featured,
    ...agents.filter((agent) => !featuredSlugs.has(agent.slug)),
  ];
}
