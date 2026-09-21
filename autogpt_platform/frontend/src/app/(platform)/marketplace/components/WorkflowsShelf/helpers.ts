import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";

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
