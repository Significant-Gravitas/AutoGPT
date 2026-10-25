import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export function filterAgents(agents: LibraryAgent[], term?: string | null) {
  const t = term?.trim().toLowerCase();
  if (!t) return agents;
  return agents.filter(
    (a) =>
      a.name.toLowerCase().includes(t) ||
      a.description.toLowerCase().includes(t),
  );
}
