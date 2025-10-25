import { getV2ListLibraryAgentsResponse } from "@/app/api/__generated__/endpoints/library/library";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";

export function filterAgents(agents: LibraryAgent[], term?: string | null) {
  const t = term?.trim().toLowerCase();
  if (!t) return agents;
  return agents.filter(
    (a) =>
      a.name.toLowerCase().includes(t) ||
      a.description.toLowerCase().includes(t),
  );
}

export function getInitialData(
  cachedAgents: LibraryAgent[],
  searchTerm: string | null,
  pageSize: number,
) {
  const filtered = filterAgents(
    cachedAgents as unknown as LibraryAgent[],
    searchTerm,
  );

  if (!filtered.length) {
    return undefined;
  }

  const firstPageAgents: LibraryAgent[] = filtered.slice(0, pageSize);
  const totalItems = filtered.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  const firstPage: getV2ListLibraryAgentsResponse = {
    status: 200,
    data: {
      agents: firstPageAgents,
      pagination: {
        total_items: totalItems,
        total_pages: totalPages,
        current_page: 1,
        page_size: pageSize,
      },
    } satisfies LibraryAgentResponse,
    headers: new Headers(),
  };

  return { pageParams: [1], pages: [firstPage] };
}
