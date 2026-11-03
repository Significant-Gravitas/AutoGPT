import type { GetV2ListLibraryAgentsInfiniteQueryResult } from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, okData } from "@/app/api/helpers";

// Preserve the 2000-workflow metadata budget from the library pagination.
const MAX_PAGES = 20;

export function getNextLibraryPage(
  lastPage: GetV2ListLibraryAgentsInfiniteQueryResult,
  pages: GetV2ListLibraryAgentsInfiniteQueryResult[],
  lastPageParam: number | undefined,
) {
  const last = okData(lastPage);
  const first = okData(pages[0]);
  if (!last || !first || !last.agents.length || pages.length >= MAX_PAGES)
    return undefined;

  const pagination = last.pagination;
  if (
    !Number.isSafeInteger(pagination.current_page) ||
    pagination.current_page !== (lastPageParam ?? 1) ||
    !Number.isSafeInteger(pagination.page_size) ||
    pagination.page_size <= 0 ||
    !Number.isSafeInteger(pagination.total_items) ||
    pagination.total_items < 0 ||
    !Number.isSafeInteger(first.pagination.total_pages) ||
    first.pagination.total_pages < 1
  ) {
    return undefined;
  }

  if (pages.length > 1) {
    const previousIDs = new Set(
      pages
        .slice(0, -1)
        .flatMap((page) => okData(page)?.agents.map((agent) => agent.id) ?? []),
    );
    if (last.agents.every((agent) => previousIDs.has(agent.id)))
      return undefined;
  }

  const nextPage = getPaginationNextPageNumber(lastPage);
  // Bound the scan to its initial size even if workflows are added as it runs.
  return nextPage && nextPage <= first.pagination.total_pages
    ? nextPage
    : undefined;
}
