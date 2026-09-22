import { useEffect, useMemo } from "react";

import { useGetV2ListPresetsInfinite } from "@/app/api/__generated__/endpoints/presets/presets";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { retryUnlessClientError } from "../helpers";

// Per-request page size. The hook eagerly pages through every preset for the
// agent (up to MAX_PRESET_PAGES) so membership routing has the complete set at
// mount — see deriveSelectedTriggerKind.
export const PRESETS_PAGE_SIZE = 100;

// Safety bound on eager pagination: 20 * 100 = 2000 presets per agent. Beyond
// it we stop paginating and the graceful by-ID fallback in
// deriveSelectedTriggerKind takes over for unknown IDs.
const MAX_PRESET_PAGES = 20;

/**
 * The agent's presets (webhook triggers + templates), paged into a single
 * list. Shared by the sidebar and the detail-pane router so both read one
 * React Query cache entry.
 *
 * `presetsComplete` is true once every page has loaded, so membership is
 * authoritative. `presetsSettled` is true once no more pages will be fetched
 * (complete, stopped at the page cap, or a later page errored) — the router
 * waits while it's false rather than resolving against a partial list. A
 * later-page failure only degrades gracefully; `isError` flags a hard failure
 * (the first page failed, so there's no data at all).
 */
export function useAgentPresetsQuery(graphId: string | undefined) {
  const query = useGetV2ListPresetsInfinite(
    { graph_id: graphId ?? "", page: 1, page_size: PRESETS_PAGE_SIZE },
    {
      query: {
        enabled: !!graphId,
        getNextPageParam: getPaginationNextPageNumber,
        retry: retryUnlessClientError,
      },
    },
  );

  const { hasNextPage, isFetching, isFetchNextPageError, fetchNextPage } =
    query;
  const reachedPageCap = (query.data?.pages.length ?? 0) >= MAX_PRESET_PAGES;
  // Stop paginating at the cap, or once a page fetch has errored — otherwise a
  // persistently failing page keeps hasNextPage true and re-fires forever.
  const morePagesPending =
    hasNextPage && !reachedPageCap && !isFetchNextPageError;

  useEffect(() => {
    // Guard on isFetching (not just isFetchingNextPage) so this never races an
    // initial load or background refetch.
    if (morePagesPending && !isFetching) {
      fetchNextPage();
    }
  }, [morePagesPending, isFetching, fetchNextPage]);

  const presets = useMemo(
    () => (query.data ? unpaginate(query.data, "presets") : undefined),
    [query.data],
  );
  const presetsComplete = query.isSuccess && !hasNextPage;
  // A later-page failure flips the query to error while keeping the earlier
  // pages, so settle on it too — routing then resolves against the partial
  // list and the by-ID fallback instead of stalling on "loading".
  const presetsSettled =
    !morePagesPending && (query.isSuccess || isFetchNextPageError);
  // Only a first-page failure (no data at all) is a hard error worth surfacing;
  // a mid-pagination failure leaves partial data the UI can still use.
  const hasHardFailure = query.isError && presets === undefined;

  return {
    presets,
    presetsComplete,
    presetsSettled,
    isError: hasHardFailure,
    isStale: query.isStale,
    error: hasHardFailure ? query.error : null,
    refetch: query.refetch,
  };
}
