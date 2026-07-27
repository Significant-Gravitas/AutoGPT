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
 * (either complete, or stopped at the page cap) — the router waits while it's
 * false rather than resolving a selection against a partial list.
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

  const {
    hasNextPage,
    isFetchingNextPage,
    isFetchNextPageError,
    fetchNextPage,
  } = query;
  const reachedPageCap = (query.data?.pages.length ?? 0) >= MAX_PRESET_PAGES;
  // Stop paginating at the cap, or once a page fetch has errored — otherwise a
  // persistently failing page keeps hasNextPage true and re-fires forever.
  const morePagesPending =
    hasNextPage && !reachedPageCap && !isFetchNextPageError;

  useEffect(() => {
    if (morePagesPending && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [morePagesPending, isFetchingNextPage, fetchNextPage]);

  const presets = useMemo(
    () => (query.data ? unpaginate(query.data, "presets") : undefined),
    [query.data],
  );
  const presetsComplete = query.isSuccess && !hasNextPage;
  const presetsSettled = query.isSuccess && !morePagesPending;

  return {
    presets,
    presetsComplete,
    presetsSettled,
    isSuccess: query.isSuccess,
    isError: query.isError,
    isStale: query.isStale,
    error: query.error,
    refetch: query.refetch,
  };
}
