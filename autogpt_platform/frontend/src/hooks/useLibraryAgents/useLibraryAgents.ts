import { useGetV2ListLibraryAgentsInfinite } from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import { useEffect, useMemo } from "react";
import { buildAgentInfoMap } from "./store";

const PAGE_SIZE = 100;

// The Library briefing counts and the activity dropdown's agent names both
// need the whole library, so every page is fetched, up to 20 * 100 = 2000.
const MAX_PAGES = 20;

export function useLibraryAgents() {
  const {
    data: agentsQueryData,
    isLoading: isRefreshing,
    hasNextPage,
    isFetching,
    isFetchNextPageError,
    fetchNextPage,
  } = useGetV2ListLibraryAgentsInfinite(
    {
      page: 1,
      page_size: PAGE_SIZE,
      is_hidden: false,
    },
    {
      query: {
        getNextPageParam: getPaginationNextPageNumber,
        // Don't block rendering - fetch in background
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 5 * 60 * 1000, // 5 minutes
      },
    },
  );

  const loadedPages = agentsQueryData?.pages.length ?? 0;
  // A failed page keeps hasNextPage true, so stop on the error instead of
  // re-requesting it forever.
  const morePagesPending =
    hasNextPage && loadedPages < MAX_PAGES && !isFetchNextPageError;

  useEffect(() => {
    // isFetching (not only isFetchingNextPage) so this never races the first
    // load or a background refetch. loadedPages is a dependency so every page
    // that lands requests the next one, even when React batches away the
    // in-between isFetching render. Several components mount this hook;
    // cancelRefetch: false makes a second caller join the in-flight request
    // instead of cancelling and re-sending it.
    if (morePagesPending && !isFetching) {
      void fetchNextPage({ cancelRefetch: false });
    }
  }, [morePagesPending, isFetching, loadedPages, fetchNextPage]);

  // Stable between renders, so consumers keyed on it (status maps, the
  // execution event subscription) only recompute when the data changes.
  const agents = useMemo(
    () => (agentsQueryData ? unpaginate(agentsQueryData, "agents") : []),
    [agentsQueryData],
  );

  // Use agents.length as dependency to avoid recreating map unnecessarily
  const agentInfoMap = useMemo(
    () => buildAgentInfoMap(agents),
    [agents.length, agents.map((a) => a.id).join(",")],
  );

  return { agents, agentInfoMap, isRefreshing, lastUpdatedAt: undefined };
}
