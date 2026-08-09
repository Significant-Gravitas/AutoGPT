import { useGetV2GetPendingReviews } from "@/app/api/__generated__/endpoints/executions/executions";
import { okData } from "@/app/api/helpers";

const PAGE_SIZE = 100;

export function useNeedsAttention() {
  const { data, isLoading, isError, refetch } = useGetV2GetPendingReviews(
    { page: 1, page_size: PAGE_SIZE },
    {
      query: {
        select: (res) => okData(res) ?? [],
        // Pending reviews only change on the user's own action or an
        // overnight run, so refresh when the tab comes back into focus
        // rather than polling this (now enriched, multi-query) endpoint
        // every 30s for every open home tab.
        refetchOnWindowFocus: true,
        refetchInterval: 5 * 60_000,
      },
    },
  );
  const reviews = data ?? [];
  return {
    reviews,
    // A full page means there may be more behind it; don't present a
    // truncated count as if it were the total.
    countLabel:
      reviews.length >= PAGE_SIZE ? `${PAGE_SIZE}+` : String(reviews.length),
    isLoading,
    isError,
    refetch,
  };
}
