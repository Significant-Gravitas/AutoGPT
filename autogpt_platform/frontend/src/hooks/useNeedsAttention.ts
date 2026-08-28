import {
  getGetV2GetPendingReviewsQueryKey,
  useGetV2GetPendingReviews,
} from "@/app/api/__generated__/endpoints/executions/executions";
import { okData } from "@/app/api/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

const PAGE_SIZE = 100;

export function useNeedsAttention() {
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  const isReady = useOrgTeamStore((state) => state.isLoaded);
  const params = { page: 1, page_size: PAGE_SIZE };
  const { data, isLoading, isError, refetch } = useGetV2GetPendingReviews(
    params,
    {
      query: {
        select: (res) => okData(res) ?? [],
        // Pending reviews only change on the user's own action or an
        // overnight run, so refresh when the tab comes back into focus
        // rather than polling this (now enriched, multi-query) endpoint
        // every 30s for every open home tab.
        refetchOnWindowFocus: true,
        refetchInterval: 5 * 60_000,
        enabled: isReady,
        queryKey: getTeamScopedQueryKey(
          getGetV2GetPendingReviewsQueryKey(params),
          organizationId,
          teamId,
        ),
      },
      request: getTenantRequestInit(organizationId, teamId, isReady),
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
