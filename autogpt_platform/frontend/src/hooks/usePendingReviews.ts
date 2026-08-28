import {
  useGetV2GetPendingReviews,
  useGetV2GetPendingReviewsForExecution,
  getGetV2GetPendingReviewsQueryKey,
  getGetV2GetPendingReviewsForExecutionQueryKey,
} from "@/app/api/__generated__/endpoints/executions/executions";
import { okData } from "@/app/api/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

export function usePendingReviews() {
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  const isReady = useOrgTeamStore((state) => state.isLoaded);
  const query = useGetV2GetPendingReviews(undefined, {
    query: {
      enabled: isReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2GetPendingReviewsQueryKey(),
        organizationId,
        teamId,
      ),
    },
    request: getTenantRequestInit(organizationId, teamId, isReady),
  });

  return {
    pendingReviews: okData(query.data) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}

interface UsePendingReviewsForExecutionOptions {
  enabled?: boolean;
  refetchInterval?: number | false;
}

export function usePendingReviewsForExecution(
  graphExecId: string,
  organizationId: string | null,
  teamId: string | null,
  options?: UsePendingReviewsForExecutionOptions,
) {
  const query = useGetV2GetPendingReviewsForExecution(graphExecId, {
    query: {
      enabled: options?.enabled ?? !!graphExecId,
      queryKey: getTeamScopedQueryKey(
        getGetV2GetPendingReviewsForExecutionQueryKey(graphExecId),
        organizationId,
        teamId,
      ),
      refetchInterval: options?.refetchInterval,
      refetchIntervalInBackground: !!options?.refetchInterval,
    },
    request: getTenantRequestInit(organizationId, teamId),
  });

  return {
    pendingReviews: okData(query.data) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}
