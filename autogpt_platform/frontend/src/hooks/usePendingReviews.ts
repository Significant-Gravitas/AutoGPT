import {
  useGetV2GetPendingReviews,
  useGetV2GetPendingReviewsForExecution,
} from "@/app/api/__generated__/endpoints/executions/executions";
import { okData } from "@/app/api/helpers";

export function usePendingReviews() {
  const query = useGetV2GetPendingReviews();

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
  options?: UsePendingReviewsForExecutionOptions,
) {
  const query = useGetV2GetPendingReviewsForExecution(graphExecId, {
    query: {
      enabled: options?.enabled ?? !!graphExecId,
      refetchInterval: options?.refetchInterval,
      refetchIntervalInBackground: !!options?.refetchInterval,
    },
  });

  return {
    pendingReviews: okData(query.data) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}
