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

export function usePendingReviewsForExecution(graphExecId: string) {
  const query = useGetV2GetPendingReviewsForExecution(graphExecId);

  return {
    pendingReviews: okData(query.data) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}
