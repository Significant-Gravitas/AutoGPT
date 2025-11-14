import {
  useGetV2GetPendingReviews,
  useGetV2GetPendingReviewsForExecution,
} from "@/app/api/__generated__/endpoints/execution-review/execution-review";

/**
 * Hook to fetch pending reviews for the current user
 */
export function usePendingReviews() {
  const query = useGetV2GetPendingReviews();

  return {
    pendingReviews: (query.data?.status === 200 ? query.data.data : []) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}

/**
 * Hook to fetch pending reviews for a specific graph execution
 */
export function usePendingReviewsForExecution(graphExecId: string) {
  const query = useGetV2GetPendingReviewsForExecution(graphExecId);

  return {
    pendingReviews: (query.data?.status === 200 ? query.data.data : []) || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}
