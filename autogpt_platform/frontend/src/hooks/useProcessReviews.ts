import {
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsQueryKey,
  usePostV2ProcessReviewAction,
} from "@/app/api/__generated__/endpoints/executions/executions";
import type { ReviewItem } from "@/app/api/__generated__/models/reviewItem";
import { useQueryClient } from "@tanstack/react-query";

export function useProcessReviews({
  onSettled,
}: { onSettled?: () => void } = {}) {
  const queryClient = useQueryClient();
  const { mutateAsync, isPending } = usePostV2ProcessReviewAction();

  async function processReviews(items: ReviewItem[], graphExecIds: string[]) {
    try {
      return await mutateAsync({ data: { reviews: items } });
    } finally {
      // Awaited so callers can keep a row locked until the refetch settles;
      // firing and forgetting leaves React Query serving the just-acted-on
      // review for the whole GET.
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetV2GetPendingReviewsQueryKey(),
        }),
        ...[...new Set(graphExecIds)].map((graphExecId) =>
          queryClient.invalidateQueries({
            queryKey:
              getGetV2GetPendingReviewsForExecutionQueryKey(graphExecId),
          }),
        ),
      ]);
      onSettled?.();
    }
  }

  return { processReviews, isProcessing: isPending };
}
