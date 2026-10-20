import {
  getGetV2GetPendingReviewsForChatSessionQueryKey,
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsQueryKey,
  usePostV2ProcessReviewAction,
} from "@/app/api/__generated__/endpoints/executions/executions";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewItem } from "@/app/api/__generated__/models/reviewItem";
import { useQueryClient } from "@tanstack/react-query";

type ReviewScope = Pick<
  PendingHumanReviewModel,
  "graph_exec_id" | "session_id"
>;

export function useProcessReviews({
  onSettled,
}: { onSettled?: () => void } = {}) {
  const queryClient = useQueryClient();
  const { mutateAsync, isPending } = usePostV2ProcessReviewAction();

  async function processReviews(items: ReviewItem[], scopes: ReviewScope[]) {
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
        ...scopeQueryKeys(scopes).map((queryKey) =>
          queryClient.invalidateQueries({ queryKey }),
        ),
      ]);
      onSettled?.();
    }
  }

  return { processReviews, isProcessing: isPending };
}

function scopeQueryKeys(scopes: ReviewScope[]) {
  const keys = new Map<string, readonly unknown[]>();
  for (const { graph_exec_id, session_id } of scopes) {
    const key = session_id
      ? getGetV2GetPendingReviewsForChatSessionQueryKey(session_id)
      : graph_exec_id
        ? getGetV2GetPendingReviewsForExecutionQueryKey(graph_exec_id)
        : null;
    if (key) keys.set(JSON.stringify(key), key);
  }
  return [...keys.values()];
}
