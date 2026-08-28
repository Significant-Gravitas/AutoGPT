import {
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsQueryKey,
  postV2ProcessReviewAction,
  type postV2ProcessReviewActionResponse,
} from "@/app/api/__generated__/endpoints/executions/executions";
import type { ReviewItem } from "@/app/api/__generated__/models/reviewItem";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export interface ScopedReviewAction {
  item: ReviewItem;
  graphExecId: string;
  organizationId: string | null;
  teamId: string | null;
}

function scopeKey(action: ScopedReviewAction) {
  return JSON.stringify([action.organizationId, action.teamId]);
}

export function useProcessReviews({
  onSettled,
}: { onSettled?: () => void } = {}) {
  const queryClient = useQueryClient();
  const [isProcessing, setIsProcessing] = useState(false);

  async function processReviews(
    actions: ScopedReviewAction[],
  ): Promise<postV2ProcessReviewActionResponse> {
    if (actions.length === 0) {
      throw new Error("No reviews to process");
    }

    const groups = new Map<string, ScopedReviewAction[]>();
    for (const action of actions) {
      const key = scopeKey(action);
      groups.set(key, [...(groups.get(key) ?? []), action]);
    }

    setIsProcessing(true);
    try {
      const responses = await Promise.all(
        [...groups.values()].map((group) =>
          postV2ProcessReviewAction(
            { reviews: group.map((action) => action.item) },
            getTenantRequestInit(group[0].organizationId, group[0].teamId),
          ),
        ),
      );
      const failedResponse = responses.find(
        (response) => response.status !== 200,
      );
      if (failedResponse) return failedResponse;

      const successful = responses.filter(
        (response) => response.status === 200,
      );
      return {
        status: 200,
        headers: successful[0].headers,
        data: {
          approved_count: successful.reduce(
            (total, response) => total + response.data.approved_count,
            0,
          ),
          rejected_count: successful.reduce(
            (total, response) => total + response.data.rejected_count,
            0,
          ),
          failed_count: successful.reduce(
            (total, response) => total + response.data.failed_count,
            0,
          ),
          error:
            successful
              .map((response) => response.data.error)
              .filter(Boolean)
              .join("; ") || undefined,
        },
      };
    } finally {
      // Awaited so callers can keep a row locked until the refetch settles;
      // firing and forgetting leaves React Query serving the just-acted-on
      // review for the whole GET.
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetV2GetPendingReviewsQueryKey(),
        }),
        ...[
          ...new Map(
            actions.map((action) => [
              JSON.stringify([
                action.graphExecId,
                action.organizationId,
                action.teamId,
              ]),
              action,
            ]),
          ).values(),
        ].map((action) =>
          queryClient.invalidateQueries({
            queryKey: getTeamScopedQueryKey(
              getGetV2GetPendingReviewsForExecutionQueryKey(action.graphExecId),
              action.organizationId,
              action.teamId,
            ),
          }),
        ),
      ]);
      setIsProcessing(false);
      onSettled?.();
    }
  }

  return { processReviews, isProcessing };
}
