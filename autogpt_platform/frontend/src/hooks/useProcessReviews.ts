import {
  getGetV2GetPendingReviewsForChatSessionQueryKey,
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsQueryKey,
  postV2ProcessReviewAction,
  type postV2ProcessReviewActionResponse,
} from "@/app/api/__generated__/endpoints/executions/executions";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewItem } from "@/app/api/__generated__/models/reviewItem";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export interface ScopedReviewAction {
  item: ReviewItem;
  graphExecId: string | null;
  sessionId: string | null;
  organizationId: string | null;
  teamId: string | null;
}

function scopeKey(action: ScopedReviewAction) {
  return JSON.stringify([action.organizationId, action.teamId]);
}

type ReviewScope = Pick<
  PendingHumanReviewModel,
  | "node_exec_id"
  | "graph_exec_id"
  | "session_id"
  | "organization_id"
  | "team_id"
>;

export function useProcessReviews({
  onSettled,
}: { onSettled?: () => void } = {}) {
  const queryClient = useQueryClient();
  const [isProcessing, setIsProcessing] = useState(false);

  async function processReviews(
    items: ReviewItem[],
    scopes: ReviewScope[],
    { waitForOtherCards = false }: { waitForOtherCards?: boolean } = {},
  ): Promise<postV2ProcessReviewActionResponse> {
    const actions = items.map((item): ScopedReviewAction => {
      const scope = scopes.find(
        (scope) => scope.node_exec_id === item.node_exec_id,
      );
      if (!scope) throw new Error("Review scope is missing");
      return {
        item,
        graphExecId: scope.graph_exec_id ?? null,
        sessionId: scope.session_id ?? null,
        organizationId: scope.organization_id ?? null,
        teamId: scope.team_id ?? null,
      };
    });
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
            {
              reviews: group.map((action) => action.item),
              ...(waitForOtherCards ? { wait_for_other_cards: true } : {}),
            },
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
          // A turn follows once any scope's answers start one.
          awaiting_other_cards: successful.every(
            (response) => response.data.awaiting_other_cards === true,
          ),
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
        ...scopeQueryKeys(scopes).map((queryKey) =>
          queryClient.invalidateQueries({ queryKey }),
        ),
      ]);
      setIsProcessing(false);
      onSettled?.();
    }
  }

  return { processReviews, isProcessing };
}

function scopeQueryKeys(scopes: ReviewScope[]) {
  const keys = new Map<string, readonly unknown[]>();
  for (const {
    graph_exec_id,
    session_id,
    organization_id,
    team_id,
  } of scopes) {
    const key = session_id
      ? getGetV2GetPendingReviewsForChatSessionQueryKey(session_id)
      : graph_exec_id
        ? getGetV2GetPendingReviewsForExecutionQueryKey(graph_exec_id)
        : null;
    if (key) {
      const scopedKey = getTeamScopedQueryKey(
        key,
        organization_id ?? null,
        team_id ?? null,
      );
      keys.set(JSON.stringify(scopedKey), scopedKey);
    }
  }
  return [...keys.values()];
}
