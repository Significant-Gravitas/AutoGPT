import { useMemo, useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewItem } from "@/app/api/__generated__/models/reviewItem";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useProcessReviews } from "@/hooks/useProcessReviews";
import {
  Alert01Icon,
  ArrowDown01Icon,
  ArrowRight01Icon,
  Clock01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { SpendApprovalCard } from "./components/SpendApprovalCard/SpendApprovalCard";
import { spendSummary } from "./components/SpendApprovalCard/helpers";

interface PendingReviewsListProps {
  reviews: PendingHumanReviewModel[];
  onReviewComplete?: () => void;
  emptyMessage?: string;
}

// Copilot action-gate approvals bind exact arguments; auto-approve cannot apply.
export const COPILOT_GATE_NODE_PREFIX = "copilot-node-gate-";
// An expert's spend approval (backend SPEND_REVIEW_MARKER) unlocks her week,
// not a node, so auto-approving its node would do nothing.
const SPEND_REVIEW_MARKER = "expert-spend:";

export function PendingReviewsList({
  reviews,
  onReviewComplete,
  emptyMessage = "No pending reviews",
}: PendingReviewsListProps) {
  const [reviewDataMap, setReviewDataMap] = useState<Record<string, string>>(
    () => {
      const initialData: Record<string, string> = {};
      reviews.forEach((review) => {
        initialData[review.node_exec_id] = JSON.stringify(
          review.payload,
          null,
          2,
        );
      });
      return initialData;
    },
  );

  const [pendingAction, setPendingAction] = useState<{
    nodeId: string;
    action: "approve" | "reject";
  } | null>(null);

  const [autoApproveFutureMap, setAutoApproveFutureMap] = useState<
    Record<string, boolean>
  >({});

  const [collapsedGroups, setCollapsedGroups] = useState<
    Record<string, boolean>
  >({});

  const { toast } = useToast();

  const groupedReviews = useMemo(() => {
    return reviews.reduce(
      (acc, review) => {
        const nodeId = review.node_id || "unknown";
        if (!acc[nodeId]) {
          acc[nodeId] = [];
        }
        acc[nodeId].push(review);
        return acc;
      },
      {} as Record<string, PendingHumanReviewModel[]>,
    );
  }, [reviews]);

  const { processReviews: submitReviewAction, isProcessing } =
    useProcessReviews();
  // Held for a whole group, whose runs are sent one request after another.
  const isBusy = isProcessing || pendingAction !== null;

  function handleReviewDataChange(nodeExecId: string, data: string) {
    setReviewDataMap((prev) => ({ ...prev, [nodeExecId]: data }));
  }

  function handleAutoApproveFutureToggle(nodeId: string, enabled: boolean) {
    setAutoApproveFutureMap((prev) => ({
      ...prev,
      [nodeId]: enabled,
    }));

    if (enabled) {
      const nodeReviews = groupedReviews[nodeId] || [];
      setReviewDataMap((prev) => {
        const updated = { ...prev };
        nodeReviews.forEach((review) => {
          updated[review.node_exec_id] = JSON.stringify(
            review.payload,
            null,
            2,
          );
        });
        return updated;
      });
    }
  }

  function toggleGroupCollapse(nodeId: string) {
    setCollapsedGroups((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  }

  // Scoped to one node's reviews on purpose: these are the rows rendered
  // directly above the button that calls this, and a decision must never
  // reach a row the reviewer has not been shown.
  async function processGroup(nodeId: string, approved: boolean) {
    const groupReviews = groupedReviews[nodeId] || [];

    if (groupReviews.length === 0) {
      toast({
        title: "No reviews to process",
        description: "No reviews found to process.",
        variant: "destructive",
      });
      return;
    }

    setPendingAction({ nodeId, action: approved ? "approve" : "reject" });
    const reviewItems: ReviewItem[] = [];

    for (const review of groupReviews) {
      const reviewData = reviewDataMap[review.node_exec_id];
      const autoApproveThisNode = autoApproveFutureMap[review.node_id || ""];

      let parsedData: any = undefined;

      if (!autoApproveThisNode) {
        // An emptied draft is "" here and must fail the parse below, not
        // fall back to the original payload the reviewer just cleared.
        if (review.editable && reviewData !== undefined) {
          try {
            parsedData = JSON.parse(reviewData);
          } catch (error) {
            // A rejection sends nothing the reviewer edited, so a broken
            // draft only has to block an approval.
            if (approved) {
              toast({
                title: "Invalid JSON",
                description: `Please fix the JSON format in review for node ${review.node_exec_id}: ${error instanceof Error ? error.message : "Invalid syntax"}`,
                variant: "destructive",
              });
              setPendingAction(null);
              return;
            }
            parsedData = review.payload;
          }
        } else {
          parsedData = review.payload;
        }
      }

      reviewItems.push({
        node_exec_id: review.node_exec_id,
        approved,
        reviewed_data: parsedData,
        auto_approve_future: autoApproveThisNode && approved,
      });
    }

    let submitted = false;
    try {
      const result = {
        approved_count: 0,
        rejected_count: 0,
        failed_count: 0,
        errors: [] as string[],
      };
      const failures: string[] = [];
      let failedReviews = 0;
      // A group can span several runs, and the endpoint takes one at a time;
      // a run whose request fails must not keep the others from being sent.
      for (const batch of batchesByScope(reviewItems, groupReviews)) {
        submitted = true;
        try {
          const res = await submitReviewAction(batch.items, batch.reviews);
          if (res.status === 200) {
            result.approved_count += res.data.approved_count;
            result.rejected_count += res.data.rejected_count;
            result.failed_count += res.data.failed_count;
            if (res.data.error) result.errors.push(res.data.error);
            continue;
          }
          failures.push("Unexpected response from server");
        } catch (error) {
          failures.push(
            error instanceof Error ? error.message : "An error occurred",
          );
        }
        failedReviews += batch.items.length;
      }

      const succeeded = result.approved_count + result.rejected_count;
      if (failures.length > 0 && succeeded === 0) {
        toast({
          title: "Failed to process reviews",
          description: [...new Set(failures)].join(" "),
          variant: "destructive",
        });
      } else if (failures.length > 0 || result.failed_count > 0) {
        toast({
          title: "Reviews partially processed",
          description: `${succeeded} succeeded, ${result.failed_count + failedReviews} failed. ${[...result.errors, ...new Set(failures)].join(" ") || "Some reviews could not be processed."}`,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Reviews processed successfully",
          description: `${result.approved_count} approved, ${result.rejected_count} rejected`,
          variant: "default",
        });
      }
    } finally {
      setPendingAction(null);
      if (submitted) onReviewComplete?.();
    }
  }

  if (reviews.length === 0) {
    return (
      <div
        data-slot="pending-reviews-list-empty"
        className="flex flex-col items-center justify-center py-12 text-center"
      >
        <Icon
          icon={Clock01Icon}
          size={48}
          className="mb-4 text-muted-foreground"
        />
        <Text variant="h4" className="text-muted-foreground">
          {emptyMessage}
        </Text>
        <Text variant="body" className="mt-2 max-w-md text-muted-foreground">
          When workflows have human-in-the-loop blocks, they will appear here
          for your review and approval.
        </Text>
      </div>
    );
  }

  const groups = Object.entries(groupedReviews);
  const spendCards = groups.flatMap(([nodeId, nodeReviews]) => {
    const summary = isSpendReview(nodeId) ? spendSummary(nodeReviews) : null;
    return summary ? [{ nodeId, summary }] : [];
  });
  const reviewGroups = groups.filter(
    ([nodeId]) => !spendCards.some((card) => card.nodeId === nodeId),
  );

  return (
    <div data-slot="pending-reviews-list-stack" className="flex flex-col gap-3">
      {spendCards.map(({ nodeId, summary }) => (
        <SpendApprovalCard
          key={nodeId}
          summary={summary}
          pendingAction={
            pendingAction?.nodeId === nodeId ? pendingAction.action : null
          }
          disabled={isBusy}
          onDecide={(approved) => processGroup(nodeId, approved)}
        />
      ))}
      {reviewGroups.length > 0 && (
        <AttentionCard.Root
          variant="caution"
          density="spacious"
          data-slot="pending-reviews-list"
        >
          <div data-slot="pending-reviews-list-header" className="space-y-6">
            <AttentionCard.Row
              align="start"
              data-slot="pending-reviews-list-title"
            >
              <AttentionCard.Icon>
                <Icon icon={Alert01Icon} size={28} />
              </AttentionCard.Icon>
              <Text
                variant="large-semibold"
                className="overflow-hidden text-ellipsis text-textBlack"
              >
                Your review is needed
              </Text>
            </AttentionCard.Row>
            <Text variant="large" className="text-textGrey">
              {reviewGroups.every(([nodeId]) => isGateReview(nodeId))
                ? "Otto is waiting for your approval before the action below."
                : "This workflow is paused until you approve the step below. Check what it will do, and edit it if needed."}
            </Text>
          </div>

          <div data-slot="pending-reviews-list-groups" className="space-y-7">
            {reviewGroups.map(([nodeId, nodeReviews]) => {
              const isCollapsed = collapsedGroups[nodeId] ?? false;
              const reviewCount = nodeReviews.length;
              const groupAction =
                pendingAction?.nodeId === nodeId ? pendingAction.action : null;

              const firstReview = nodeReviews[0];
              const reviewTitle =
                firstReview?.action ?? firstReview?.instructions ?? "Review";
              const workflowName = firstReview?.agent_name;

              const getShortenedNodeId = (id: string) => {
                if (id.length <= 8) return id;
                return `${id.slice(0, 4)}...${id.slice(-4)}`;
              };

              return (
                <div
                  key={nodeId}
                  data-slot="pending-reviews-list-group"
                  className="space-y-4"
                >
                  <AttentionCard.Row asChild align="center">
                    <button
                      onClick={() => toggleGroupCollapse(nodeId)}
                      data-slot="pending-reviews-list-group-toggle"
                      className="w-full text-left"
                    >
                      {isCollapsed ? (
                        <Icon
                          icon={ArrowRight01Icon}
                          size={20}
                          // TODO(tokens): no role for Tailwind's default gray-600;
                          // the zinc roles are a different grey.
                          className="text-gray-600"
                        />
                      ) : (
                        <Icon
                          icon={ArrowDown01Icon}
                          size={20}
                          // TODO(tokens): no role for Tailwind's default gray-600;
                          // the zinc roles are a different grey.
                          className="text-gray-600"
                        />
                      )}
                      <div
                        data-slot="pending-reviews-list-group-heading"
                        className="flex-1"
                      >
                        <Text
                          // TODO(tokens): no role for Tailwind's default gray-900; the
                          // zinc roles are a different grey.
                          variant="body"
                          className="font-semibold text-gray-900"
                        >
                          {reviewTitle}
                        </Text>
                        {(workflowName ||
                          (!firstReview?.action && !isGateReview(nodeId))) && (
                          <Text
                            // TODO(tokens): no role for Tailwind's default gray-500;
                            // the zinc roles are a different grey.
                            variant="small"
                            className="text-gray-500"
                          >
                            {workflowName
                              ? `In workflow “${workflowName}”`
                              : `Node #${getShortenedNodeId(nodeId)}`}
                          </Text>
                        )}
                      </div>
                      <AttentionCard.Count data-slot="pending-reviews-list-group-count">
                        {reviewCount} {reviewCount === 1 ? "review" : "reviews"}
                      </AttentionCard.Count>
                    </button>
                  </AttentionCard.Row>

                  {!isCollapsed && (
                    <div
                      data-slot="pending-reviews-list-group-body"
                      className="space-y-4"
                    >
                      {nodeReviews.map((review) => (
                        <PendingReviewCard
                          key={review.node_exec_id}
                          review={review}
                          onReviewDataChange={handleReviewDataChange}
                          autoApproveFuture={
                            autoApproveFutureMap[nodeId] || false
                          }
                          externalDataValue={reviewDataMap[review.node_exec_id]}
                          showAutoApprove={false}
                        />
                      ))}

                      {!isGateReview(nodeId) && !isSpendReview(nodeId) && (
                        <div
                          data-slot="pending-reviews-list-auto-approve"
                          className="flex items-center gap-3 pt-2"
                        >
                          <Switch
                            checked={autoApproveFutureMap[nodeId] || false}
                            onCheckedChange={(enabled: boolean) =>
                              handleAutoApproveFutureToggle(nodeId, enabled)
                            }
                          />
                          <Text
                            // TODO(tokens): no role for Tailwind's default gray-700;
                            // the zinc roles are a different grey.
                            variant="small"
                            className="text-gray-700"
                          >
                            Auto-approve future executions of this node
                          </Text>
                        </div>
                      )}

                      <AttentionCard.Actions
                        placement="footer"
                        data-slot="pending-reviews-list-actions"
                      >
                        <Button
                          onClick={() => processGroup(nodeId, true)}
                          disabled={isBusy}
                          variant="primary"
                          className="flex min-w-20 items-center justify-center gap-2 rounded-full px-4 py-3"
                          loading={groupAction === "approve"}
                        >
                          {reviewCount === 1
                            ? "Approve"
                            : `Approve ${reviewCount} reviews`}
                        </Button>
                        <Button
                          onClick={() => processGroup(nodeId, false)}
                          disabled={isBusy}
                          variant="destructive"
                          className="flex min-w-20 items-center justify-center gap-2 rounded-full bg-error-hover px-4 py-3"
                          loading={groupAction === "reject"}
                        >
                          {reviewCount === 1
                            ? "Reject"
                            : `Reject ${reviewCount} reviews`}
                        </Button>
                      </AttentionCard.Actions>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <Text variant="small" className="text-textGrey">
            Each decision applies only to the reviews shown above it. You can
            turn auto-approval on or off using the toggle for each node.
          </Text>
        </AttentionCard.Root>
      )}
    </div>
  );
}

function isGateReview(nodeId: string | null | undefined) {
  return !!nodeId?.startsWith(COPILOT_GATE_NODE_PREFIX);
}

function isSpendReview(nodeId: string) {
  return nodeId.includes(SPEND_REVIEW_MARKER);
}

// One request per run or chat, in the order the group shows its reviews.
function batchesByScope(
  items: ReviewItem[],
  reviews: PendingHumanReviewModel[],
) {
  const batches = new Map<
    string,
    { items: ReviewItem[]; reviews: PendingHumanReviewModel[] }
  >();
  reviews.forEach((review, i) => {
    const scope = review.graph_exec_id ?? review.session_id ?? "";
    const batch = batches.get(scope) ?? { items: [], reviews: [] };
    batch.items.push(items[i]);
    batch.reviews.push(review);
    batches.set(scope, batch);
  });
  return [...batches.values()];
}
