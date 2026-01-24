import { useMemo, useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ClockIcon,
  WarningIcon,
  CaretDownIcon,
  CaretRightIcon,
} from "@phosphor-icons/react";
import { usePostV2ProcessReviewAction } from "@/app/api/__generated__/endpoints/executions/executions";
import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";

interface PendingReviewsListProps {
  reviews: PendingHumanReviewModel[];
  onReviewComplete?: () => void;
  emptyMessage?: string;
}

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

  const [pendingAction, setPendingAction] = useState<
    "approve" | "reject" | null
  >(null);

  // Track per-review auto-approval state
  const [autoApproveFutureMap, setAutoApproveFutureMap] = useState<
    Record<string, boolean>
  >({});

  // Track collapsed state for each group
  const [collapsedGroups, setCollapsedGroups] = useState<
    Record<string, boolean>
  >({});

  const { toast } = useToast();

  // Get the graph_id from the first review (all reviews from same execution)
  const graphId = reviews[0]?.graph_id;

  // Fetch the graph to get node metadata
  const { data: graph } = useGetV1GetSpecificGraph(graphId, undefined, {
    query: {
      enabled: !!graphId,
    },
  });

  // Create a map of node_id -> node display name
  const nodeNameMap = useMemo(() => {
    if (graph?.status !== 200) return {};
    const nodes = graph.data.nodes;
    if (!nodes) return {};

    return nodes.reduce((acc: Record<string, string>, node) => {
      const displayName =
        (node.metadata?.customized_name as string | undefined) ||
        node.block_id ||
        "Unknown Block";
      acc[node.id || ""] = displayName;
      return acc;
    }, {} as Record<string, string>);
  }, [graph]);

  // Group reviews by node_id
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

  const reviewActionMutation = usePostV2ProcessReviewAction({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) {
          toast({
            title: "Failed to process reviews",
            description: "Unexpected response from server",
            variant: "destructive",
          });
          return;
        }

        const result = res.data;

        if (result.failed_count > 0) {
          toast({
            title: "Reviews partially processed",
            description: `${result.approved_count + result.rejected_count} succeeded, ${result.failed_count} failed. ${result.error || "Some reviews could not be processed."}`,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Reviews processed successfully",
            description: `${result.approved_count} approved, ${result.rejected_count} rejected`,
            variant: "default",
          });
        }

        setPendingAction(null);
        onReviewComplete?.();
      },
      onError: (error: Error) => {
        setPendingAction(null);
        toast({
          title: "Failed to process reviews",
          description: error.message || "An error occurred",
          variant: "destructive",
        });
      },
    },
  });

  function handleReviewDataChange(nodeExecId: string, data: string) {
    setReviewDataMap((prev) => ({ ...prev, [nodeExecId]: data }));
  }

  // Handle per-review auto-approval toggle
  function handleAutoApproveFutureToggle(nodeExecId: string, enabled: boolean) {
    setAutoApproveFutureMap((prev) => ({
      ...prev,
      [nodeExecId]: enabled,
    }));

    if (enabled) {
      // Reset this review's data to original value
      const review = reviews.find((r) => r.node_exec_id === nodeExecId);
      if (review) {
        setReviewDataMap((prev) => ({
          ...prev,
          [nodeExecId]: JSON.stringify(review.payload, null, 2),
        }));
      }
    }
  }

  function toggleGroupCollapse(nodeId: string) {
    setCollapsedGroups((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  }

  function processReviews(approved: boolean) {
    if (reviews.length === 0) {
      toast({
        title: "No reviews to process",
        description: "No reviews found to process.",
        variant: "destructive",
      });
      return;
    }

    setPendingAction(approved ? "approve" : "reject");
    const reviewItems = [];

    for (const review of reviews) {
      const reviewData = reviewDataMap[review.node_exec_id];
      const autoApproveThisReview = autoApproveFutureMap[review.node_exec_id];

      // When auto-approving future actions for this review, send undefined (use original data)
      // Otherwise, parse and send the edited data if available
      let parsedData: any = undefined;

      if (!autoApproveThisReview) {
        // For regular approve/reject, use edited data if available
        if (review.editable && reviewData) {
          try {
            parsedData = JSON.parse(reviewData);
          } catch (error) {
            toast({
              title: "Invalid JSON",
              description: `Please fix the JSON format in review for node ${review.node_exec_id}: ${error instanceof Error ? error.message : "Invalid syntax"}`,
              variant: "destructive",
            });
            setPendingAction(null);
            return;
          }
        } else {
          // No edits, use original payload
          parsedData = review.payload;
        }
      }
      // When autoApproveThisReview is true, parsedData stays undefined
      // Backend will use the original payload stored in the database

      reviewItems.push({
        node_exec_id: review.node_exec_id,
        approved,
        reviewed_data: parsedData,
        auto_approve_future: autoApproveThisReview && approved,
      });
    }

    reviewActionMutation.mutate({
      data: {
        reviews: reviewItems,
      },
    });
  }

  if (reviews.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <ClockIcon size={48} className="mb-4 text-muted-foreground" />
        <Text variant="h4" className="text-muted-foreground">
          {emptyMessage}
        </Text>
        <Text variant="body" className="mt-2 max-w-md text-muted-foreground">
          When agents have human-in-the-loop blocks, they will appear here for
          your review and approval.
        </Text>
      </div>
    );
  }

  return (
    <div className="space-y-7 rounded-xl border border-yellow-150 bg-yellow-25 p-6">
      {/* Warning Box Header */}
      <div className="space-y-6">
        <div className="flex items-start gap-2">
          <WarningIcon
            size={28}
            className="fill-yellow-600 text-white"
            weight="fill"
          />
          <Text
            variant="large-semibold"
            className="overflow-hidden text-ellipsis text-textBlack"
          >
            Your review is needed
          </Text>
        </div>
        <Text variant="large" className="text-textGrey">
          This task is paused until you approve the changes below. Please review
          and edit if needed.
        </Text>
      </div>

      <div className="space-y-7">
        {Object.entries(groupedReviews).map(([nodeId, nodeReviews]) => {
          const isCollapsed = collapsedGroups[nodeId];
          const displayName =
            nodeNameMap[nodeId] || nodeReviews[0]?.node_id || "Unknown Block";
          const reviewCount = nodeReviews.length;

          return (
            <div key={nodeId} className="space-y-4">
              {/* Group Header - Only show if there are multiple groups */}
              {Object.keys(groupedReviews).length > 1 && (
                <button
                  onClick={() => toggleGroupCollapse(nodeId)}
                  className="flex w-full items-center gap-2 rounded-lg bg-white p-3 text-left hover:bg-gray-50"
                >
                  {isCollapsed ? (
                    <CaretRightIcon size={20} className="text-gray-600" />
                  ) : (
                    <CaretDownIcon size={20} className="text-gray-600" />
                  )}
                  <Text
                    variant="body"
                    className="flex-1 font-semibold text-gray-900"
                  >
                    {displayName}
                  </Text>
                  <span className="rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-600">
                    {reviewCount} {reviewCount === 1 ? "review" : "reviews"}
                  </span>
                </button>
              )}

              {/* Reviews in this group */}
              {!isCollapsed && (
                <div className="space-y-4">
                  {nodeReviews.map((review) => (
                    <PendingReviewCard
                      key={review.node_exec_id}
                      review={review}
                      onReviewDataChange={handleReviewDataChange}
                      autoApproveFuture={
                        autoApproveFutureMap[review.node_exec_id] || false
                      }
                      onAutoApproveFutureChange={handleAutoApproveFutureToggle}
                      externalDataValue={reviewDataMap[review.node_exec_id]}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => processReviews(true)}
            disabled={reviewActionMutation.isPending || reviews.length === 0}
            variant="primary"
            className="flex min-w-20 items-center justify-center gap-2 rounded-full px-4 py-3"
            loading={
              pendingAction === "approve" && reviewActionMutation.isPending
            }
          >
            Approve
          </Button>
          <Button
            onClick={() => processReviews(false)}
            disabled={reviewActionMutation.isPending || reviews.length === 0}
            variant="destructive"
            className="flex min-w-20 items-center justify-center gap-2 rounded-full bg-red-600 px-4 py-3"
            loading={
              pendingAction === "reject" && reviewActionMutation.isPending
            }
          >
            Reject
          </Button>
        </div>

        <Text variant="small" className="text-textGrey">
          You can turn auto-approval on or off anytime in this agent&apos;s
          settings.
        </Text>
      </div>
    </div>
  );
}
