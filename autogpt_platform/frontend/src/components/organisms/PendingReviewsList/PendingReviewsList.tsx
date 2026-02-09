import { useMemo, useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ClockIcon,
  WarningIcon,
  CaretDownIcon,
  CaretRightIcon,
} from "@phosphor-icons/react";
import { usePostV2ProcessReviewAction } from "@/app/api/__generated__/endpoints/executions/executions";

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
      const autoApproveThisNode = autoApproveFutureMap[review.node_id || ""];

      let parsedData: any = undefined;

      if (!autoApproveThisNode) {
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
          const isCollapsed = collapsedGroups[nodeId] ?? nodeReviews.length > 1;
          const reviewCount = nodeReviews.length;

          const firstReview = nodeReviews[0];
          const blockName = firstReview?.instructions;
          const reviewTitle = `Review required for ${blockName}`;

          const getShortenedNodeId = (id: string) => {
            if (id.length <= 8) return id;
            return `${id.slice(0, 4)}...${id.slice(-4)}`;
          };

          return (
            <div key={nodeId} className="space-y-4">
              <button
                onClick={() => toggleGroupCollapse(nodeId)}
                className="flex w-full items-center gap-2 text-left"
              >
                {isCollapsed ? (
                  <CaretRightIcon size={20} className="text-gray-600" />
                ) : (
                  <CaretDownIcon size={20} className="text-gray-600" />
                )}
                <div className="flex-1">
                  <Text variant="body" className="font-semibold text-gray-900">
                    {reviewTitle}
                  </Text>
                  <Text variant="small" className="text-gray-500">
                    Node #{getShortenedNodeId(nodeId)}
                  </Text>
                </div>
                <span className="text-xs text-gray-600">
                  {reviewCount} {reviewCount === 1 ? "review" : "reviews"}
                </span>
              </button>

              {!isCollapsed && (
                <div className="space-y-4">
                  {nodeReviews.map((review) => (
                    <PendingReviewCard
                      key={review.node_exec_id}
                      review={review}
                      onReviewDataChange={handleReviewDataChange}
                      autoApproveFuture={autoApproveFutureMap[nodeId] || false}
                      externalDataValue={reviewDataMap[review.node_exec_id]}
                      showAutoApprove={false}
                    />
                  ))}

                  <div className="flex items-center gap-3 pt-2">
                    <Switch
                      checked={autoApproveFutureMap[nodeId] || false}
                      onCheckedChange={(enabled: boolean) =>
                        handleAutoApproveFutureToggle(nodeId, enabled)
                      }
                    />
                    <Text variant="small" className="text-gray-700">
                      Auto-approve future executions of this node
                    </Text>
                  </div>
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
          You can turn auto-approval on or off using the toggle above for each
          node.
        </Text>
      </div>
    </div>
  );
}
