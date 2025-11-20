import { useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ClockIcon, PlayIcon } from "@phosphor-icons/react";
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
  // State to track data changes for each review
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

  const [reviewMessageMap, setReviewMessageMap] = useState<
    Record<string, string>
  >({});
  const [disabledReviews, setDisabledReviews] = useState<Set<string>>(
    new Set(),
  );

  const { toast } = useToast();

  const reviewActionMutation = usePostV2ProcessReviewAction({
    mutation: {
      onSuccess: (data) => {
        // Check if the response is successful
        if (data.status !== 200) {
          toast({
            title: "Failed to process reviews",
            description: "Unexpected response from server",
            variant: "destructive",
          });
          return;
        }

        const response = data.data;

        if (response.failed_count > 0) {
          // Partial failure
          toast({
            title: "Reviews partially processed",
            description: `${response.approved_count + response.rejected_count} succeeded, ${response.failed_count} failed. ${response.error || "Some reviews could not be processed."}`,
            variant: "destructive",
          });
        } else {
          // Complete success
          toast({
            title: "Reviews processed successfully",
            description: `${response.approved_count} approved, ${response.rejected_count} rejected`,
            variant: "default",
          });
        }

        onReviewComplete?.();
      },
      onError: (error: Error) => {
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

  function handleReviewMessageChange(nodeExecId: string, message: string) {
    setReviewMessageMap((prev) => ({ ...prev, [nodeExecId]: message }));
  }

  function handleToggleDisabled(nodeExecId: string) {
    setDisabledReviews((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(nodeExecId)) {
        newSet.delete(nodeExecId);
      } else {
        newSet.add(nodeExecId);
      }
      return newSet;
    });
  }

  function handleContinue() {
    // Process ALL reviews - enabled as approved, disabled as rejected
    const enabledReviews = reviews.filter(
      (review) => !disabledReviews.has(review.node_exec_id),
    );
    const disabledReviewIds = reviews
      .filter((review) => disabledReviews.has(review.node_exec_id))
      .map((review) => review.node_exec_id);

    if (enabledReviews.length === 0 && disabledReviewIds.length === 0) {
      toast({
        title: "No reviews to process",
        description: "No reviews found to process.",
        variant: "destructive",
      });
      return;
    }

    // Validate JSON for enabled (approved) reviews
    const approvedReviewItems: Array<{
      node_exec_id: string;
      reviewed_data?: any;
      message?: string;
    }> = [];

    for (const review of enabledReviews) {
      const reviewData = reviewDataMap[review.node_exec_id];
      const reviewMessage = reviewMessageMap[review.node_exec_id];

      let parsedData;
      if (review.editable && reviewData) {
        try {
          parsedData = JSON.parse(reviewData);
          // Check if data actually changed
          if (JSON.stringify(parsedData) === JSON.stringify(review.payload)) {
            parsedData = undefined; // No change, don't send reviewed_data
          }
        } catch (error) {
          toast({
            title: "Invalid JSON",
            description: `Please fix the JSON format in review for node ${review.node_exec_id}: ${error instanceof Error ? error.message : "Invalid syntax"}`,
            variant: "destructive",
          });
          return; // Exit early on validation failure
        }
      }

      approvedReviewItems.push({
        node_exec_id: review.node_exec_id,
        reviewed_data: parsedData,
        message: reviewMessage || undefined,
      });
    }

    // Process ALL reviews - enabled as approved, disabled as rejected
    reviewActionMutation.mutate({
      data: {
        approved_reviews: approvedReviewItems,
        rejected_review_ids: disabledReviewIds,
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
    <div className="space-y-6">
      <div className="space-y-4">
        {reviews.map((review) => (
          <PendingReviewCard
            key={review.node_exec_id}
            review={review}
            reviewData={reviewDataMap[review.node_exec_id] || ""}
            onReviewDataChange={handleReviewDataChange}
            reviewMessage={reviewMessageMap[review.node_exec_id] || ""}
            onReviewMessageChange={handleReviewMessageChange}
            isDisabled={disabledReviews.has(review.node_exec_id)}
            onToggleDisabled={handleToggleDisabled}
          />
        ))}
      </div>

      {/* Continue Action */}
      <div className="flex justify-center border-t pt-6">
        <div className="space-y-3 text-center">
          <div>
            <Text variant="small" className="text-muted-foreground">
              {disabledReviews.size > 0 ? (
                <>
                  Approve {reviews.length - disabledReviews.size}, reject{" "}
                  {disabledReviews.size} of {reviews.length} items
                </>
              ) : (
                <>Approve all {reviews.length} items</>
              )}
            </Text>
          </div>
          <Button
            onClick={handleContinue}
            disabled={reviewActionMutation.isPending || reviews.length === 0}
            variant="primary"
            size="large"
            leftIcon={<PlayIcon size={16} />}
          >
            Continue Execution
          </Button>
        </div>
      </div>
    </div>
  );
}
