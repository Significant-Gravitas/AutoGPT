import { useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ClockIcon, WarningIcon } from "@phosphor-icons/react";
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

  const [reviewMessageMap, setReviewMessageMap] = useState<
    Record<string, string>
  >({});

  const [pendingAction, setPendingAction] = useState<
    "approve" | "reject" | null
  >(null);

  const { toast } = useToast();

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

  function handleReviewMessageChange(nodeExecId: string, message: string) {
    setReviewMessageMap((prev) => ({ ...prev, [nodeExecId]: message }));
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
      const reviewMessage = reviewMessageMap[review.node_exec_id];

      let parsedData: any = review.payload; // Default to original payload

      // Parse edited data if available and editable
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
      }

      reviewItems.push({
        node_exec_id: review.node_exec_id,
        approved,
        reviewed_data: parsedData,
        message: reviewMessage || undefined,
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
        {reviews.map((review) => (
          <PendingReviewCard
            key={review.node_exec_id}
            review={review}
            onReviewDataChange={handleReviewDataChange}
            onReviewMessageChange={handleReviewMessageChange}
            reviewMessage={reviewMessageMap[review.node_exec_id] || ""}
          />
        ))}
      </div>

      <div className="space-y-7">
        <Text variant="body" className="text-textGrey">
          Note: Changes you make here apply only to this task
        </Text>

        <div className="flex gap-2">
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
      </div>
    </div>
  );
}
