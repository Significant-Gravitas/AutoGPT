import { useState } from "react";
import { PendingHumanReviewResponse } from "@/app/api/__generated__/models/pendingHumanReviewResponse";
import { ReviewActionRequest } from "@/app/api/__generated__/models/reviewActionRequest";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Card } from "@/components/atoms/Card/Card";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CheckIcon, XIcon } from "@phosphor-icons/react";
import { usePostV2ReviewData } from "@/app/api/__generated__/endpoints/execution-review/execution-review";

interface PendingReviewCardProps {
  review: PendingHumanReviewResponse;
  onReviewComplete?: () => void;
}

interface ReviewDataStructure {
  data?: unknown;
  message?: string;
  editable?: boolean;
}

function isReviewDataStructure(data: unknown): data is ReviewDataStructure {
  return typeof data === "object" && data !== null;
}

function extractDataFromReview(reviewData: unknown): string {
  if (isReviewDataStructure(reviewData) && "data" in reviewData) {
    return JSON.stringify(reviewData.data, null, 2);
  }
  return JSON.stringify(reviewData, null, 2);
}

function extractMessageFromReview(reviewData: unknown): string | null {
  if (
    isReviewDataStructure(reviewData) &&
    typeof reviewData.message === "string"
  ) {
    return reviewData.message;
  }
  return null;
}

export function PendingReviewCard({
  review,
  onReviewComplete,
}: PendingReviewCardProps) {
  const [reviewData, setReviewData] = useState<string>(
    extractDataFromReview(review.data),
  );
  const [reviewMessage, setReviewMessage] = useState<string>("");
  const { toast } = useToast();

  const reviewActionMutation = usePostV2ReviewData({
    mutation: {
      onSuccess: () => {
        toast({
          title: "Review submitted successfully",
          variant: "default",
        });
        onReviewComplete?.();
      },
      onError: (error: Error) => {
        toast({
          title: "Failed to submit review",
          description: error.message || "An error occurred",
          variant: "destructive",
        });
      },
    },
  });

  function handleApprove() {
    let parsedData;
    try {
      parsedData = JSON.parse(reviewData);
    } catch (_error) {
      toast({
        title: "Invalid JSON",
        description: "Please fix the JSON format before approving",
        variant: "destructive",
      });
      return;
    }

    const requestData: ReviewActionRequest = {
      action: "approve",
      reviewed_data: parsedData,
      message: reviewMessage || undefined,
    };

    reviewActionMutation.mutate({
      reviewId: review.id,
      data: requestData,
    });
  }

  function handleReject() {
    const requestData: ReviewActionRequest = {
      action: "reject",
      message: reviewMessage || "Rejected by user",
    };

    reviewActionMutation.mutate({
      reviewId: review.id,
      data: requestData,
    });
  }

  return (
    <Card className="w-full">
      <div className="space-y-4">
        <div className="flex items-center justify-between border-b pb-4">
          <Text variant="h3">Pending Review</Text>
          <Text variant="small" className="text-muted-foreground">
            {new Date(review.created_at).toLocaleString()}
          </Text>
        </div>
        {/* Review Message */}
        {(() => {
          const message = extractMessageFromReview(review.data);
          return message ? (
            <div>
              <Text variant="body" className="mb-2 font-semibold">
                Instructions:
              </Text>
              <Text variant="body">{message}</Text>
            </div>
          ) : null;
        })()}

        {/* Data Editor */}
        <div>
          <Text variant="body" className="mb-2 font-semibold">
            Data to Review:
          </Text>
          <Textarea
            value={reviewData}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
              setReviewData(e.target.value)
            }
            placeholder="Edit the JSON data..."
            className="min-h-[200px] font-mono text-sm"
            disabled={reviewActionMutation.isPending}
          />
        </div>

        {/* Review Message */}
        <div>
          <Text variant="body" className="mb-2 font-semibold">
            Review Notes (Optional):
          </Text>
          <Textarea
            value={reviewMessage}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
              setReviewMessage(e.target.value)
            }
            placeholder="Add any notes about your review..."
            className="min-h-[100px]"
            disabled={reviewActionMutation.isPending}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <Button
            onClick={handleApprove}
            disabled={reviewActionMutation.isPending}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
          >
            <CheckIcon size={16} />
            Approve
          </Button>
          <Button
            onClick={handleReject}
            variant="destructive"
            disabled={reviewActionMutation.isPending}
            className="flex items-center gap-2"
          >
            <XIcon size={16} />
            Reject
          </Button>
        </div>

        {/* Graph and Node Info */}
        <div className="mt-4 border-t pt-4">
          <Text variant="small" className="text-muted-foreground">
            Graph ID: {review.graph_id} • Node: {review.node_exec_id}
          </Text>
        </div>
      </div>
    </Card>
  );
}
