import { useState } from "react";
import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { ReviewActionRequest } from "@/app/api/__generated__/models/reviewActionRequest";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Card } from "@/components/atoms/Card/Card";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CheckIcon, XIcon } from "@phosphor-icons/react";
import { usePostV2ReviewData } from "@/app/api/__generated__/endpoints/execution-review/execution-review";

// Type guard for structured review payload
interface StructuredReviewPayload {
  data: unknown;
  instructions?: string;
}

function isStructuredReviewPayload(
  payload: unknown,
): payload is StructuredReviewPayload {
  return (
    payload !== null &&
    typeof payload === "object" &&
    "data" in payload &&
    (typeof (payload as any).instructions === "string" ||
      (payload as any).instructions === undefined)
  );
}

function extractReviewData(payload: unknown): {
  data: unknown;
  instructions?: string;
} {
  if (isStructuredReviewPayload(payload)) {
    return {
      data: payload.data,
      instructions: payload.instructions,
    };
  }

  // Fallback: treat entire payload as data
  return { data: payload };
}

interface PendingReviewCardProps {
  review: PendingHumanReviewModel;
  onReviewComplete?: () => void;
}

export function PendingReviewCard({
  review,
  onReviewComplete,
}: PendingReviewCardProps) {
  // Extract structured data and instructions from payload
  const extractedData = extractReviewData(review.payload);

  const [reviewData, setReviewData] = useState<string>(
    JSON.stringify(extractedData.data, null, 2),
  );
  const [reviewMessage, setReviewMessage] = useState<string>("");
  const isDataEditable = review.editable;
  const { toast } = useToast();

  // Use instructions from payload or from API field
  const instructions = extractedData.instructions || review.instructions;

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

    // Only parse data if it's editable
    if (isDataEditable) {
      try {
        parsedData = JSON.parse(reviewData);
      } catch (error) {
        toast({
          title: "Invalid JSON",
          description: `Please fix the JSON format: ${error instanceof Error ? error.message : "Invalid syntax"}`,
          variant: "destructive",
        });
        return;
      }

      // Validate the parsed data has reasonable size constraints
      const dataString = JSON.stringify(parsedData);
      if (dataString.length > 1_000_000) {
        // 1MB limit
        toast({
          title: "Data too large",
          description: "Review data cannot exceed 1MB in size",
          variant: "destructive",
        });
        return;
      }
    }

    const requestData: ReviewActionRequest = {
      action: "approve",
      reviewed_data: isDataEditable ? parsedData : undefined,
      message: reviewMessage || undefined,
    };

    reviewActionMutation.mutate({
      nodeExecId: review.node_exec_id,
      data: requestData,
    });
  }

  function handleReject() {
    const requestData: ReviewActionRequest = {
      action: "reject",
      message: reviewMessage || "Rejected by user",
    };

    reviewActionMutation.mutate({
      nodeExecId: review.node_exec_id,
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
        {/* Review Instructions */}
        {instructions && (
          <div>
            <Text variant="body" className="mb-2 font-semibold">
              Instructions:
            </Text>
            <Text variant="body">{instructions}</Text>
          </div>
        )}

        {/* Data Editor */}
        <div>
          <Text variant="body" className="mb-2 font-semibold">
            Data to Review:
            {!isDataEditable && (
              <span className="ml-2 text-xs text-muted-foreground">
                (Read-only)
              </span>
            )}
          </Text>
          <Textarea
            value={reviewData}
            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
              setReviewData(e.target.value)
            }
            placeholder={
              isDataEditable
                ? "Edit the JSON data..."
                : "Data is read-only - you can approve or reject without changes"
            }
            className="min-h-[200px] font-mono text-sm"
            disabled={reviewActionMutation.isPending || !isDataEditable}
            readOnly={!isDataEditable}
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
            maxLength={2000}
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
