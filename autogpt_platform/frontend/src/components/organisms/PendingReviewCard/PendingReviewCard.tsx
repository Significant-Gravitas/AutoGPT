import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { TrashIcon, EyeSlashIcon } from "@phosphor-icons/react";

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
  reviewData: string;
  onReviewDataChange: (nodeExecId: string, data: string) => void;
  reviewMessage: string;
  onReviewMessageChange: (nodeExecId: string, message: string) => void;
  isDisabled: boolean;
  onToggleDisabled: (nodeExecId: string) => void;
}

export function PendingReviewCard({
  review,
  reviewData,
  onReviewDataChange,
  reviewMessage,
  onReviewMessageChange,
  isDisabled,
  onToggleDisabled,
}: PendingReviewCardProps) {
  // Extract structured data and instructions from payload
  const extractedData = extractReviewData(review.payload);
  const isDataEditable = review.editable;

  // Use instructions from payload or from API field
  const instructions = extractedData.instructions || review.instructions;

  return (
    <div
      className={`space-y-4 rounded-lg border p-4 ${isDisabled ? "bg-muted/50 opacity-60" : ""}`}
    >
      {/* Header with title and disable toggle */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {isDisabled && (
            <Text variant="small" className="text-muted-foreground">
              This item will be rejected
            </Text>
          )}
        </div>
        <Button
          onClick={() => onToggleDisabled(review.node_exec_id)}
          variant={isDisabled ? "primary" : "secondary"}
          size="small"
          leftIcon={
            isDisabled ? <EyeSlashIcon size={14} /> : <TrashIcon size={14} />
          }
        >
          {isDisabled ? "Include" : "Exclude"}
        </Button>
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
            onReviewDataChange(review.node_exec_id, e.target.value)
          }
          placeholder={
            isDataEditable
              ? "Edit the JSON data..."
              : "Data is read-only - you can approve or reject without changes"
          }
          className="min-h-[200px] font-mono text-sm"
          disabled={!isDataEditable || isDisabled}
          readOnly={!isDataEditable || isDisabled}
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
            onReviewMessageChange(review.node_exec_id, e.target.value)
          }
          placeholder="Add any notes about your review..."
          className="min-h-[100px]"
          maxLength={2000}
          disabled={isDisabled}
        />
      </div>
    </div>
  );
}
