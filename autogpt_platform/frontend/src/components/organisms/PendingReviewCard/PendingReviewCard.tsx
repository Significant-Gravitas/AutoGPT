import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { TrashIcon, EyeSlashIcon } from "@phosphor-icons/react";
import { useState } from "react";

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

  return { data: payload };
}

interface PendingReviewCardProps {
  review: PendingHumanReviewModel;
  onReviewDataChange: (nodeExecId: string, data: string) => void;
  reviewMessage: string;
  onReviewMessageChange: (nodeExecId: string, message: string) => void;
  isDisabled: boolean;
  onToggleDisabled: (nodeExecId: string) => void;
}

export function PendingReviewCard({
  review,
  onReviewDataChange,
  reviewMessage,
  onReviewMessageChange,
  isDisabled,
  onToggleDisabled,
}: PendingReviewCardProps) {
  const extractedData = extractReviewData(review.payload);
  const isDataEditable = review.editable;
  const instructions = extractedData.instructions || review.instructions;
  const [currentData, setCurrentData] = useState(extractedData.data);

  const handleDataChange = (newValue: unknown) => {
    setCurrentData(newValue);
    onReviewDataChange(review.node_exec_id, JSON.stringify(newValue, null, 2));
  };

  const handleMessageChange = (newMessage: string) => {
    onReviewMessageChange(review.node_exec_id, newMessage);
  };

  const renderDataInput = () => {
    const data = currentData;

    if (typeof data === "string") {
      return (
        <Input
          id="data-string"
          label="Value"
          hideLabel
          size="small"
          type="textarea"
          rows={3}
          value={data}
          onChange={(e) => handleDataChange(e.target.value)}
          placeholder="Enter text"
        />
      );
    } else if (typeof data === "number") {
      return (
        <Input
          id="data-number"
          label="Value"
          hideLabel
          size="small"
          type="number"
          value={data}
          onChange={(e) => handleDataChange(Number(e.target.value))}
          placeholder="Enter number"
        />
      );
    } else if (typeof data === "boolean") {
      return (
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {data ? "Enabled" : "Disabled"}
          </span>
          <Switch
            className="ml-auto"
            checked={data}
            onCheckedChange={(checked: boolean) => handleDataChange(checked)}
          />
        </div>
      );
    } else {
      return (
        <Input
          id="data-json"
          label="Value"
          hideLabel
          size="small"
          type="textarea"
          rows={6}
          value={JSON.stringify(data, null, 2)}
          onChange={(e) => {
            try {
              const parsed = JSON.parse(e.target.value);
              handleDataChange(parsed);
            } catch {}
          }}
          placeholder="Edit JSON data"
          className="font-mono text-sm"
        />
      );
    }
  };

  return (
    <div
      className={`space-y-4 rounded-lg border p-4 ${isDisabled ? "bg-muted/50 opacity-60" : ""}`}
    >
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

      {instructions && (
        <div>
          <Text variant="body" className="mb-2 font-semibold">
            Instructions:
          </Text>
          <Text variant="body">{instructions}</Text>
        </div>
      )}

      <div>
        <Text variant="body" className="mb-2 font-semibold">
          Data to Review:
          {!isDataEditable && (
            <span className="ml-2 text-xs text-muted-foreground">
              (Read-only)
            </span>
          )}
        </Text>
        {isDataEditable && !isDisabled ? (
          renderDataInput()
        ) : (
          <div className="rounded border bg-muted p-3">
            <Text variant="small" className="font-mono text-muted-foreground">
              {JSON.stringify(currentData, null, 2)}
            </Text>
          </div>
        )}
      </div>

      {isDisabled && (
        <div>
          <Text variant="body" className="mb-2 font-semibold">
            Rejection Reason (Optional):
          </Text>
          <Input
            id="rejection-reason"
            label="Rejection Reason"
            hideLabel
            size="small"
            type="textarea"
            rows={3}
            value={reviewMessage}
            onChange={(e) => handleMessageChange(e.target.value)}
            placeholder="Add any notes about why you're rejecting this..."
          />
        </div>
      )}
    </div>
  );
}
