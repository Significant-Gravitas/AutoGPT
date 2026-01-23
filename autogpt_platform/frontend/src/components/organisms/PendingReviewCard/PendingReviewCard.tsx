import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Text } from "@/components/atoms/Text/Text";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
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
  autoApproveFuture?: boolean;
  onAutoApproveFutureChange?: (nodeExecId: string, enabled: boolean) => void;
}

export function PendingReviewCard({
  review,
  onReviewDataChange,
  autoApproveFuture = false,
  onAutoApproveFutureChange,
}: PendingReviewCardProps) {
  const extractedData = extractReviewData(review.payload);
  const isDataEditable = review.editable;
  const instructions = extractedData.instructions || review.instructions;
  const [currentData, setCurrentData] = useState(extractedData.data);

  const handleDataChange = (newValue: unknown) => {
    setCurrentData(newValue);
    onReviewDataChange(review.node_exec_id, JSON.stringify(newValue, null, 2));
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

  // Helper function to get proper field label
  const getFieldLabel = (instructions?: string) => {
    if (instructions)
      return instructions.charAt(0).toUpperCase() + instructions.slice(1);
    return "Data to Review";
  };

  // Use the existing HITL review interface
  return (
    <div className="space-y-4">
      {/* Show instructions as field label */}
      {instructions && (
        <div className="space-y-3">
          <Text variant="body" className="font-semibold text-gray-900">
            {getFieldLabel(instructions)}
          </Text>
          {isDataEditable && !autoApproveFuture ? (
            renderDataInput()
          ) : (
            <div className="rounded-lg border border-gray-200 bg-white p-3">
              <Text variant="small" className="text-gray-600">
                {JSON.stringify(currentData, null, 2)}
              </Text>
            </div>
          )}
        </div>
      )}

      {/* If no instructions, show data directly */}
      {!instructions && (
        <div className="space-y-3">
          <Text variant="body" className="font-semibold text-gray-900">
            Data to Review
            {!isDataEditable && (
              <span className="ml-2 text-xs text-muted-foreground">
                (Read-only)
              </span>
            )}
          </Text>
          {isDataEditable && !autoApproveFuture ? (
            renderDataInput()
          ) : (
            <div className="rounded-lg border border-gray-200 bg-white p-3">
              <Text variant="small" className="text-gray-600">
                {JSON.stringify(currentData, null, 2)}
              </Text>
            </div>
          )}
        </div>
      )}

      {/* Auto-approve toggle for this review */}
      {onAutoApproveFutureChange && (
        <div className="space-y-2 pt-2">
          <div className="flex items-center gap-3">
            <Switch
              checked={autoApproveFuture}
              onCheckedChange={(enabled: boolean) =>
                onAutoApproveFutureChange(review.node_exec_id, enabled)
              }
            />
            <Text variant="small" className="text-gray-700">
              Auto-approve future executions of this block
            </Text>
          </div>
          {autoApproveFuture && (
            <Text variant="small" className="pl-11 text-gray-500">
              Original data will be used for this and all future reviews from
              this block.
            </Text>
          )}
        </div>
      )}
    </div>
  );
}
