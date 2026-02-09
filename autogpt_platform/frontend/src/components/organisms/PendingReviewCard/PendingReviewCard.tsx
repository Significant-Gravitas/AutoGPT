import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Text } from "@/components/atoms/Text/Text";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useEffect, useState } from "react";

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
  externalDataValue?: string;
  showAutoApprove?: boolean;
  nodeId?: string;
}

export function PendingReviewCard({
  review,
  onReviewDataChange,
  autoApproveFuture = false,
  onAutoApproveFutureChange,
  externalDataValue,
  showAutoApprove = true,
  nodeId,
}: PendingReviewCardProps) {
  const extractedData = extractReviewData(review.payload);
  const isDataEditable = review.editable;

  let instructions = review.instructions;

  const isHITLBlock = instructions && !instructions.includes("Block");

  if (instructions && !isHITLBlock) {
    instructions = undefined;
  }

  const [currentData, setCurrentData] = useState(extractedData.data);

  useEffect(() => {
    if (externalDataValue !== undefined) {
      try {
        const parsedData = JSON.parse(externalDataValue);
        setCurrentData(parsedData);
      } catch {}
    }
  }, [externalDataValue]);

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

  const getShortenedNodeId = (id: string) => {
    if (id.length <= 8) return id;
    return `${id.slice(0, 4)}...${id.slice(-4)}`;
  };

  return (
    <div className="space-y-4">
      {nodeId && (
        <Text variant="small" className="text-gray-500">
          Node #{getShortenedNodeId(nodeId)}
        </Text>
      )}

      <div className="space-y-3">
        {instructions && (
          <Text variant="body" className="font-semibold text-gray-900">
            {instructions}
          </Text>
        )}

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

      {/* Auto-approve toggle for this review */}
      {showAutoApprove && onAutoApproveFutureChange && (
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
