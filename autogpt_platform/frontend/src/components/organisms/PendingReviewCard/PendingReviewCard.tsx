import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Text } from "@/components/atoms/Text/Text";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useEffect, useState } from "react";
import { JsonDataInput } from "./components/JsonDataInput/JsonDataInput";
import { ReviewInputFields } from "./components/ReviewInputFields/ReviewInputFields";
import { reviewFields } from "./components/ReviewInputFields/helpers";
import { useReviewBlockSchema } from "./components/ReviewInputFields/useReviewBlockSchema";

type ReviewPayload = PendingHumanReviewModel["payload"];

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
  const isDataEditable = review.editable;

  // A block's review names the block in `action`; a human-in-the-loop block's
  // instructions are the author's own words, shown above its data.
  const instructions = review.action ? undefined : review.instructions;
  const { schema, isLoading: isSchemaLoading } = useReviewBlockSchema(
    review.block_id,
  );

  // Render the payload as stored: it is exactly what the reviewer's decision
  // sends back as the block's input, so unwrapping any key here would show
  // one thing and execute another.
  const [currentData, setCurrentData] = useState<ReviewPayload>(review.payload);

  useEffect(() => {
    if (externalDataValue !== undefined) {
      try {
        const parsedData = JSON.parse(externalDataValue);
        setCurrentData(parsedData);
      } catch {}
    }
  }, [externalDataValue]);

  function handleDataChange(newValue: ReviewPayload) {
    setCurrentData(newValue);
    onReviewDataChange(review.node_exec_id, JSON.stringify(newValue, null, 2));
  }

  const renderFields = (readOnly: boolean) => {
    if (!isPlainObject(currentData) || isSchemaLoading) return null;
    return (
      <ReviewInputFields
        fields={reviewFields(currentData, schema)}
        values={currentData}
        onChange={(values) => handleDataChange(values as ReviewPayload)}
        readOnly={readOnly}
      />
    );
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
        <div
          data-slot="pending-review-card-toggle"
          className="flex items-center gap-3"
        >
          <span
            data-slot="pending-review-card-toggle-label"
            // TODO(tokens): no role for Tailwind's default gray-500; the zinc
            // roles are a different grey.
            className="text-sm text-gray-500"
          >
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
        <JsonDataInput
          value={data}
          initialText={externalDataValue}
          onChange={(value) => handleDataChange(value as ReviewPayload)}
          onInvalidDraft={(text) =>
            onReviewDataChange(review.node_exec_id, text)
          }
        />
      );
    }
  };

  const getShortenedNodeId = (id: string) => {
    if (id.length <= 8) return id;
    return `${id.slice(0, 4)}...${id.slice(-4)}`;
  };

  return (
    <div data-slot="pending-review-card" className="space-y-4">
      {nodeId && (
        <Text
          // TODO(tokens): no role for Tailwind's default gray-500; the zinc roles
          // are a different grey.
          variant="small"
          className="text-gray-500"
        >
          Node #{getShortenedNodeId(nodeId)}
        </Text>
      )}

      <div data-slot="pending-review-card-body" className="space-y-3">
        {instructions && (
          <Text
            // TODO(tokens): no role for Tailwind's default gray-900; the zinc
            // roles are a different grey.
            variant="body"
            className="font-semibold text-gray-900"
          >
            {instructions}
          </Text>
        )}

        {isPlainObject(currentData) ? (
          renderFields(!isDataEditable || autoApproveFuture)
        ) : isDataEditable && !autoApproveFuture ? (
          renderDataInput()
        ) : (
          <div
            data-slot="pending-review-card-readonly"
            // TODO(tokens): no role for Tailwind's default gray-200; the zinc
            // roles are a different grey.
            className="rounded-lg border border-gray-200 bg-surface p-3"
          >
            <Text
              // TODO(tokens): no role for Tailwind's default gray-600; the zinc
              // roles are a different grey.
              variant="small"
              className="text-gray-600"
            >
              {JSON.stringify(currentData, null, 2)}
            </Text>
          </div>
        )}
      </div>

      {/* Auto-approve toggle for this review */}
      {showAutoApprove && onAutoApproveFutureChange && (
        <div
          data-slot="pending-review-card-auto-approve"
          className="space-y-2 pt-2"
        >
          <div
            data-slot="pending-review-card-auto-approve-row"
            className="flex items-center gap-3"
          >
            <Switch
              checked={autoApproveFuture}
              onCheckedChange={(enabled: boolean) =>
                onAutoApproveFutureChange(review.node_exec_id, enabled)
              }
            />
            <Text
              // TODO(tokens): no role for Tailwind's default gray-700; the zinc
              // roles are a different grey.
              variant="small"
              className="text-gray-700"
            >
              Auto-approve future executions of this block
            </Text>
          </div>
          {autoApproveFuture && (
            <Text
              // TODO(tokens): no role for Tailwind's default gray-500; the zinc
              // roles are a different grey.
              variant="small"
              className="pl-11 text-gray-500"
            >
              Original data will be used for this and all future reviews from
              this block.
            </Text>
          )}
        </div>
      )}
    </div>
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
