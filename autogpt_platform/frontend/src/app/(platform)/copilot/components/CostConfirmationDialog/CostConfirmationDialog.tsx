"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

type CostEstimate = {
  resolved_path: "baseline" | "sdk";
  resolved_model: string;
  estimated_llm_calls: number;
  estimated_cost_usd: number;
  confirmation_threshold_usd: number;
  rationale: string;
};

interface Props {
  estimate: CostEstimate | null;
  onConfirm: () => void;
  onCancel: () => void;
}

function formatUsd(amount: number): string {
  return `$${amount.toFixed(2)}`;
}

export function CostConfirmationDialog({
  estimate,
  onConfirm,
  onCancel,
}: Props) {
  const isOpen = estimate !== null;
  if (!estimate) return null;

  return (
    <Dialog
      title="Confirm estimated cost"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Text variant="body">
            This request is estimated to cost{" "}
            <Text variant="body-medium" as="span">
              {formatUsd(estimate.estimated_cost_usd)}
            </Text>
            .
          </Text>
          <Text variant="body-sm" className="text-neutral-600">
            Threshold: {formatUsd(estimate.confirmation_threshold_usd)}
          </Text>
          <Text variant="body-sm" className="text-neutral-600">
            Path: {estimate.resolved_path} - Model: {estimate.resolved_model}
          </Text>
          <Text variant="body-sm" className="text-neutral-600">
            Estimated model calls: {estimate.estimated_llm_calls}
          </Text>
          <Text variant="body-sm" className="text-neutral-600">
            {estimate.rationale}
          </Text>
          <Text variant="body-xs" className="text-neutral-500">
            This is an estimate. Actual usage may vary based on tool calls and
            context.
          </Text>
        </div>
        <Dialog.Footer className="!justify-center">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm}>
            Continue
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
