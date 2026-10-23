"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  planName: string;
  isOpen: boolean;
  isChanging: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ChangePlanDialog({
  planName,
  isOpen,
  isChanging,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <Dialog
      title={`Change plan to ${planName}?`}
      styling={{ maxWidth: "480px" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body" className="text-foreground-secondary">
          Their AutoPilot limits change right away. If they pay through Stripe,
          the next billing sync sets the plan back to what they pay for.
        </Text>
        <Dialog.Footer>
          <Button variant="secondary" size="small" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            loading={isChanging}
            onClick={onConfirm}
          >
            Change plan
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
