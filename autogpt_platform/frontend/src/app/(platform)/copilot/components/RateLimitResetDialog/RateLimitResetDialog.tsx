"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useResetRateLimit } from "../../hooks/useResetRateLimit";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  resetCost: number;
  resetMessage: string;
}

function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

export function RateLimitResetDialog({
  isOpen,
  onClose,
  resetCost,
  resetMessage,
}: Props) {
  const { resetUsage, isPending } = useResetRateLimit(onClose);

  return (
    <Dialog
      title="Usage limit reached"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Text variant="body">{resetMessage}</Text>
          <Text variant="body">
            You can spend{" "}
            <Text variant="body-medium" as="span">
              {formatCents(resetCost)}
            </Text>{" "}
            in credits to reset your daily limit and continue working.
          </Text>
        </div>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose} disabled={isPending}>
            Wait for reset
          </Button>
          <Button
            variant="primary"
            onClick={() => resetUsage()}
            loading={isPending}
          >
            Reset for {formatCents(resetCost)}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
