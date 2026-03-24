"use client";

import {
  usePostV2ResetCopilotUsage,
  getGetV2GetCopilotUsageQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

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
  const queryClient = useQueryClient();
  const { mutate: resetUsage, isPending } = usePostV2ResetCopilotUsage({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2GetCopilotUsageQueryKey(),
        });
        toast({
          title: "Rate limit reset",
          description:
            "Your daily usage limit has been reset. You can continue working.",
        });
        onClose();
      },
      onError: (error: unknown) => {
        const message =
          error instanceof Error ? error.message : "Failed to reset limit.";
        toast({
          title: "Reset failed",
          description: message,
          variant: "destructive",
        });
      },
    },
  });

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
