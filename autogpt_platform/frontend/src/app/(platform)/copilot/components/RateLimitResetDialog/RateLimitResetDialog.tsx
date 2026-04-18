"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { useResetRateLimit } from "../../hooks/useResetRateLimit";
import { formatCents } from "../usageHelpers";

export { formatCents };

interface Props {
  isOpen: boolean;
  onClose: () => void;
  resetCost: number;
  resetMessage: string;
  isWeeklyExhausted?: boolean;
  hasInsufficientCredits?: boolean;
  isBillingEnabled?: boolean;
  onCreditChange?: () => void;
}

export function RateLimitResetDialog({
  isOpen,
  onClose,
  resetCost,
  resetMessage,
  isWeeklyExhausted = false,
  hasInsufficientCredits = false,
  isBillingEnabled = false,
  onCreditChange,
}: Props) {
  const { resetUsage, isPending } = useResetRateLimit({
    onSuccess: onClose,
    onCreditChange,
  });
  const router = useRouter();

  // Stable ref for the callback so the effect only re-fires when
  // `isOpen` changes, not when the function reference changes.
  const onCreditChangeRef = useRef(onCreditChange);
  onCreditChangeRef.current = onCreditChange;

  // Refresh the credit balance each time the dialog opens so we never
  // block a valid reset due to a stale client-side balance.
  useEffect(() => {
    if (isOpen) onCreditChangeRef.current?.();
  }, [isOpen]);

  // Whether to hide the reset button entirely
  const cannotReset = isWeeklyExhausted || hasInsufficientCredits;

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
          {isWeeklyExhausted ? (
            <Text variant="body">
              Your weekly limit is also reached, so resetting the daily limit
              won&apos;t help. Please wait for your limits to reset.
            </Text>
          ) : hasInsufficientCredits ? (
            <Text variant="body">
              You don&apos;t have enough credits to reset your daily limit.
              {isBillingEnabled
                ? " Add credits to continue working."
                : " Please wait for your limits to reset."}
            </Text>
          ) : (
            <Text variant="body">
              You can spend{" "}
              <Text variant="body-medium" as="span">
                {formatCents(resetCost)}
              </Text>{" "}
              in credits to reset your daily limit and continue working.
            </Text>
          )}
        </div>
        <Dialog.Footer className="!justify-center">
          <Button variant="secondary" onClick={onClose} disabled={isPending}>
            {cannotReset ? "OK" : "Wait for reset"}
          </Button>
          {hasInsufficientCredits && isBillingEnabled && (
            <Button
              variant="primary"
              onClick={() => {
                onClose();
                router.push("/profile/credits");
              }}
            >
              Add credits
            </Button>
          )}
          {!cannotReset && (
            <Button
              variant="primary"
              onClick={() => resetUsage()}
              loading={isPending}
            >
              Reset for {formatCents(resetCost)}
            </Button>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
