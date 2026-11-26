"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { ReasonFlow } from "./components/ReasonFlow";
import { CARD_FAILED_VALUE, UPGRADE_CHECKOUT_REASONS } from "./helpers";

const TITLE = "Your plan hasn't changed. What held you back?";

interface Props {
  initialReason?: string | null;
  isRetrying?: boolean;
  onDismiss?: () => void;
  onReason?: (value: string) => void;
  onSubmit?: (answer: { reason: string; note: string }) => void;
  /** Absent when there is no checkout to start again. */
  onTryAnotherCard?: () => void;
}

export function CheckoutAbandonedDialog({
  initialReason = null,
  isRetrying = false,
  onDismiss,
  onReason,
  onSubmit,
  onTryAnotherCard,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [sentReason, setSentReason] = useState<string | null>(null);
  const offersRetry = sentReason === CARD_FAILED_VALUE && onTryAnotherCard;

  function send(answer: { reason: string; note: string }) {
    setSentReason(answer.reason);
    onSubmit?.(answer);
  }

  return (
    <Dialog
      title={TITLE}
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: (open) => {
          setIsOpen(open);
          if (!open) onDismiss?.();
        },
      }}
    >
      <Dialog.Content>
        <NoInitialFocus />
        <div className="flex flex-col gap-4">
          <Text variant="body" tone="muted">
            One tap helps us fix the right thing.
          </Text>
          <ReasonFlow
            label={TITLE}
            reasons={UPGRADE_CHECKOUT_REASONS}
            initialReason={initialReason}
            onReason={onReason}
            onSubmit={send}
          />
        </div>
        {offersRetry ? (
          <Dialog.Footer>
            <Button
              size="small"
              loading={isRetrying}
              disabled={isRetrying}
              onClick={onTryAnotherCard}
            >
              Try another card
            </Button>
          </Dialog.Footer>
        ) : null}
      </Dialog.Content>
    </Dialog>
  );
}
