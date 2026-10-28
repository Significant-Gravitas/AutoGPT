"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { ReasonFlow } from "./components/ReasonFlow";
import { CHECKOUT_REASONS } from "./helpers";

interface Props {
  initialReason?: string | null;
  initialSent?: boolean;
}

export function CheckoutAbandonedDialog({
  initialReason = null,
  initialSent = false,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <Dialog
      title="No charge was made. What stopped you?"
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Content>
        <NoInitialFocus />
        <div className="flex flex-col gap-4">
          <Text variant="body" className="!text-foreground-tertiary">
            You left checkout before paying. Your plan hasn&apos;t changed.
          </Text>
          <ReasonFlow
            reasons={CHECKOUT_REASONS}
            initialReason={initialReason}
            initialSent={initialSent}
            thanks="Thanks. If your card was the problem, try another one or reply to any of our emails and we'll sort it out."
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
