"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useState } from "react";
import { NoInitialFocus } from "./components/NoInitialFocus";
import { ReasonFlow } from "./components/ReasonFlow";
import { TRIAL_CHECKOUT_REASONS, UPGRADE_CHECKOUT_REASONS } from "./helpers";

type Kind = "trial" | "upgrade";

const COPY: Record<Kind, { title: string; body: string; thanks: string }> = {
  trial: {
    title: "Your free trial is ready when you are",
    body: "Anything holding you back?",
    thanks:
      "Thanks. You can start your trial from Billing whenever you're ready.",
  },
  upgrade: {
    title: "Your plan hasn't changed. What held you back?",
    body: "One tap helps us fix the right thing.",
    thanks:
      "Thanks. If your card was the problem, try another one or reply to any of our emails and we'll sort it out.",
  },
};

interface Props {
  kind: Kind;
  initialReason?: string | null;
}

export function CheckoutAbandonedDialog({ kind, initialReason = null }: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const copy = COPY[kind];

  return (
    <Dialog
      title={copy.title}
      styling={{ maxWidth: "38rem", minWidth: "auto" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Content>
        <NoInitialFocus />
        <div className="flex flex-col gap-4">
          <Text variant="body" className="!text-foreground-tertiary">
            {copy.body}
          </Text>
          <ReasonFlow
            label={copy.title}
            reasons={
              kind === "trial"
                ? TRIAL_CHECKOUT_REASONS
                : UPGRADE_CHECKOUT_REASONS
            }
            initialReason={initialReason}
            thanks={copy.thanks}
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
