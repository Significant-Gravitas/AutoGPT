"use client";

import { AlertTriangleIcon } from "lucide-react";
import Link from "next/link";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { TopUpForm } from "../TopUpForm/TopUpForm";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  // "out-of-credits" is the reactive nudge shown when the balance hits zero;
  // "add-credits" is the same form opened deliberately from the wallet.
  variant?: "out-of-credits" | "add-credits";
}

export function TopUpDialog({
  isOpen,
  onClose,
  variant = "out-of-credits",
}: Props) {
  const isOutOfCredits = variant === "out-of-credits";

  function handleOpenChange(open: boolean) {
    if (!open) onClose();
  }

  return (
    <Dialog
      title={
        isOutOfCredits ? (
          <span className="inline-flex items-center gap-2">
            <AlertTriangleIcon className="h-[1.125rem] w-[1.125rem] text-orange-600" />
            You&apos;re out of automation credits
          </span>
        ) : (
          "Add automation credits"
        )
      }
      styling={{ maxWidth: "28rem" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Content>
        <Text variant="large">
          {isOutOfCredits
            ? "Top up to keep your agents and Autopilot running. You can also "
            : "Credits are used to run your agents and Autopilot. You can also "}
          <Link href="/settings/billing" className="underline">
            enable auto-refill in billing settings
          </Link>
          .
        </Text>
        <TopUpForm submitLabel="Top up" />
      </Dialog.Content>
    </Dialog>
  );
}
