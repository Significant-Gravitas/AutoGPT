"use client";

import { Alert02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import Link from "next/link";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  TOP_UP_QUESTION,
  TopUpQuestion,
} from "@/components/organisms/FeedbackPrompts/TopUpQuestion";

import { TopUpForm } from "../TopUpForm/TopUpForm";
import { useTopUpDialog } from "./useTopUpDialog";

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
  const dialog = useTopUpDialog({ onClose });

  function handleOpenChange(open: boolean) {
    if (!open) dialog.close();
  }

  return (
    <Dialog
      title={
        dialog.isAsking ? (
          TOP_UP_QUESTION
        ) : isOutOfCredits ? (
          <span className="inline-flex items-center gap-2">
            <span className="flex text-orange-foreground">
              <Icon
                aria-hidden
                icon={Alert02Icon}
                className="h-[1.125rem] w-[1.125rem]"
              />
            </span>
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
        {dialog.isAsking ? (
          <TopUpQuestion
            onReason={dialog.pickReason}
            onSubmit={dialog.answer}
          />
        ) : (
          <>
            <TopUpOffer isOutOfCredits={isOutOfCredits} />
            {isOutOfCredits ? (
              <div className="flex justify-center">
                <Button
                  variant="ghost"
                  size="small"
                  loading={dialog.isClaiming}
                  disabled={dialog.isClaiming}
                  onClick={dialog.notNow}
                >
                  Not now
                </Button>
              </div>
            ) : null}
          </>
        )}
      </Dialog.Content>
    </Dialog>
  );
}

interface OfferProps {
  isOutOfCredits: boolean;
}

function TopUpOffer({ isOutOfCredits }: OfferProps) {
  return (
    <>
      <Text variant="large">
        {isOutOfCredits
          ? "Top up to keep your workflows and experts running. You can also "
          : "Credits are used to run your workflows and experts. You can also "}
        <Link href="/settings/billing" className="underline">
          enable auto-refill in billing settings
        </Link>
        .
      </Text>
      <TopUpForm submitLabel="Top up" />
    </>
  );
}
