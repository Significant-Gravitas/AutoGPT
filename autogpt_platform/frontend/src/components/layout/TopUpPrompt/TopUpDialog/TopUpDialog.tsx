"use client";

import { AlertTriangleIcon } from "lucide-react";
import Link from "next/link";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { TopUpForm } from "../TopUpForm/TopUpForm";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function TopUpDialog({ isOpen, onClose }: Props) {
  function handleOpenChange(open: boolean) {
    if (!open) onClose();
  }

  return (
    <Dialog
      title={
        <span className="inline-flex items-center gap-2">
          <AlertTriangleIcon className="h-[1.125rem] w-[1.125rem] text-orange-600" />
          You&apos;re out of automation credits
        </span>
      }
      styling={{ maxWidth: "28rem" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Content>
        <Text variant="large">
          Top up to keep your agents and Autopilot running. You can also{" "}
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
