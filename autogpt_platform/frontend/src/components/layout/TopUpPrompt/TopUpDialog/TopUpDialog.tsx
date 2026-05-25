"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { TopUpForm } from "../TopUpForm/TopUpForm";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function TopUpDialog({ isOpen, onClose }: Props) {
  return (
    <Dialog
      title="You're out of automation credits"
      styling={{ maxWidth: "28rem" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          Top up to keep your agents and Autopilot running.
        </Text>
        <TopUpForm submitLabel="Top up" />
      </Dialog.Content>
    </Dialog>
  );
}
