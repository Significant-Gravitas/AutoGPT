"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function CancelRecordingDialog({
  isOpen,
  onOpenChange,
  onConfirm,
}: Props) {
  return (
    <Dialog
      title="Discard recording?"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <Text variant="body">
          This permanently deletes your current take. You can keep recording or
          discard it and start again.
        </Text>
        <Dialog.Footer>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Keep recording
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            Discard recording
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
