"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function RestartConfirmDialog({ open, onOpenChange, onConfirm }: Props) {
  return (
    <Dialog
      title="Start over?"
      styling={{ maxWidth: "26rem" }}
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-800">
            This clears every answer you have given so far and takes you back to
            the first question. It cannot be undone.
          </Text>

          <div className="flex justify-end gap-2 pt-2">
            <Button
              variant="secondary"
              size="small"
              onClick={() => onOpenChange(false)}
            >
              Keep going
            </Button>
            <Button variant="destructive" size="small" onClick={onConfirm}>
              Start over
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
