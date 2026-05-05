"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  targetTierLabel: string;
  body: string;
  isSaving: boolean;
  onConfirm: () => void;
}

export function SwitchTierDialog({
  isOpen,
  onOpenChange,
  targetTierLabel,
  body,
  isSaving,
  onConfirm,
}: Props) {
  return (
    <Dialog
      title={`Upgrade to ${targetTierLabel}?`}
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" as="span" className="text-zinc-700">
            {body}
          </Text>
        </div>

        <Dialog.Footer>
          <Button
            type="button"
            variant="ghost"
            size="small"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            size="small"
            onClick={onConfirm}
            disabled={isSaving}
            loading={isSaving}
          >
            Upgrade to {targetTierLabel}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
