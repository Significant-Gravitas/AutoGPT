"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

const DEFAULT_DESCRIPTION =
  "AutoPilot will not ask before anything in this chat. Edits, commands and actions outside the platform, like sending a message or an email, run without your approval.";

interface Props {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title?: string;
  description?: string;
}

export function UnsupervisedConfirmDialog({
  isOpen,
  onConfirm,
  onCancel,
  title = "Run this chat unsupervised?",
  description = DEFAULT_DESCRIPTION,
}: Props) {
  return (
    <Dialog
      title={title}
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">{description}</Text>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm}>
            Run unsupervised
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
