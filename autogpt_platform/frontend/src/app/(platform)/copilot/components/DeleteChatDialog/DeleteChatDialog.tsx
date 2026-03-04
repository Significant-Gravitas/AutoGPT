"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  session: { id: string; title: string | null | undefined } | null;
  isDeleting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteChatDialog({
  session,
  isDeleting,
  onConfirm,
  onCancel,
}: Props) {
  return (
    <Dialog
      title="Delete chat"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen: !!session,
        set: async (open) => {
          if (!open && !isDeleting) {
            onCancel();
          }
        },
      }}
      onClose={isDeleting ? undefined : onCancel}
    >
      <Dialog.Content>
        <Text variant="body">
          Are you sure you want to delete{" "}
          <Text variant="body-medium" as="span">
            &quot;{session?.title || "Untitled chat"}&quot;
          </Text>
          ? This action cannot be undone.
        </Text>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            loading={isDeleting}
          >
            Delete
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
