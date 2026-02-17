"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface DeleteChatDialogProps {
  /** The session to delete, or null if dialog should be closed */
  session: { id: string; title: string | null | undefined } | null;
  /** Whether deletion is in progress */
  isDeleting: boolean;
  /** Called when user confirms deletion */
  onConfirm: () => void;
  /** Called when user cancels (only works when not deleting) */
  onCancel: () => void;
}

export function DeleteChatDialog({
  session,
  isDeleting,
  onConfirm,
  onCancel,
}: DeleteChatDialogProps) {
  return (
    <Dialog
      title="Delete chat"
      controlled={{
        isOpen: !!session,
        set: async (open) => {
          if (!open && !isDeleting) {
            onCancel();
          }
        },
      }}
      onClose={onCancel}
    >
      <Dialog.Content>
        <p className="text-neutral-600">
          Are you sure you want to delete{" "}
          <span className="font-medium">
            &quot;{session?.title || "Untitled chat"}&quot;
          </span>
          ? This action cannot be undone.
        </p>
        <Dialog.Footer>
          <Button
            variant="ghost"
            size="small"
            onClick={onCancel}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            onClick={onConfirm}
            loading={isDeleting}
            className="bg-red-600 hover:bg-red-700"
          >
            Delete
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
