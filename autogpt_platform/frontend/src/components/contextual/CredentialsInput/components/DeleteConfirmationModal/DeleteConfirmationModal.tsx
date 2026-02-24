import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  credentialToDelete: { id: string; title: string } | null;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

export function DeleteConfirmationModal({
  credentialToDelete,
  isDeleting,
  onClose,
  onConfirm,
}: Props) {
  return (
    <Dialog
      controlled={{
        isOpen: credentialToDelete !== null,
        set: (open) => {
          if (!open) onClose();
        },
      }}
      title="Delete credential"
      styling={{ maxWidth: "32rem" }}
    >
      <Dialog.Content>
        <Text variant="large">
          Are you sure you want to delete &quot;{credentialToDelete?.title}
          &quot;? This action cannot be undone.
        </Text>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose} disabled={isDeleting}>
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
