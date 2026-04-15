import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  credentialToDelete: { id: string; title: string } | null;
  warningMessage?: string | null;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onForceConfirm: () => void;
}

export function DeleteConfirmationModal({
  credentialToDelete,
  warningMessage,
  isDeleting,
  onClose,
  onConfirm,
  onForceConfirm,
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
        {warningMessage ? (
          <Text variant="large">{warningMessage}</Text>
        ) : (
          <Text variant="large">
            Are you sure you want to delete &quot;{credentialToDelete?.title}
            &quot;? This action cannot be undone.
          </Text>
        )}
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          {warningMessage ? (
            <Button
              variant="destructive"
              onClick={onForceConfirm}
              loading={isDeleting}
            >
              Force Delete
            </Button>
          ) : (
            <Button
              variant="destructive"
              onClick={onConfirm}
              loading={isDeleting}
            >
              Delete
            </Button>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
