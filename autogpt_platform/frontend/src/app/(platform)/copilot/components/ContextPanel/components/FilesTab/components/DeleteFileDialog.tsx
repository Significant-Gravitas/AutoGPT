import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  fileName: string | null;
  isDeleting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteFileDialog({
  fileName,
  isDeleting,
  onConfirm,
  onCancel,
}: Props) {
  return (
    <Dialog
      title="Delete file"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen: !!fileName,
        set: async (open) => {
          if (!open && !isDeleting) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          Delete{" "}
          <Text variant="body-medium" as="span">
            &quot;{fileName}&quot;
          </Text>
          ? This removes it from the workspace and cannot be undone.
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
