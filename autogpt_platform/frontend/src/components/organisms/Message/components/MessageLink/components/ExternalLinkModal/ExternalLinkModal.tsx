import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  url: string;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

export function ExternalLinkModal({ url, isOpen, onClose, onConfirm }: Props) {
  return (
    <Dialog
      title="Open external link"
      styling={{ maxWidth: "30rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          You&apos;re about to visit an external website:
        </Text>
        <div className="mt-2 rounded-md bg-surface-hover p-3">
          <Text variant="small-mono" className="break-all">
            {url}
          </Text>
        </div>
        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm}>
            Open link
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
