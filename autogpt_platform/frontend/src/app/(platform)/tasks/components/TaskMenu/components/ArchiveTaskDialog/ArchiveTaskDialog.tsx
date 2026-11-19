import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  title: string;
  isPending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ArchiveTaskDialog({
  isOpen,
  title,
  isPending,
  onCancel,
  onConfirm,
}: Props) {
  return (
    <Dialog
      title="Archive this task?"
      variant="compact"
      className="max-w-md max-lg:inset-x-0 max-lg:mx-auto"
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body" tone="text" className="break-words">
          “{title}” leaves your task list.
        </Text>
        <Dialog.Footer>
          <Button variant="ghost-muted" size="small" onClick={onCancel}>
            Keep it
          </Button>
          <Button
            variant="destructive"
            size="small"
            loading={isPending}
            onClick={onConfirm}
          >
            Archive task
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
